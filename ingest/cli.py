"""python -m ingest build | publish | promote | latest | diff | lint | summary"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ingest import build as build_mod
from ingest import publish as publish_mod
from ingest import promote as promote_mod
from ingest.chunking import MAX_WORDS, OVERLAP, Chunking
from ingest.docs import DocError, lint_docs
from ingest.embeddings import PROVIDERS
from ingest.index_format import MANIFEST
from ingest.safe import mask, say
from ingest.store import (INDEXES, StoreError, index_uri, latest_s3_version, resolve_store, VERSION_RE)
from ingest.summary import render_files

REGION = "ap-southeast-2"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or default


def _s3():
    import boto3
    return boto3.client("s3", region_name=_env("AWS_REGION", REGION))


def _ssm():
    import boto3
    return boto3.client("ssm", region_name=_env("AWS_REGION", REGION))


def _add_build_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--docs", default="docs", help="folder with the markdown documents")
    p.add_argument("--embeddings", default=_env("EMBEDDINGS", "tfidf"), choices=PROVIDERS,
                   help="titan (Bedrock) | tfidf (always a full rebuild) | hash (offline stand-in, never published). "
                        "Default: $EMBEDDINGS or tfidf")
    p.add_argument("--previous", default=None,
                   help="index to build on: a folder, s3://<kb>/indexes/<version>/, or 'latest' (needs --bucket)")
    p.add_argument("--bucket", default=_env("KB_BUCKET"), help="knowledge bucket (default: $KB_BUCKET)")
    p.add_argument("--max-words", type=int, default=MAX_WORDS)
    p.add_argument("--overlap", type=int, default=OVERLAP)


def _previous(args: argparse.Namespace):
    store = resolve_store(args.previous, bucket=args.bucket, s3_client=_s3() if (args.previous or "").startswith(("s3://", "latest")) else None)
    if store is None:
        say("previous: none (first build)" if args.previous else "previous: not given (building from scratch)")
        return None
    data = build_mod.load_previous(store)
    say(f"previous: {data.manifest['version']} ({data.manifest['embeddings']}, {data.manifest['chunks']} chunks)")
    return data


def cmd_build(args: argparse.Namespace) -> int:
    chunking = Chunking(args.max_words, args.overlap)
    previous = _previous(args)
    say(f"embeddings: {args.embeddings}")
    result = build_mod.build_index(args.docs, args.out, args.embeddings, previous=previous, chunking=chunking)
    say(result.plan.line())
    if result.plan.reason:
        say(f"  why: {result.plan.reason}")
    say(f"built {result.version}: {result.manifest['chunks']} chunks, {len(result.manifest['docs'])} docs, "
        f"{result.manifest['embedding_model']} ({result.manifest['dims']} dims) -> {args.out}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    previous = _previous(args)
    plan = build_mod.plan_only(args.docs, args.embeddings, previous, Chunking(args.max_words, args.overlap))
    say(plan.line())
    for sign, paths in (("+", plan.added), ("~", plan.changed), ("-", plan.removed)):
        for path in paths:
            say(f"  {sign} {path}")
    if plan.reason:
        say(f"  why: {plan.reason}")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    manifest = json.loads((Path(args.source) / MANIFEST).read_text(encoding="utf-8"))
    dest = args.dest
    if not dest:
        if not args.bucket:
            raise publish_mod.PublishError("give --bucket (or $KB_BUCKET) or --dest")
        dest = index_uri(args.bucket, manifest["version"])
    uri = publish_mod.publish(args.source, dest, s3_client=_s3() if dest.startswith("s3://") else None,
                              require_gate=not args.no_gate)
    say(f"published {manifest['version']} -> {uri} (never overwritten)")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    if not args.bucket:
        raise promote_mod.PromoteError("give --bucket or set KB_BUCKET")
    promote_mod.promote(args.env, args.version, args.bucket, ssm=_ssm(), s3=_s3(), run=args.run, wait=not args.no_wait,
                        summary_path=args.summary, timeout=args.timeout, interval=args.interval, grace=args.grace)
    return 0


def cmd_latest(args: argparse.Namespace) -> int:
    if args.dir:
        base = Path(args.dir)
        versions = sorted(p.name for p in base.iterdir() if p.is_dir() and VERSION_RE.match(p.name) and (p / MANIFEST).is_file()) if base.is_dir() else []
        version = versions[-1] if versions else None
    else:
        if not args.bucket:
            raise StoreError("give --bucket (or $KB_BUCKET) or --dir")
        version = latest_s3_version(_s3(), args.bucket)
    if not version:
        print("no published index", file=sys.stderr)
        return 1
    print(version)
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    problems = lint_docs(args.docs)
    for problem in problems:
        say(f"  {problem}")
    say(f"front matter: {'OK' if not problems else str(len(problems)) + ' problem(s)'}")
    return 1 if problems else 0


def cmd_summary(args: argparse.Namespace) -> int:
    text = render_files(args.plan, args.report, args.baseline, args.title, args.baseline_label)
    print(mask(text))
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m ingest", description="Build, publish and promote the knowledge index.")
    sub = top.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="documents -> a local index folder (incremental when --previous is given)")
    _add_build_options(p)
    p.add_argument("--out", default="build/index")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("diff", help="print the incremental plan without building or embedding anything")
    _add_build_options(p)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("publish", help="upload a built, evaluated index to s3://<kb>/indexes/<version>/ (never overwrites)")
    p.add_argument("--from", dest="source", default="build/index")
    p.add_argument("--bucket", default=_env("KB_BUCKET"))
    p.add_argument("--dest", default=None, help="s3://<kb>/indexes/<version>/ or a local folder (CI, tests)")
    p.add_argument("--no-gate", action="store_true", help="allow an index without eval_report.json (local use only)")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("promote", help="point an environment at an index version and wait until the app serves it")
    p.add_argument("--env", required=True, choices=promote_mod.ENVS)
    p.add_argument("--version", required=True)
    p.add_argument("--bucket", default=_env("KB_BUCKET"))
    p.add_argument("--run", default=_env("RUN_URL", "local"), help="URL of the workflow run (recorded in the pointer)")
    p.add_argument("--no-wait", action="store_true", help="write the pointer only; do not poll <app_url>/ready")
    p.add_argument("--timeout", type=float, default=promote_mod.READY_TIMEOUT_SECONDS)
    p.add_argument("--interval", type=float, default=promote_mod.POLL_SECONDS)
    p.add_argument("--grace", type=float, default=promote_mod.GRACE_SECONDS)
    p.add_argument("--summary", default=_env("GITHUB_STEP_SUMMARY"))
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("latest", help="print the newest published version")
    p.add_argument("--bucket", default=_env("KB_BUCKET"))
    p.add_argument("--dir", default=None, help="a local folder of versions instead of S3")
    p.set_defaults(func=cmd_latest)

    p = sub.add_parser("lint", help="check every document's front matter")
    p.add_argument("--docs", default="docs")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("summary", help="markdown for $GITHUB_STEP_SUMMARY")
    p.add_argument("--plan", default=None)
    p.add_argument("--report", default=None)
    p.add_argument("--baseline", default=None)
    p.add_argument("--title", default="Knowledge base check")
    p.add_argument("--baseline-label", default="main")
    p.set_defaults(func=cmd_summary)
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (DocError, StoreError, ValueError, publish_mod.PublishError, promote_mod.PromoteError) as exc:
        say(f"ERROR: {exc}")
        return 2
