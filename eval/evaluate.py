"""Score an index against eval/questions.yaml, and decide whether it may be published.

    python -m eval.evaluate --index build/index [--baseline ssm:/finrisk/prod/index]

Metrics (same definitions as the risk service's week-9 evaluation):
  recall@5   share of answerable questions whose expected document is in the top 5 retrieved chunks
  MRR        mean of 1/rank of the first correct chunk (0 when it never appears)
  refusals   every must_refuse question is refused by the compliance patterns, and no answerable one is

The gate (exit code 3 when it fails):
  1. recall@5 >= 0.80
  2. recall@5 has not dropped by more than 0.05 against the report of the index currently in prod
  3. every refusal check holds
The report is written to eval_report.json next to the index, so it is published with it and becomes
the baseline for the next build."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from eval.policy import is_out_of_policy
from ingest.index_format import EVAL_REPORT
from ingest.reader import IndexReader
from ingest.safe import say
from ingest.store import INDEXES, S3Store, parse_s3_uri

K = 5
MIN_RECALL_AT_5 = 0.80
MAX_DROP = 0.05
EXIT_GATE_FAILED = 3
DEFAULT_QUESTIONS = Path(__file__).with_name("questions.yaml")
PROD_POINTER = "/finrisk/prod/index"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expected_docs: tuple[str, ...] = ()
    must_refuse: bool = False


@dataclass
class Baseline:
    report: dict | None
    status: str                  # "ok" | "none" (nothing exists yet) | "unreadable"
    note: str = ""
    source: str = ""


def load_questions(path: Path | str = DEFAULT_QUESTIONS) -> list[Question]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["questions"]
    questions = []
    for item in raw:
        refuse = bool(item.get("must_refuse", False))
        expected = tuple(item.get("expected_docs", []))
        if refuse == bool(expected):
            raise ValueError(f"{item.get('id')}: give either expected_docs or must_refuse: true")
        questions.append(Question(item["id"], item["question"], expected, refuse))
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("question ids must be unique")
    return questions


def reciprocal_rank(docs: list[str], expected: tuple[str, ...]) -> float:
    for rank, doc in enumerate(docs, start=1):
        if doc in expected:
            return 1.0 / rank
    return 0.0


def evaluate_index(reader: IndexReader, questions: list[Question], k: int = K) -> dict:
    rows, hits, rr_total, answerable = [], 0, 0.0, 0
    refuse_expected = refuse_ok = false_refusals = 0
    for q in questions:
        refused = is_out_of_policy(q.question)
        if q.must_refuse:
            refuse_expected += 1
            refuse_ok += int(refused)
            rows.append({"id": q.id, "question": q.question, "kind": "must_refuse", "refused": refused, "ok": refused})
            continue
        answerable += 1
        false_refusals += int(refused)
        top = reader.search(q.question, k)
        docs = [h.doc_path for h in top]
        hit = any(d in q.expected_docs for d in docs[:k])
        rr = reciprocal_rank(docs, q.expected_docs)
        hits += int(hit)
        rr_total += rr
        rows.append({"id": q.id, "question": q.question, "kind": "answer", "expected_docs": list(q.expected_docs),
                     "top_docs": docs, "hit_at_5": hit, "rr": round(rr, 4), "refused": refused, "ok": hit and not refused})
    return {
        "index_version": reader.version,
        "embeddings": reader.data.manifest["embeddings"],
        "questions": len(questions),
        "answerable": answerable,
        "recall_at_5": round(hits / answerable, 4) if answerable else 0.0,
        "mrr": round(rr_total / answerable, 4) if answerable else 0.0,
        "refusals": {"expected": refuse_expected, "refused": refuse_ok, "false_refusals": false_refusals},
        "per_question": rows,
    }


def apply_gate(metrics: dict, baseline: Baseline | None, min_recall: float = MIN_RECALL_AT_5,
               max_drop: float = MAX_DROP, require_baseline: bool = False) -> dict:
    checks = []
    recall = metrics["recall_at_5"]
    checks.append({"name": "recall_at_5_floor", "ok": recall >= min_recall,
                   "detail": f"recall@5 {recall:.4f} (floor {min_recall:.2f})"})

    base_recall = None
    if baseline is not None and baseline.report is not None:
        base_recall = baseline.report.get("recall_at_5")
    if base_recall is not None:
        drop = round(base_recall - recall, 4)
        checks.append({"name": "no_drop_vs_prod", "ok": drop <= max_drop,
                       "detail": f"prod {base_recall:.4f} -> {recall:.4f} (drop {drop:+.4f}, allowed {max_drop:.2f})"})
    elif baseline is not None and baseline.status == "unreadable":
        checks.append({"name": "no_drop_vs_prod", "ok": not require_baseline,
                       "detail": f"WARNING: prod baseline unreadable ({baseline.note}); only the floor was checked"})
    else:
        why = baseline.note if baseline is not None and baseline.note else "no baseline given"
        checks.append({"name": "no_drop_vs_prod", "ok": True, "detail": f"skipped: {why}"})

    refusals = metrics["refusals"]
    refusal_ok = refusals["refused"] == refusals["expected"] and refusals["false_refusals"] == 0
    checks.append({"name": "refusals", "ok": refusal_ok,
                   "detail": f"{refusals['refused']}/{refusals['expected']} must-refuse questions refused, "
                             f"{refusals['false_refusals']} answerable ones wrongly refused"})
    return {
        "min_recall_at_5": min_recall,
        "max_drop": max_drop,
        "baseline": ({"version": baseline.report.get("index_version"), "recall_at_5": base_recall,
                      "source": baseline.source} if base_recall is not None else None),
        "baseline_status": baseline.status if baseline is not None else "not requested",
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
    }


# ------------------------------------------------------------------------------ baselines


def _report_from_store(store: S3Store) -> Baseline:
    if not store.exists(EVAL_REPORT):
        return Baseline(None, "none", f"{store} has no eval_report.json (published before the gate existed)", str(store))
    return Baseline(json.loads(store.read(EVAL_REPORT)), "ok", source=str(store))


def load_baseline(spec: str | None, ssm=None, s3=None) -> Baseline | None:
    """``spec``: a report file, an index folder, ``s3://.../indexes/<v>/`` or ``ssm:<pointer parameter>``."""
    if not spec:
        return None
    if spec.startswith("ssm:") or spec.startswith("s3://"):
        try:
            if spec.startswith("ssm:"):
                if ssm is None:
                    import boto3
                    ssm = boto3.client("ssm")
                pointer = json.loads(ssm.get_parameter(Name=spec[4:])["Parameter"]["Value"])
                if not pointer.get("uri"):
                    return Baseline(None, "none", f"{spec[4:]} points at no index yet (first release)", spec)
                uri = pointer["uri"]
            else:
                uri = spec
            bucket, key = parse_s3_uri(uri)
            if s3 is None:
                import boto3
                s3 = boto3.client("s3")
            return _report_from_store(S3Store(s3, bucket, key))
        except Exception as exc:                                                       # noqa: BLE001
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
            if code == "ParameterNotFound":
                return Baseline(None, "none", f"{spec[4:]} does not exist yet", spec)
            return Baseline(None, "unreadable", f"{type(exc).__name__} {code}".strip(), spec)
    path = Path(spec)
    path = path / EVAL_REPORT if path.is_dir() else path
    if not path.is_file():
        return Baseline(None, "none", f"{path} does not exist", spec)
    return Baseline(json.loads(path.read_text(encoding="utf-8")), "ok", source=str(path))


# ------------------------------------------------------------------------------ CLI


def print_report(report: dict) -> None:
    say(f"index {report['index_version']} ({report['embeddings']}): {report['questions']} questions, {report['answerable']} answerable")
    for row in report["per_question"]:
        if row["kind"] == "must_refuse":
            say(f"  {'ok  ' if row['ok'] else 'FAIL'} {row['id']}  refused={row['refused']}  {row['question']}")
        else:
            top = row["top_docs"][0] if row["top_docs"] else "-"
            say(f"  {'ok  ' if row['ok'] else 'FAIL'} {row['id']}  hit@5={str(row['hit_at_5']):5}  rr={row['rr']:.2f}  top={top}  {row['question']}")
    say(f"recall@5 {report['recall_at_5']:.4f}   MRR {report['mrr']:.4f}")
    for check in report["gate"]["checks"]:
        say(f"  gate {'ok  ' if check['ok'] else 'FAIL'} {check['name']}: {check['detail']}")
    say("GATE " + ("PASSED" if report["passed"] else "FAILED"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.evaluate", description=__doc__.split("\n")[0])
    parser.add_argument("--index", required=True, help="folder holding the built index")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--baseline", default=None, help=f"report file | index folder | s3://.../indexes/<v>/ | ssm:{PROD_POINTER}")
    parser.add_argument("--out", default=None, help=f"default: <index>/{EVAL_REPORT}")
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL_AT_5)
    parser.add_argument("--max-drop", type=float, default=MAX_DROP)
    parser.add_argument("--require-baseline", action="store_true", help="fail when the prod report exists but cannot be read")
    args = parser.parse_args(argv)

    reader = IndexReader.open(args.index)
    metrics = evaluate_index(reader, load_questions(args.questions))
    baseline = load_baseline(args.baseline)
    gate = apply_gate(metrics, baseline, args.min_recall, args.max_drop, args.require_baseline)
    report = {**metrics, "gate": gate, "passed": gate["passed"]}
    out = Path(args.out) if args.out else Path(args.index) / EVAL_REPORT
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_report(report)
    say(f"report: {out}")
    return 0 if report["passed"] else EXIT_GATE_FAILED


if __name__ == "__main__":
    sys.exit(main())
