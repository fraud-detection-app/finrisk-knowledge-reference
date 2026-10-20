"""Publish a built index to ``s3://<kb-bucket>/indexes/<version>/`` -- and NEVER overwrite one.

An index version is immutable: a pointer names a version, a rollback is "point at the older one".
That only works if a published version can never change underneath a pointer. Two guards:

1. the prefix must be empty before the first upload (clear error message);
2. every object is written with ``IfNoneMatch="*"``, so S3 itself refuses to replace an existing key
   even when two publishes race.

Files are uploaded in ``UPLOAD_ORDER`` with manifest.json LAST: if the manifest exists, the index is complete."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ingest.embeddings import PUBLISHABLE
from ingest.index_format import EVAL_REPORT, MANIFEST, UPLOAD_ORDER
from ingest.store import INDEXES, check_version, index_uri, parse_s3_uri

CONTENT_TYPES = {".json": "application/json", ".jsonl": "application/x-ndjson", ".npy": "application/octet-stream",
                 ".joblib": "application/octet-stream"}


class PublishError(Exception):
    pass


def _check_ready(local_dir: Path, require_gate: bool) -> dict:
    if not (local_dir / MANIFEST).is_file():
        raise PublishError(f"{local_dir} has no manifest.json: build first")
    manifest = json.loads((local_dir / MANIFEST).read_text(encoding="utf-8"))
    check_version(manifest["version"])
    if manifest["embeddings"] not in PUBLISHABLE:
        raise PublishError(f"embeddings={manifest['embeddings']!r} is an offline stand-in and is never published "
                           f"(publishable: {', '.join(PUBLISHABLE)})")
    report = local_dir / EVAL_REPORT
    if report.is_file():
        if not json.loads(report.read_text(encoding="utf-8")).get("passed"):
            raise PublishError("eval_report.json says the gate FAILED: not publishing")
    elif require_gate:
        raise PublishError("no eval_report.json: run `python -m eval.evaluate` (the gate) before publishing")
    return manifest


def publish_s3(local_dir: Path | str, bucket: str, s3_client, require_gate: bool = True) -> str:
    local_dir = Path(local_dir)
    manifest = _check_ready(local_dir, require_gate)
    prefix = f"{INDEXES}{manifest['version']}/"
    uri = index_uri(bucket, manifest["version"])
    if s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1).get("Contents"):
        raise PublishError(f"{uri} already exists: published indexes are never overwritten (build a new version)")
    for name in UPLOAD_ORDER:
        path = local_dir / name
        if not path.is_file():
            continue
        try:
            s3_client.put_object(Bucket=bucket, Key=prefix + name, Body=path.read_bytes(), IfNoneMatch="*",
                                 ContentType=CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        except Exception as exc:                                           # noqa: BLE001
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
            if code in {"PreconditionFailed", "ConditionalRequestConflict"}:
                raise PublishError(f"{uri}{name} already exists: published indexes are never overwritten") from exc
            raise
    return uri


def publish_local(local_dir: Path | str, dest: Path | str, require_gate: bool = True) -> str:
    local_dir, dest = Path(local_dir), Path(dest)
    _check_ready(local_dir, require_gate)
    if dest.exists() and any(dest.iterdir()):
        raise PublishError(f"{dest} already exists and is not empty: published indexes are never overwritten")
    dest.mkdir(parents=True, exist_ok=True)
    for name in UPLOAD_ORDER:
        if (local_dir / name).is_file():
            shutil.copy2(local_dir / name, dest / name)
    return str(dest)


def publish(local_dir: Path | str, dest: str, s3_client=None, require_gate: bool = True) -> str:
    """``dest`` is ``s3://<bucket>/indexes/<version>/`` or a local folder."""
    if dest.startswith("s3://"):
        bucket, key = parse_s3_uri(dest)
        manifest = _check_ready(Path(local_dir), require_gate)
        if key.rstrip("/") != f"{INDEXES}{manifest['version']}".rstrip("/"):
            raise PublishError(f"{dest} does not match the index version {manifest['version']}")
        if s3_client is None:
            import boto3
            s3_client = boto3.client("s3")
        return publish_s3(local_dir, bucket, s3_client, require_gate)
    return publish_local(local_dir, dest, require_gate)
