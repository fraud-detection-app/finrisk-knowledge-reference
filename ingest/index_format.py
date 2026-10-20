"""The index format -- the contract with the risk service (spec 3.4).

    indexes/<version>/
        manifest.json         version, created, source_sha, embeddings, embedding_model, dims, docs, chunks, chunking
        chunks.jsonl          one JSON object per line: id, doc_path, section, text, sha256
        embeddings.npy        float32 (chunks, dims); row i belongs to line i of chunks.jsonl      [titan, hash]
        tfidf_vectorizer.joblib + tfidf_matrix.joblib                                              [tfidf]
        eval_report.json      written by eval/evaluate.py before publishing (the service ignores it)
        build_plan.json       what the build reused / embedded / dropped (the service ignores it)

``manifest.json`` is uploaded LAST, so an index whose manifest exists is complete."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from ingest.chunking import Chunk
from ingest.store import Store

MANIFEST = "manifest.json"
CHUNKS = "chunks.jsonl"
EMBEDDINGS = "embeddings.npy"
TFIDF_VECTORIZER = "tfidf_vectorizer.joblib"
TFIDF_MATRIX = "tfidf_matrix.joblib"
EVAL_REPORT = "eval_report.json"
BUILD_PLAN = "build_plan.json"
# manifest last; everything else first
UPLOAD_ORDER = (CHUNKS, EMBEDDINGS, TFIDF_VECTORIZER, TFIDF_MATRIX, EVAL_REPORT, BUILD_PLAN, MANIFEST)
MANIFEST_KEYS = ("version", "created", "source_sha", "embeddings", "embedding_model", "dims", "docs", "chunks")


@dataclass
class IndexData:
    manifest: dict
    chunks: list[dict]
    embeddings: np.ndarray | None = None
    tfidf_vectorizer: object | None = None
    tfidf_matrix: object | None = None


def chunks_to_jsonl(chunks: list[Chunk] | list[dict]) -> str:
    rows = [c.row() if isinstance(c, Chunk) else c for c in chunks]
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def write_index(out_dir: Path, manifest: dict, chunks: list[Chunk] | list[dict],
                embeddings: np.ndarray | None = None, tfidf: tuple[object, object] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CHUNKS).write_text(chunks_to_jsonl(chunks), encoding="utf-8")
    if embeddings is not None:
        np.save(out_dir / EMBEDDINGS, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
    if tfidf is not None:
        joblib.dump(tfidf[0], out_dir / TFIDF_VECTORIZER)
        joblib.dump(tfidf[1], out_dir / TFIDF_MATRIX)
    ordered = {key: manifest[key] for key in MANIFEST_KEYS}
    ordered.update({k: v for k, v in manifest.items() if k not in ordered})
    (out_dir / MANIFEST).write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_index(store: Store) -> IndexData:
    manifest = json.loads(store.read(MANIFEST))
    chunks = [json.loads(line) for line in store.read(CHUNKS).decode("utf-8").splitlines() if line.strip()]
    data = IndexData(manifest=manifest, chunks=chunks)
    if manifest["embeddings"] == "tfidf":
        data.tfidf_vectorizer = joblib.load(io.BytesIO(store.read(TFIDF_VECTORIZER)))
        data.tfidf_matrix = joblib.load(io.BytesIO(store.read(TFIDF_MATRIX)))
    else:
        data.embeddings = np.load(io.BytesIO(store.read(EMBEDDINGS)), allow_pickle=False)
    return data


def check_index(data: IndexData) -> list[str]:
    """Problems that would make the service reject this index. Empty list = the format holds."""
    problems: list[str] = []
    manifest = data.manifest
    for key in MANIFEST_KEYS:
        if key not in manifest:
            problems.append(f"manifest is missing '{key}'")
    if problems:
        return problems
    if manifest["chunks"] != len(data.chunks):
        problems.append(f"manifest says {manifest['chunks']} chunks, chunks.jsonl has {len(data.chunks)}")
    for row in data.chunks:
        if list(row) != ["id", "doc_path", "section", "text", "sha256"]:
            problems.append(f"chunk {row.get('id')} has the wrong keys {list(row)}")
            break
    per_doc: dict[str, int] = {}
    for row in data.chunks:
        per_doc[row["doc_path"]] = per_doc.get(row["doc_path"], 0) + 1
    if per_doc != {path: info["chunks"] for path, info in manifest["docs"].items()}:
        problems.append("manifest docs/chunks counts do not match chunks.jsonl")
    if manifest["embeddings"] == "tfidf":
        if data.tfidf_matrix is None or data.tfidf_matrix.shape[0] != len(data.chunks):
            problems.append("tfidf matrix rows do not match chunks.jsonl")
    else:
        e = data.embeddings
        if e is None:
            problems.append("embeddings.npy is missing")
        else:
            if e.dtype != np.float32:
                problems.append(f"embeddings dtype is {e.dtype}, expected float32")
            if e.ndim != 2 or e.shape != (len(data.chunks), manifest["dims"]):
                problems.append(f"embeddings shape {e.shape}, expected {(len(data.chunks), manifest['dims'])}")
    return problems
