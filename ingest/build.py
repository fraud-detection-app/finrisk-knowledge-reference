"""Build one index version from the documents, reusing the previous index where it can.

Algorithm (embedding providers: titan, hash):
  1. load + clean + chunk every document (cheap);
  2. compare each document's sha256 with the previous manifest -> added / changed / removed / unchanged;
  3. unchanged documents keep their chunk rows AND their embedding rows;
  4. changed documents lose their old chunks; added + changed documents are embedded in one batch;
  5. write manifest, chunks, embeddings. tfidf skips 3-4: it is always rebuilt from all chunks."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ingest.chunking import Chunk, Chunking, chunk_document
from ingest.docs import DocError, load_docs
from ingest.embeddings import PROVIDERS, get_embedder
from ingest.index_format import (BUILD_PLAN, CHUNKS, EMBEDDINGS, EVAL_REPORT, MANIFEST, TFIDF_MATRIX,
                                 TFIDF_VECTORIZER, IndexData, read_index, write_index)
from ingest.plan import BuildPlan, make_plan
from ingest.store import Store

TFIDF_MODEL = "tfidf"


@dataclass
class BuildResult:
    version: str
    out_dir: Path
    plan: BuildPlan
    manifest: dict


def git_sha(cwd: Path | str | None = None) -> str:
    """The commit being indexed: GITHUB_SHA in CI, `git rev-parse HEAD` locally, else 'unknown'."""
    env = os.environ.get("GITHUB_SHA", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def make_version(now: dt.datetime, source_sha: str) -> str:
    short = source_sha[:7].lower() if re.fullmatch(r"[0-9a-fA-F]{7,}", source_sha) else "0000000"
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{short}"


def load_previous(store: Store | None) -> IndexData | None:
    return read_index(store) if store is not None else None


def _clear_index_files(out_dir: Path) -> None:
    for name in (MANIFEST, CHUNKS, EMBEDDINGS, TFIDF_VECTORIZER, TFIDF_MATRIX, EVAL_REPORT, BUILD_PLAN):
        (out_dir / name).unlink(missing_ok=True)


def build_index(
    docs_dir: Path | str,
    out_dir: Path | str,
    provider: str,
    previous: IndexData | None = None,
    embedder=None,
    now: dt.datetime | None = None,
    source_sha: str | None = None,
    chunking: Chunking = Chunking(),
    doc_prefix: str = "docs",
) -> BuildResult:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown EMBEDDINGS={provider!r}; expected one of {', '.join(PROVIDERS)}")
    out_dir = Path(out_dir)
    docs = load_docs(docs_dir, prefix=doc_prefix)               # raises DocError on bad front matter
    if not docs:
        raise DocError(f"no markdown documents in {docs_dir}")

    fresh = {d.path: chunk_document(d.path, d.body, chunking) for d in docs}
    empty = [p for p, c in fresh.items() if not c]
    if empty:
        raise DocError("no text left after cleaning: " + ", ".join(empty))

    if provider == "tfidf":
        embedder, model, dims = None, TFIDF_MODEL, None
    else:
        embedder = embedder or get_embedder(provider)
        model, dims = embedder.model, embedder.dims

    plan, keep = make_plan(docs, fresh, previous, provider, model, dims, chunking)

    rows: list[Chunk] = []
    reused_vectors: dict[int, np.ndarray] = {}        # row position -> vector (embedding providers)
    new_positions: list[int] = []
    prev_by_doc: dict[str, list[int]] = {}
    if previous is not None:
        for i, row in enumerate(previous.chunks):
            prev_by_doc.setdefault(row["doc_path"], []).append(i)
    for doc in docs:
        if doc.path in keep:
            for i in prev_by_doc[doc.path]:
                old = previous.chunks[i]
                reused_vectors[len(rows)] = previous.embeddings[i]
                rows.append(Chunk(old["id"], old["doc_path"], old["section"], old["text"], old["sha256"]))
        else:
            for chunk in fresh[doc.path]:
                new_positions.append(len(rows))
                rows.append(chunk)

    embeddings = None
    tfidf = None
    if provider == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform([c.text for c in rows])
        tfidf, dims = (vectorizer, matrix), int(matrix.shape[1])
    else:
        embeddings = np.zeros((len(rows), dims), dtype=np.float32)
        for position, vector in reused_vectors.items():
            embeddings[position] = vector
        if new_positions:
            vectors = embedder.embed([rows[p].text for p in new_positions])
            if vectors.shape != (len(new_positions), dims):
                raise ValueError(f"embedder returned {vectors.shape}, expected {(len(new_positions), dims)}")
            embeddings[new_positions] = vectors

    now = now or dt.datetime.now(dt.timezone.utc)
    sha = source_sha or git_sha(Path(docs_dir).resolve().parent)
    version = make_version(now, sha)
    per_doc = {d.path: {"sha256": d.sha256, "chunks": len(fresh[d.path]) if d.path not in keep else len(prev_by_doc[d.path])}
               for d in docs}
    manifest = {
        "version": version,
        "created": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_sha": sha,
        "embeddings": provider,
        "embedding_model": model,
        "dims": dims,
        "docs": per_doc,
        "chunks": len(rows),
        "chunking": chunking.to_dict(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_index_files(out_dir)
    write_index(out_dir, manifest, rows, embeddings=embeddings, tfidf=tfidf)
    (out_dir / BUILD_PLAN).write_text(json.dumps({"version": version, **plan.to_dict()}, indent=2) + "\n", encoding="utf-8")
    return BuildResult(version=version, out_dir=out_dir, plan=plan, manifest=manifest)


def plan_only(
    docs_dir: Path | str,
    provider: str,
    previous: IndexData | None,
    chunking: Chunking = Chunking(),
    doc_prefix: str = "docs",
) -> BuildPlan:
    """What a build WOULD reuse, embed and drop. Reads nothing but the documents; embeds nothing."""
    from ingest.embeddings import HASH_DIMS, TITAN_DIMS, TITAN_MODEL_ID

    docs = load_docs(docs_dir, prefix=doc_prefix)
    fresh = {d.path: chunk_document(d.path, d.body, chunking) for d in docs}
    model, dims = {"titan": (TITAN_MODEL_ID, TITAN_DIMS), "hash": ("hash-512", HASH_DIMS), "tfidf": (TFIDF_MODEL, None)}[provider]
    plan, _ = make_plan(docs, fresh, previous, provider, model, dims, chunking)
    return plan
