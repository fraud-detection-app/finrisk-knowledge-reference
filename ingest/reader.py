"""A tiny reader: question in, best chunks out.

This is what the eval uses, and what the tests use to prove an index can be queried. The risk
service has its own reader; this one only has to agree with it on the format and on how a question
is embedded (same provider, same model, same request body)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ingest.embeddings import HashEmbedder, TitanEmbedder
from ingest.index_format import IndexData, check_index, read_index
from ingest.store import LocalStore, Store


@dataclass(frozen=True)
class Hit:
    id: str
    doc_path: str
    section: str
    text: str
    score: float


class IndexReader:
    def __init__(self, data: IndexData, embedder=None) -> None:
        problems = check_index(data)
        if problems:
            raise ValueError("index format problems: " + "; ".join(problems))
        self.data = data
        provider = data.manifest["embeddings"]
        if provider == "titan":
            self.embedder = embedder or TitanEmbedder()
        elif provider == "hash":
            self.embedder = embedder or HashEmbedder(dims=data.manifest["dims"])
        else:
            self.embedder = None                         # tfidf: the stored vectorizer embeds the question

    @classmethod
    def open(cls, where: Store | Path | str, embedder=None) -> "IndexReader":
        store = where if hasattr(where, "read") else LocalStore(where)
        return cls(read_index(store), embedder=embedder)

    @property
    def version(self) -> str:
        return self.data.manifest["version"]

    def scores(self, question: str) -> np.ndarray:
        if self.data.manifest["embeddings"] == "tfidf":
            return cosine_similarity(self.data.tfidf_vectorizer.transform([question]), self.data.tfidf_matrix).ravel()
        query = np.asarray(self.embedder.embed([question])[0], dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm:
            query = query / norm
        return self.data.embeddings @ query

    def search(self, question: str, k: int = 5) -> list[Hit]:
        scores = self.scores(question)
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(c["id"], c["doc_path"], c["section"], c["text"], float(scores[i]))
                for i in order for c in [self.data.chunks[int(i)]] if scores[i] > 0]
