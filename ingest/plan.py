"""The incremental plan: what can this build keep from the previous index?

    docs: +1 ~2 -0 · chunks: reused 40, embedded 7, dropped 5

* ``+`` added documents, ``~`` changed documents (different sha256), ``-`` deleted documents
* ``reused``   chunks kept from the previous index (row AND embedding), because their document is unchanged
* ``embedded`` chunks embedded (or vectorised) in this build
* ``dropped``  chunks of the previous index that are not in the new one

Planning is pure: it reads no network and embeds nothing, so ``python -m ingest diff`` is free."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ingest.chunking import Chunk, Chunking
from ingest.docs import Document
from ingest.index_format import IndexData

TFIDF_REASON = (
    "tfidf is ALWAYS a full rebuild: the IDF weight of every word depends on the whole corpus, "
    "so changing one document changes the vector of every chunk"
)


@dataclass
class BuildPlan:
    provider: str
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    reused: int = 0
    embedded: int = 0
    dropped: int = 0
    full_rebuild: bool = True
    reason: str | None = None
    previous_version: str | None = None

    def line(self) -> str:
        return (f"docs: +{len(self.added)} ~{len(self.changed)} -{len(self.removed)} · "
                f"chunks: reused {self.reused}, embedded {self.embedded}, dropped {self.dropped}")

    def to_dict(self) -> dict:
        return {**asdict(self), "line": self.line()}


def _incompatible(previous: IndexData, provider: str, model: str, dims: int | None, chunking: Chunking) -> str | None:
    """Why the previous index's vectors/chunks cannot be reused, or None when they can."""
    manifest = previous.manifest
    if manifest.get("embeddings") != provider or manifest.get("embedding_model") != model:
        return (f"the previous index was built with {manifest.get('embeddings')} ({manifest.get('embedding_model')}); "
                f"its vectors cannot be mixed with {provider} ({model})")
    if dims is not None and manifest.get("dims") != dims:
        return f"the previous index has {manifest.get('dims')} dimensions, this build {dims}"
    if manifest.get("chunking") != chunking.to_dict():
        return f"the chunk settings changed ({manifest.get('chunking')} -> {chunking.to_dict()})"
    if previous.embeddings is None or previous.embeddings.shape[0] != len(previous.chunks):
        return "the previous embeddings do not line up with its chunks"
    return None


def make_plan(
    docs: list[Document],
    fresh: dict[str, list[Chunk]],
    previous: IndexData | None,
    provider: str,
    model: str,
    dims: int | None,
    chunking: Chunking,
) -> tuple[BuildPlan, set[str]]:
    """Return the plan and the set of documents whose old rows are kept as they are."""
    plan = BuildPlan(provider=provider)
    current = {d.path: d.sha256 for d in docs}
    if previous is None:
        plan.added = sorted(current)
        plan.embedded = sum(len(fresh[p]) for p in current)
        plan.reason = "first build: there is no previous index"
        return plan, set()

    plan.previous_version = previous.manifest.get("version")
    old_docs = previous.manifest.get("docs", {})
    rows_per_doc: dict[str, int] = {}
    for row in previous.chunks:
        rows_per_doc[row["doc_path"]] = rows_per_doc.get(row["doc_path"], 0) + 1

    plan.added = sorted(set(current) - set(old_docs))
    plan.removed = sorted(set(old_docs) - set(current))
    for path in sorted(set(current) & set(old_docs)):
        same = old_docs[path].get("sha256") == current[path]
        intact = rows_per_doc.get(path, 0) == old_docs[path].get("chunks")      # a damaged index is re-chunked
        (plan.unchanged if same and intact else plan.changed).append(path)

    if provider == "tfidf":
        reason = TFIDF_REASON
    else:
        reason = _incompatible(previous, provider, model, dims, chunking)
    plan.reason = reason
    plan.full_rebuild = reason is not None

    if plan.full_rebuild:
        keep: set[str] = set()
        plan.embedded = sum(len(fresh[p]) for p in current)
    else:
        keep = set(plan.unchanged)
        plan.embedded = sum(len(fresh[p]) for p in plan.added + plan.changed)
    plan.reused = sum(rows_per_doc[p] for p in keep)
    plan.dropped = len(previous.chunks) - plan.reused
    return plan, keep
