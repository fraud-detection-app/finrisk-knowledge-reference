"""The incremental build: what is kept, what is embedded, what is dropped."""

import numpy as np

from ingest.build import build_index, load_previous
from ingest.chunking import Chunking, chunk_document
from ingest.plan import TFIDF_REASON
from ingest.docs import load_docs
from ingest.store import LocalStore
from tests.conftest import CountingEmbedder, write_doc


def build(docs, out, provider="titan", previous=None, embedder=None, **kw):
    prev = load_previous(LocalStore(previous)) if previous else None
    return build_index(docs, out, provider, previous=prev, embedder=embedder, source_sha="a" * 40, **kw)


def n_chunks(docs_dir, name):
    doc = next(d for d in load_docs(docs_dir) if d.path == f"docs/{name}")
    return len(chunk_document(doc.path, doc.body))


def test_first_build_embeds_everything(docs_dir, tmp_path, embedder):
    result = build(docs_dir, tmp_path / "v1", embedder=embedder)
    plan = result.plan
    assert plan.line() == f"docs: +3 ~0 -0 · chunks: reused 0, embedded {result.manifest['chunks']}, dropped 0"
    assert len(embedder.texts) == result.manifest["chunks"]


def test_no_op_reuses_every_chunk_and_embeds_nothing(docs_dir, tmp_path, embedder):
    v1 = build(docs_dir, tmp_path / "v1", embedder=embedder)
    embedder.batches.clear()
    v2 = build(docs_dir, tmp_path / "v2", embedder=embedder, previous=tmp_path / "v1")
    assert v2.plan.line() == f"docs: +0 ~0 -0 · chunks: reused {v1.manifest['chunks']}, embedded 0, dropped 0"
    assert embedder.texts == []                                    # not one Bedrock call
    assert not v2.plan.full_rebuild
    assert (tmp_path / "v2" / "chunks.jsonl").read_text() == (tmp_path / "v1" / "chunks.jsonl").read_text()
    assert np.array_equal(np.load(tmp_path / "v2" / "embeddings.npy"), np.load(tmp_path / "v1" / "embeddings.npy"))


def test_adding_a_document_embeds_only_its_chunks(docs_dir, tmp_path, embedder):
    v1 = build(docs_dir, tmp_path / "v1", embedder=embedder)
    embedder.batches.clear()
    write_doc(docs_dir, "new_policy.md", "# New Policy\n\nWire transfers over the limit need a second analyst.\n", "New Policy")
    v2 = build(docs_dir, tmp_path / "v2", embedder=embedder, previous=tmp_path / "v1")
    added = n_chunks(docs_dir, "new_policy.md")
    assert v2.plan.added == ["docs/new_policy.md"]
    assert v2.plan.line() == f"docs: +1 ~0 -0 · chunks: reused {v1.manifest['chunks']}, embedded {added}, dropped 0"
    assert len(embedder.texts) == added and all("Wire transfers" in t for t in embedder.texts)


def test_changing_a_document_drops_its_old_chunks_and_embeds_the_new_ones(docs_dir, tmp_path, embedder):
    v1 = build(docs_dir, tmp_path / "v1", embedder=embedder)
    old_changed = v1.manifest["docs"]["docs/fraud_policy.md"]["chunks"]
    embedder.batches.clear()
    path = docs_dir / "fraud_policy.md"
    path.write_text(path.read_text() + "\n## Escalation\n\nEscalate to the fraud lead after two hours.\n", encoding="utf-8")
    v2 = build(docs_dir, tmp_path / "v2", embedder=embedder, previous=tmp_path / "v1")
    new_count = n_chunks(docs_dir, "fraud_policy.md")
    assert v2.plan.changed == ["docs/fraud_policy.md"]
    assert v2.plan.reused == v1.manifest["chunks"] - old_changed
    assert v2.plan.embedded == new_count and v2.plan.dropped == old_changed
    assert len(embedder.texts) == new_count                        # only the changed document was embedded
    assert v2.manifest["chunks"] == v1.manifest["chunks"] - old_changed + new_count


def test_unchanged_chunks_keep_their_exact_embedding_rows(docs_dir, tmp_path, embedder):
    build(docs_dir, tmp_path / "v1", embedder=embedder)
    path = docs_dir / "runbook.md"
    path.write_text(path.read_text() + "\nMore rollback words.\n", encoding="utf-8")
    build(docs_dir, tmp_path / "v2", embedder=embedder, previous=tmp_path / "v1")

    def by_id(version):
        rows = [__import__("json").loads(l) for l in (tmp_path / version / "chunks.jsonl").read_text().splitlines()]
        vectors = np.load(tmp_path / version / "embeddings.npy")
        return {r["id"]: (r["doc_path"], vectors[i]) for i, r in enumerate(rows)}

    old, new = by_id("v1"), by_id("v2")
    kept = [i for i, (doc, _) in old.items() if doc != "docs/runbook.md"]
    assert kept and all(i in new and np.array_equal(old[i][1], new[i][1]) for i in kept)


def test_deleting_a_document_drops_its_chunks(docs_dir, tmp_path, embedder):
    v1 = build(docs_dir, tmp_path / "v1", embedder=embedder)
    gone = v1.manifest["docs"]["docs/runbook.md"]["chunks"]
    embedder.batches.clear()
    (docs_dir / "runbook.md").unlink()
    v2 = build(docs_dir, tmp_path / "v2", embedder=embedder, previous=tmp_path / "v1")
    assert v2.plan.line() == f"docs: +0 ~0 -1 · chunks: reused {v1.manifest['chunks'] - gone}, embedded 0, dropped {gone}"
    assert embedder.texts == [] and "docs/runbook.md" not in v2.manifest["docs"]
    assert "runbook" not in (tmp_path / "v2" / "chunks.jsonl").read_text()


def test_mixed_change_prints_the_plan_line(docs_dir, tmp_path, embedder):
    build(docs_dir, tmp_path / "v1", embedder=embedder)
    (docs_dir / "runbook.md").unlink()
    (docs_dir / "model_card.md").write_text((docs_dir / "model_card.md").read_text() + "\nA new sentence.\n")
    write_doc(docs_dir, "extra.md", "# Extra\n\nSomething extra.\n")
    plan = build(docs_dir, tmp_path / "v2", embedder=embedder, previous=tmp_path / "v1").plan
    assert plan.line().startswith("docs: +1 ~1 -1 · chunks: reused ")
    assert (plan.added, plan.changed, plan.removed) == (["docs/extra.md"], ["docs/model_card.md"], ["docs/runbook.md"])


def test_tfidf_is_always_a_full_rebuild_and_says_why(docs_dir, tmp_path):
    v1 = build(docs_dir, tmp_path / "v1", "tfidf")
    v2 = build(docs_dir, tmp_path / "v2", "tfidf", previous=tmp_path / "v1")       # nothing changed at all
    assert v2.plan.full_rebuild and v2.plan.reason == TFIDF_REASON
    assert "IDF" in v2.plan.reason and "whole corpus" in v2.plan.reason
    assert v2.plan.reused == 0 and v2.plan.embedded == v1.manifest["chunks"]
    assert v2.plan.dropped == v1.manifest["chunks"]
    assert (tmp_path / "v2" / "tfidf_vectorizer.joblib").is_file() and (tmp_path / "v2" / "tfidf_matrix.joblib").is_file()
    assert not (tmp_path / "v2" / "embeddings.npy").exists()


def test_switching_provider_rebuilds_everything(docs_dir, tmp_path, embedder):
    build(docs_dir, tmp_path / "v1", "tfidf")
    plan = build(docs_dir, tmp_path / "v2", embedder=embedder, previous=tmp_path / "v1").plan
    assert plan.full_rebuild and plan.reused == 0 and "cannot be mixed" in plan.reason


def test_changed_chunk_settings_rebuild_everything(docs_dir, tmp_path, embedder):
    build(docs_dir, tmp_path / "v1", embedder=embedder)
    plan = build(docs_dir, tmp_path / "v2", embedder=embedder, previous=tmp_path / "v1", chunking=Chunking(60, 10)).plan
    assert plan.full_rebuild and "chunk settings changed" in plan.reason


def test_a_damaged_previous_document_is_rechunked_not_trusted(docs_dir, tmp_path, embedder):
    import json
    build(docs_dir, tmp_path / "v1", embedder=embedder)
    manifest = json.loads((tmp_path / "v1" / "manifest.json").read_text())
    manifest["docs"]["docs/runbook.md"]["chunks"] += 1                  # the manifest lies about one document
    (tmp_path / "v1" / "manifest.json").write_text(json.dumps(manifest))
    plan = build(docs_dir, tmp_path / "v2", embedder=embedder, previous=tmp_path / "v1").plan
    assert plan.changed == ["docs/runbook.md"]
