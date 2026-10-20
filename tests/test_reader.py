"""A query round trip: build an index, embed a question the same way, find the right document."""

import pytest

from ingest.build import build_index
from ingest.reader import IndexReader


@pytest.mark.parametrize("question, expected", [
    ("Why is is_flagged_fraud excluded from the features?", "docs/model_card.md"),
    ("who must review high risk transactions", "docs/fraud_policy.md"),
    ("how do I roll back", "docs/runbook.md"),
])
def test_titan_shaped_index_finds_the_expected_document(docs_dir, tmp_path, embedder, question, expected):
    build_index(docs_dir, tmp_path / "i", "titan", embedder=embedder, source_sha="c" * 40)
    reader = IndexReader.open(tmp_path / "i", embedder=embedder)           # the SAME embedder embeds the question
    hits = reader.search(question, k=3)
    assert hits and hits[0].doc_path == expected and hits[0].score > 0


@pytest.mark.parametrize("question, expected", [
    ("Why is is_flagged_fraud excluded from the features?", "docs/model_card.md"),
    ("who must review high risk transactions", "docs/fraud_policy.md"),
])
def test_tfidf_index_round_trip(docs_dir, tmp_path, question, expected):
    build_index(docs_dir, tmp_path / "i", "tfidf", source_sha="c" * 40)
    assert IndexReader.open(tmp_path / "i").search(question, k=3)[0].doc_path == expected


def test_hash_index_needs_no_embedder_argument(docs_dir, tmp_path):
    build_index(docs_dir, tmp_path / "i", "hash", source_sha="c" * 40)
    assert IndexReader.open(tmp_path / "i").search("rollback workflow previous commit", k=1)[0].doc_path == "docs/runbook.md"


def test_reader_refuses_a_broken_index(docs_dir, tmp_path, embedder):
    build_index(docs_dir, tmp_path / "i", "titan", embedder=embedder, source_sha="c" * 40)
    (tmp_path / "i" / "chunks.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        IndexReader.open(tmp_path / "i", embedder=embedder)
