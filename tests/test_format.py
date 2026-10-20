"""The index format is a contract with the risk service (spec 3.4): these tests pin it."""

import json

import numpy as np
import pytest

from ingest.build import build_index, make_version
from ingest.embeddings import TITAN_DIMS, TITAN_MODEL_ID, TitanEmbedder, titan_request_body
from ingest.index_format import MANIFEST_KEYS, check_index, read_index
from ingest.store import VERSION_RE, LocalStore
from tests.conftest import FakeBedrock, client_error


def titan_index(docs_dir, tmp_path, embedder):
    build_index(docs_dir, tmp_path / "idx", "titan", embedder=embedder, source_sha="0123456789abcdef" * 2 + "01234567")
    return tmp_path / "idx"


def test_manifest_keys_and_order(docs_dir, tmp_path, embedder):
    folder = titan_index(docs_dir, tmp_path, embedder)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert list(manifest)[:8] == list(MANIFEST_KEYS) == ["version", "created", "source_sha", "embeddings", "embedding_model", "dims", "docs", "chunks"]
    assert VERSION_RE.match(manifest["version"]) and manifest["version"].endswith("-0123456")
    assert manifest["embeddings"] == "titan" and manifest["embedding_model"] == TITAN_MODEL_ID and manifest["dims"] == 512
    assert manifest["chunks"] == sum(d["chunks"] for d in manifest["docs"].values())
    assert set(manifest["docs"]["docs/fraud_policy.md"]) == {"sha256", "chunks"}
    assert manifest["created"].endswith("Z")


def test_chunks_jsonl_rows_and_embedding_alignment(docs_dir, tmp_path, embedder):
    folder = titan_index(docs_dir, tmp_path, embedder)
    lines = (folder / "chunks.jsonl").read_text().splitlines()
    rows = [json.loads(l) for l in lines]
    assert all(list(r) == ["id", "doc_path", "section", "text", "sha256"] for r in rows)
    assert all(len(r["id"]) == 16 and len(r["sha256"]) == 64 for r in rows)
    vectors = np.load(folder / "embeddings.npy")
    assert vectors.dtype == np.float32 and vectors.shape == (len(rows), 512)
    # row i is line i: re-embed line 7's text and compare
    i = len(rows) // 2
    assert np.allclose(embedder.embed([rows[i]["text"]])[0], vectors[i])
    data = read_index(LocalStore(folder))
    assert check_index(data) == []


def test_tfidf_format(docs_dir, tmp_path):
    result = build_index(docs_dir, tmp_path / "t", "tfidf", source_sha="b" * 40)
    folder = tmp_path / "t"
    assert sorted(p.name for p in folder.iterdir()) == ["build_plan.json", "chunks.jsonl", "manifest.json", "tfidf_matrix.joblib", "tfidf_vectorizer.joblib"]
    data = read_index(LocalStore(folder))
    assert data.manifest["embeddings"] == "tfidf" and data.manifest["embedding_model"] == "tfidf"
    assert data.tfidf_matrix.shape == (result.manifest["chunks"], data.manifest["dims"])
    assert check_index(data) == []


def test_check_index_catches_misaligned_rows(docs_dir, tmp_path, embedder):
    folder = titan_index(docs_dir, tmp_path, embedder)
    data = read_index(LocalStore(folder))
    data.embeddings = data.embeddings[:-1]
    assert any("shape" in p for p in check_index(data))
    data = read_index(LocalStore(folder))
    data.embeddings = data.embeddings.astype(np.float64)
    assert any("float32" in p for p in check_index(data))


def test_version_is_timestamp_dash_sha7():
    import datetime as dt
    now = dt.datetime(2026, 10, 16, 3, 4, 5, tzinfo=dt.timezone.utc)
    assert make_version(now, "ABCDEF0123456789" * 2 + "01234567") == "20261016-030405-abcdef0"
    assert make_version(now, "unknown") == "20261016-030405-0000000"


def test_titan_request_body_is_exactly_the_agreed_one():
    assert json.loads(titan_request_body("hello")) == {"inputText": "hello", "dimensions": 512, "normalize": True}


def test_titan_embedder_calls_bedrock_with_that_body():
    fake = FakeBedrock()
    vectors = TitanEmbedder(client=fake, sleep=lambda s: None).embed(["alpha", "beta"])
    assert vectors.shape == (2, 512) and vectors.dtype == np.float32
    assert {c["modelId"] for c in fake.calls} == {"amazon.titan-embed-text-v2:0"}
    assert sorted(json.loads(c["body"])["inputText"] for c in fake.calls) == ["alpha", "beta"]
    assert all(json.loads(c["body"])["dimensions"] == 512 and json.loads(c["body"])["normalize"] is True for c in fake.calls)


def test_titan_retries_throttling_with_backoff_then_succeeds():
    fake = FakeBedrock(fail=["ThrottlingException", "ModelTimeoutException"])
    delays = []
    vectors = TitanEmbedder(client=fake, sleep=delays.append, workers=1).embed(["x"])
    assert vectors.shape == (1, 512) and len(fake.calls) == 3
    assert len(delays) == 2 and delays[1] > delays[0] * 0.9            # backoff grows


def test_titan_gives_up_after_max_retries_and_does_not_retry_access_denied():
    fake = FakeBedrock(fail=["ThrottlingException"] * 10)
    with pytest.raises(Exception, match="ThrottlingException"):
        TitanEmbedder(client=fake, sleep=lambda s: None, max_retries=2, workers=1).embed(["x"])
    assert len(fake.calls) == 3
    denied = FakeBedrock(fail=["AccessDeniedException"])
    with pytest.raises(Exception, match="AccessDenied"):
        TitanEmbedder(client=denied, sleep=lambda s: None, workers=1).embed(["x"])
    assert len(denied.calls) == 1
