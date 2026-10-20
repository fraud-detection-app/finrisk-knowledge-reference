import json

import pytest

from ingest.build import build_index
from ingest.publish import PublishError, publish, publish_s3
from ingest.store import (LocalStore, S3Store, StoreError, check_version, index_uri, latest_s3_version, resolve_store,
                          s3_versions)

KB = "finrisk-kb-test"


def built(docs_dir, tmp_path, provider="tfidf", name="b", passed=True, embedder=None, sha="f" * 40):
    result = build_index(docs_dir, tmp_path / name, provider, embedder=embedder, source_sha=sha)
    if passed is not None:
        (tmp_path / name / "eval_report.json").write_text(json.dumps({"passed": passed, "recall_at_5": 1.0}))
    return result


def test_publish_uploads_under_the_version_prefix_manifest_last(docs_dir, tmp_path, s3):
    result = built(docs_dir, tmp_path)
    uri = publish_s3(tmp_path / "b", KB, s3)
    assert uri == f"s3://{KB}/indexes/{result.version}/"
    assert s3.put_order[-1] == f"indexes/{result.version}/manifest.json"
    names = sorted(k.rsplit("/", 1)[1] for _, k in s3.objects)
    assert names == ["build_plan.json", "chunks.jsonl", "eval_report.json", "manifest.json", "tfidf_matrix.joblib", "tfidf_vectorizer.joblib"]


def test_publish_never_overwrites_an_existing_version(docs_dir, tmp_path, s3):
    built(docs_dir, tmp_path)
    publish_s3(tmp_path / "b", KB, s3)
    before = dict(s3.objects)
    with pytest.raises(PublishError, match="never overwritten"):
        publish_s3(tmp_path / "b", KB, s3)
    assert s3.objects == before


def test_s3_itself_refuses_a_racing_overwrite(docs_dir, tmp_path, s3):
    """Even if the prefix looked empty, IfNoneMatch='*' makes S3 reject an existing key."""
    result = built(docs_dir, tmp_path)
    s3.put_object(KB, f"indexes/{result.version}/chunks.jsonl", b"someone else got there first")

    class Blind(type(s3)):                       # a listing that (wrongly) says "empty"
        def list_objects_v2(self, **kw):
            return {}

    blind = Blind()
    blind.objects = s3.objects
    with pytest.raises(PublishError, match="never overwritten"):
        publish_s3(tmp_path / "b", KB, blind)
    assert s3.objects[(KB, f"indexes/{result.version}/chunks.jsonl")] == b"someone else got there first"


def test_publish_refuses_without_a_passing_gate(docs_dir, tmp_path, s3):
    built(docs_dir, tmp_path, name="none", passed=None)
    with pytest.raises(PublishError, match="gate"):
        publish_s3(tmp_path / "none", KB, s3)
    built(docs_dir, tmp_path, name="failed", passed=False)
    with pytest.raises(PublishError, match="FAILED"):
        publish_s3(tmp_path / "failed", KB, s3)
    assert s3.objects == {}


def test_offline_hash_indexes_are_never_published(docs_dir, tmp_path, s3):
    built(docs_dir, tmp_path, provider="hash", name="h")
    with pytest.raises(PublishError, match="offline stand-in"):
        publish_s3(tmp_path / "h", KB, s3)


def test_publish_dispatch_checks_the_destination_matches_the_version(docs_dir, tmp_path, s3):
    built(docs_dir, tmp_path)
    with pytest.raises(PublishError, match="does not match"):
        publish(tmp_path / "b", f"s3://{KB}/indexes/20200101-000000-aaaaaaa/", s3_client=s3)


def test_local_publish_refuses_a_non_empty_destination(docs_dir, tmp_path):
    built(docs_dir, tmp_path)
    dest = tmp_path / "published" / "v"
    publish(tmp_path / "b", str(dest))
    assert (dest / "manifest.json").is_file()
    with pytest.raises(PublishError, match="never overwritten"):
        publish(tmp_path / "b", str(dest))


def test_latest_is_the_max_version_prefix_and_ignores_incomplete_uploads(s3):
    for v in ("20261001-100000-aaaaaaa", "20261016-090000-bbbbbbb"):
        s3.put_object(KB, f"indexes/{v}/manifest.json", b"{}")
    s3.put_object(KB, "indexes/20261020-000000-ccccccc/chunks.jsonl", b"")        # newest, but no manifest yet
    s3.put_object(KB, "indexes/not-a-version/manifest.json", b"{}")
    assert s3_versions(s3, KB) == ["20261001-100000-aaaaaaa", "20261016-090000-bbbbbbb", "20261020-000000-ccccccc"]
    assert latest_s3_version(s3, KB) == "20261016-090000-bbbbbbb"
    store = resolve_store("latest", bucket=KB, s3_client=s3)
    assert isinstance(store, S3Store) and str(store).endswith("indexes/20261016-090000-bbbbbbb/")
    assert resolve_store(f"s3://{KB}/indexes/20261001-100000-aaaaaaa/", s3_client=s3).prefix == "indexes/20261001-100000-aaaaaaa/"
    assert resolve_store(f"s3://{KB}", s3_client=s3).prefix == "indexes/20261016-090000-bbbbbbb/"


def test_resolve_store_with_nothing_published_and_local_folders(tmp_path, s3, docs_dir):
    assert resolve_store("latest", bucket=KB, s3_client=s3) is None
    assert resolve_store(None) is None and resolve_store("none") is None
    assert resolve_store(str(tmp_path / "missing")) is None
    result = built(docs_dir, tmp_path, name="idx")
    assert isinstance(resolve_store(str(tmp_path / "idx")), LocalStore)
    # a folder of versions -> the newest
    (tmp_path / "all").mkdir()
    publish(tmp_path / "idx", str(tmp_path / "all" / result.version))
    assert str(resolve_store(str(tmp_path / "all"))).endswith(result.version)
    with pytest.raises(StoreError):
        resolve_store("latest", bucket=None, s3_client=s3)


def test_version_check_and_uri():
    assert check_version("20261016-090000-abc1234") == "20261016-090000-abc1234"
    for bad in ("latest", "2026-10-16", "20261016-090000-ABC1234", "../x"):
        with pytest.raises(StoreError):
            check_version(bad)
    assert index_uri("b", "v") == "s3://b/indexes/v/"
