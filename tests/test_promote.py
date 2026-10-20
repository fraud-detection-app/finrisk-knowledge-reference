import datetime as dt
import json

import pytest

from ingest.build import build_index
from ingest.promote import ENVS, PromoteError, index_load_error, pointer_value, promote, wait_for_index
from ingest.publish import publish_s3
from tests.conftest import FakeSSM

KB = "finrisk-kb-test"
NOW = dt.datetime(2026, 10, 16, 5, 6, 7, tzinfo=dt.timezone.utc)
RUN = "https://github.com/fraud-detection-app/finrisk-knowledge/actions/runs/123"


@pytest.fixture
def published(docs_dir, tmp_path, s3):
    result = build_index(docs_dir, tmp_path / "b", "tfidf", source_sha="1" * 40)
    (tmp_path / "b" / "eval_report.json").write_text(json.dumps({"passed": True}))
    publish_s3(tmp_path / "b", KB, s3)
    return result


class Clock:
    """A fake clock: sleep() advances time, so a 10-minute timeout takes no real time."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def script(*answers):
    """An http_get that returns the given (status, body) answers in order, then repeats the last."""
    calls = []

    def http_get(url):
        calls.append(url)
        return answers[min(len(calls) - 1, len(answers) - 1)]

    http_get.calls = calls
    return http_get


def ready(version, **index):
    return 200, {"status": "ready", "index": {"version": version, **index}}


def ssm_for(env="dev"):
    return FakeSSM(**{f"/finrisk/{env}/app_url": "https://app.example.test/",
                      f"/finrisk/{env}/index": json.dumps({"version": None})})


def run_promote(published, s3, ssm, answers, env="dev", clock=None, **kw):
    clock = clock or Clock()
    return promote(env, published.version, KB, ssm=ssm, s3=s3, run=RUN, now=NOW, log=lambda *a: None,
                   http_get=script(*answers), clock=clock.time, sleep=clock.sleep, **kw), clock


def test_the_pointer_json_is_exactly_what_the_service_reads(published, s3):
    ssm = ssm_for()
    value, _ = run_promote(published, s3, ssm, [ready(published.version)])
    put = ssm.puts[0]
    assert put["Name"] == "/finrisk/dev/index" and put["Overwrite"] is True
    written = json.loads(put["Value"])
    assert list(written) == ["version", "uri", "embeddings", "docs", "chunks", "promoted_at", "run"]
    assert written == {
        "version": published.version,
        "uri": f"s3://{KB}/indexes/{published.version}/",
        "embeddings": "tfidf",
        "docs": 3,
        "chunks": published.manifest["chunks"],
        "promoted_at": "2026-10-16T05:06:07Z",
        "run": RUN,
    }
    assert written == value


def test_promote_polls_ready_until_the_version_matches(published, s3):
    ssm = ssm_for()
    http = [(0, None), (503, {"status": "starting"}), ready("20200101-000000-aaaaaaa"), ready(published.version)]
    clock = Clock()
    promote("dev", published.version, KB, ssm=ssm, s3=s3, now=NOW, log=lambda *a: None, http_get=(h := script(*http)),
            clock=clock.time, sleep=clock.sleep)
    assert h.calls == ["https://app.example.test/ready"] * 4          # trailing slash handled, /ready polled
    assert clock.sleeps == [10, 10, 10]


def test_promote_times_out_after_ten_minutes(published, s3):
    with pytest.raises(PromoteError, match="timeout after 600s"):
        run_promote(published, s3, ssm_for(), [ready("20200101-000000-aaaaaaa")])


def test_promote_fails_with_the_apps_load_error(published, s3):
    answers = [ready("old", load_error="embeddings.npy has 3 rows, chunks.jsonl has 4")]
    with pytest.raises(PromoteError, match="could not load index.*embeddings.npy has 3 rows"):
        run_promote(published, s3, ssm_for(), answers)


def test_a_stale_load_error_inside_the_pointer_ttl_is_not_blamed_on_this_version(published, s3):
    answers = [ready("bad", load_error="old failure"), ready(published.version)]
    value, clock = run_promote(published, s3, ssm_for(), answers)
    assert value["version"] == published.version and clock.now < 75


def test_load_error_shapes():
    assert index_load_error({"index": {"load_error": "x"}}) == "x"
    assert index_load_error({"index_load_error": "y"}) == "y"
    assert index_load_error({"index": {"version": "v"}}) is None and index_load_error(None) is None


def test_no_wait_writes_the_pointer_only(published, s3):
    ssm = FakeSSM()                                         # not even an app_url
    value = promote("dev", published.version, KB, ssm=ssm, s3=s3, wait=False, now=NOW, log=lambda *a: None)
    assert ssm.puts[0]["Name"] == "/finrisk/dev/index" and value["version"] == published.version


def test_missing_app_url_fails_before_anything_is_written(published, s3):
    ssm = FakeSSM()
    with pytest.raises(PromoteError, match="app_url"):
        promote("sit", published.version, KB, ssm=ssm, s3=s3, log=lambda *a: None)
    assert ssm.puts == []


def test_unknown_env_version_or_missing_index(published, s3):
    ssm = ssm_for()
    with pytest.raises(PromoteError, match="unknown environment"):
        promote("staging", published.version, KB, ssm=ssm, s3=s3)
    with pytest.raises(PromoteError, match="not an index version"):
        promote("dev", "latest", KB, ssm=ssm, s3=s3)
    with pytest.raises(PromoteError, match="no index 20200101-000000-aaaaaaa"):
        promote("dev", "20200101-000000-aaaaaaa", KB, ssm=ssm, s3=s3)
    assert ssm.puts == [] and ENVS == ("dev", "sit", "pre-prod", "prod")


def test_an_index_that_failed_its_gate_is_not_promoted(published, s3):
    s3.put_object(KB, f"indexes/{published.version}/eval_report.json", json.dumps({"passed": False}).encode())
    ssm = ssm_for()
    with pytest.raises(PromoteError, match="failed its evaluation gate"):
        promote("dev", published.version, KB, ssm=ssm, s3=s3, log=lambda *a: None)
    assert ssm.puts == []


def test_rollback_to_an_older_version_overwrites_the_pointer(docs_dir, tmp_path, s3, published):
    (docs_dir / "runbook.md").write_text((docs_dir / "runbook.md").read_text() + "\nExtra.\n")
    newer = build_index(docs_dir, tmp_path / "n", "tfidf", source_sha="2" * 40,
                        now=dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc))
    (tmp_path / "n" / "eval_report.json").write_text(json.dumps({"passed": True}))
    publish_s3(tmp_path / "n", KB, s3)
    ssm = ssm_for()
    for version in (newer.version, published.version):
        promote("dev", version, KB, ssm=ssm, s3=s3, log=lambda *a: None, wait=False)
    assert [json.loads(p["Value"])["version"] for p in ssm.puts] == [newer.version, published.version]


def test_pointer_value_helper_key_order():
    manifest = {"version": "v", "embeddings": "titan", "docs": {"a": {}, "b": {}}, "chunks": 9}
    assert list(pointer_value(manifest, "s3://b/indexes/v/", "t", "r")) == ["version", "uri", "embeddings", "docs", "chunks", "promoted_at", "run"]


def test_summary_file_gets_a_table_and_never_a_bucket_name(published, s3, tmp_path):
    summary = tmp_path / "summary.md"
    run_promote(published, s3, ssm_for(), [ready(published.version)], summary_path=str(summary))
    text = summary.read_text()
    assert "### promote dev" in text and published.version in text and KB not in text
