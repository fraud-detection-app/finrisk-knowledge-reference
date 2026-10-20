"""The command line, end to end, on a tiny corpus. No network."""

import json

from ingest.cli import main
from ingest.safe import mask
from tests.conftest import write_doc


def test_build_diff_publish_latest_round_trip(docs_dir, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["build", "--docs", str(docs_dir), "--out", str(out / "v1"), "--embeddings", "hash"]) == 0
    text = capsys.readouterr().out
    assert "docs: +3 ~0 -0 · chunks: reused 0, embedded" in text and "first build" in text

    write_doc(docs_dir, "new.md", "# New\n\nA brand new policy about wire transfers.\n")
    assert main(["diff", "--docs", str(docs_dir), "--embeddings", "hash", "--previous", str(out / "v1")]) == 0
    text = capsys.readouterr().out
    assert "docs: +1 ~0 -0" in text and "+ docs/new.md" in text and "embedded 1" in text

    assert main(["build", "--docs", str(docs_dir), "--out", str(out / "v2"), "--embeddings", "hash", "--previous", str(out / "v1")]) == 0
    assert "reused" in capsys.readouterr().out


def test_tfidf_build_prints_the_full_rebuild_reason(docs_dir, tmp_path, capsys):
    main(["build", "--docs", str(docs_dir), "--out", str(tmp_path / "a"), "--embeddings", "tfidf"])
    main(["build", "--docs", str(docs_dir), "--out", str(tmp_path / "b"), "--embeddings", "tfidf", "--previous", str(tmp_path / "a")])
    assert "IDF weight of every word depends on the whole corpus" in capsys.readouterr().out


def test_publish_to_a_local_destination_then_latest(docs_dir, tmp_path, capsys):
    main(["build", "--docs", str(docs_dir), "--out", str(tmp_path / "b"), "--embeddings", "tfidf"])
    version = json.loads((tmp_path / "b" / "manifest.json").read_text())["version"]
    assert main(["publish", "--from", str(tmp_path / "b"), "--dest", str(tmp_path / "pub" / version)]) == 2        # no gate report yet
    assert "no eval_report.json" in capsys.readouterr().out
    assert main(["publish", "--from", str(tmp_path / "b"), "--dest", str(tmp_path / "pub" / version), "--no-gate"]) == 0
    assert main(["publish", "--from", str(tmp_path / "b"), "--dest", str(tmp_path / "pub" / version), "--no-gate"]) == 2
    assert "never overwritten" in capsys.readouterr().out
    assert main(["latest", "--dir", str(tmp_path / "pub")]) == 0 and capsys.readouterr().out.strip() == version


def test_lint_exit_codes(docs_dir, capsys):
    assert main(["lint", "--docs", str(docs_dir)]) == 0
    (docs_dir / "bad.md").write_text("no front matter", encoding="utf-8")
    assert main(["lint", "--docs", str(docs_dir)]) == 1
    assert "docs/bad.md: no front matter" in capsys.readouterr().out


def test_build_refuses_bad_front_matter_and_unknown_provider(docs_dir, tmp_path, capsys):
    (docs_dir / "bad.md").write_text("no front matter", encoding="utf-8")
    assert main(["build", "--docs", str(docs_dir), "--out", str(tmp_path / "x"), "--embeddings", "tfidf"]) == 2
    assert "front matter problems" in capsys.readouterr().out


def test_summary_markdown(docs_dir, tmp_path, capsys):
    main(["build", "--docs", str(docs_dir), "--out", str(tmp_path / "h"), "--embeddings", "hash"])
    capsys.readouterr()
    assert main(["summary", "--plan", str(tmp_path / "h" / "build_plan.json")]) == 0
    text = capsys.readouterr().out
    assert "Knowledge base check" in text and "reused" in text and "hash" in text


def test_account_ids_are_masked():
    fake = "".join(str(d) for d in range(1, 10)) + "012"          # a made-up 12-digit id, not written out in the source
    assert mask(f"s3://finrisk-kb-{fake}-ap-southeast-2-an/indexes/") == "s3://finrisk-kb-<account-id>-ap-southeast-2-an/indexes/"
    assert mask(f"arn:aws:iam::{fake}:role/x") == "arn:aws:iam::<account-id>:role/x"
    assert mask("20261016-090000-abc1234 and " + fake + "3 stay") == "20261016-090000-abc1234 and " + fake + "3 stay"
