from pathlib import Path

from ingest.docs import DocError, lint_docs, load_docs
from tests.conftest import write_doc

import pytest


def test_load_docs_uses_the_docs_prefix_whatever_the_folder_is_called(tmp_path, docs_dir):
    other = tmp_path / "_main" / "docs"
    write_doc(other, "a.md", "# A\n\ntext here")
    assert [d.path for d in load_docs(other)] == ["docs/a.md"]
    assert [d.path for d in load_docs(docs_dir)] == ["docs/fraud_policy.md", "docs/model_card.md", "docs/runbook.md"]


def test_front_matter_is_parsed_and_hash_covers_the_whole_file(docs_dir):
    before = {d.path: d for d in load_docs(docs_dir)}
    assert before["docs/model_card.md"].meta == {"title": "Model Card", "owner": "Compliance", "effective": "2026-10-16"}
    path = docs_dir / "model_card.md"
    path.write_text(path.read_text().replace("owner: Compliance", "owner: Risk"), encoding="utf-8")
    after = {d.path: d for d in load_docs(docs_dir)}
    assert after["docs/model_card.md"].sha256 != before["docs/model_card.md"].sha256      # an owner change is a change
    assert after["docs/fraud_policy.md"].sha256 == before["docs/fraud_policy.md"].sha256


def test_lint_accepts_the_shipped_documents():
    assert lint_docs(Path(__file__).resolve().parents[1] / "docs") == []


def test_lint_reports_every_problem(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "none.md").write_text("# no front matter\ntext", encoding="utf-8")
    (docs / "partial.md").write_text("---\ntitle: P\n---\nbody", encoding="utf-8")
    (docs / "baddate.md").write_text("---\ntitle: D\nowner: O\neffective: 16/10/2026\n---\nbody", encoding="utf-8")
    problems = "\n".join(lint_docs(docs))
    assert "docs/none.md: no front matter" in problems
    assert "missing 'owner'" in problems and "missing 'effective'" in problems
    assert "YYYY-MM-DD" in problems


def test_strict_load_refuses_bad_front_matter(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("no front matter", encoding="utf-8")
    with pytest.raises(DocError):
        load_docs(docs)


def test_reports_and_readme_are_never_indexed(docs_dir):
    (docs_dir / "README.md").write_text("readme", encoding="utf-8")
    (docs_dir / "rag_eval_report.md").write_text("the answer key", encoding="utf-8")
    assert not {"docs/README.md", "docs/rag_eval_report.md"} & {d.path for d in load_docs(docs_dir)}
