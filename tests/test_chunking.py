import hashlib

import pytest

from ingest.chunking import Chunking, chunk_document, chunk_id, chunk_sha256, clean_markdown, split_sections
from ingest.docs import split_front_matter


def test_cleaning_keeps_underscores_in_identifiers():
    cleaned = clean_markdown("The `is_flagged_fraud` field and amount_to_origin_balance are *features*.")
    assert "is_flagged_fraud" in cleaned
    assert "amount_to_origin_balance" in cleaned
    assert "`" not in cleaned and "*" not in cleaned


def test_cleaning_drops_code_tables_links_and_comments():
    text = "Intro [the guide](https://example.com/x) here.\n```\nsecret_code()\n```\n| a | b |\n|---|---|\n| 1 | 2 |\n<!-- note -->\nEnd."
    cleaned = clean_markdown(text)
    assert "secret_code" not in cleaned and "example.com" not in cleaned and "note" not in cleaned
    assert "the guide" in cleaned and "---" not in cleaned and "|" not in cleaned


def test_chunk_id_is_sha256_of_path_newline_text_truncated():
    text = "Policy. Some text."
    expected = hashlib.sha256(b"docs/a.md\nPolicy. Some text.").hexdigest()[:16]
    assert chunk_id("docs/a.md", text) == expected and len(expected) == 16
    assert chunk_sha256(text) == hashlib.sha256(text.encode()).hexdigest()


def test_chunk_ids_are_stable_and_depend_on_path():
    body = "# Title\n\n## One\n\nalpha beta gamma\n\n## Two\n\ndelta epsilon\n"
    first = chunk_document("docs/a.md", body)
    again = chunk_document("docs/a.md", body)
    other = chunk_document("docs/b.md", body)
    assert [c.id for c in first] == [c.id for c in again]
    assert not set(c.id for c in first) & set(c.id for c in other)
    assert [c.section for c in first] == ["One", "Two"]
    assert first[0].text == "One. alpha beta gamma"
    assert first[0].sha256 == chunk_sha256(first[0].text)


def test_chunks_overlap_and_never_exceed_the_window():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_document("docs/a.md", f"# T\n\n{words}\n", Chunking(max_words=40, overlap=10))
    bodies = [c.text.removeprefix("T. ").split() for c in chunks]
    assert all(len(b) <= 40 for b in bodies) and len(chunks) == 3
    assert bodies[0][-10:] == bodies[1][:10]                      # the overlap repeats the tail
    assert bodies[-1][-1] == "w99"                                # nothing is lost at the end


def test_overlap_must_be_smaller_than_the_window():
    with pytest.raises(ValueError):
        Chunking(max_words=10, overlap=10)


def test_identical_text_inside_one_document_is_kept_once():
    body = "# A\n\nsame words here\n\n# A\n\nsame words here\n"
    assert len(chunk_document("docs/a.md", body)) == 1


def test_sections_skip_empty_headings():
    assert split_sections("# Only a title\n## Empty\n") == []


def test_front_matter_is_metadata_not_text():
    meta, body = split_front_matter("---\ntitle: T\nowner: O\neffective: 2026-01-02\n---\n\n# Body\ntext")
    assert meta["title"] == "T" and body.strip().startswith("# Body")
    assert split_front_matter("# no front matter")[0] is None
