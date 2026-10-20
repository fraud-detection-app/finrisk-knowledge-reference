"""Clean markdown and cut it into overlapping, section-tagged chunks.

Rules reused from the risk service's RAG code (week 9):

* fenced code, table rules, link URLs and heading marks are noise for a similarity score -> cleaned once, at index time;
* ``_`` STAYS: in this corpus underscores are identifiers (``is_flagged_fraud``), and stripping them
  would make every feature name unsearchable;
* chunks inherit the markdown heading they sat under, so a citation reads "faq.md § FAQ";
* ``overlap`` words repeat at the head of the next chunk so an answer that straddles a boundary survives.

Chunk id = sha256(doc_path + "\\n" + text)[:16]. The id depends only on the document path and the
chunk text, so an unchanged document always produces the same ids."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

MAX_WORDS = 140
OVERLAP = 30


@dataclass(frozen=True)
class Chunking:
    max_words: int = MAX_WORDS
    overlap: int = OVERLAP

    def __post_init__(self) -> None:
        if self.overlap >= self.max_words:
            raise ValueError(f"overlap ({self.overlap}) must be smaller than max_words ({self.max_words})")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_path: str
    section: str
    text: str
    sha256: str

    def row(self) -> dict:
        """One line of chunks.jsonl. Key order is part of the format."""
        return {"id": self.id, "doc_path": self.doc_path, "section": self.section,
                "text": self.text, "sha256": self.sha256}


def chunk_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id(doc_path: str, text: str) -> str:
    return hashlib.sha256((doc_path + "\n" + text).encode("utf-8")).hexdigest()[:16]


def clean_markdown(text: str) -> str:
    out: list[str] = []
    in_code = False
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if set(stripped) <= {"|", "-", ":", " "} and stripped:
            continue                                            # table rule rows
        stripped = stripped.replace("|", " ")                   # table cell separators
        stripped = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", stripped)   # [text](url) -> text
        stripped = re.sub(r"https?://\S+", "", stripped)
        stripped = re.sub(r"[`*>]", "", stripped)               # NOT "_": identifiers keep their underscores
        out.append(stripped)
    return "\n".join(out)


def split_sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) pairs so chunks inherit a section name."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if match:
            if buffer:
                sections.append((heading, "\n".join(buffer)))
                buffer = []
            heading = match.group(2).strip()
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer)))
    return [(h, b) for h, b in sections if b.strip()]


def chunk_document(doc_path: str, body: str, chunking: Chunking = Chunking()) -> list[Chunk]:
    """Chunks of one document, in document order. Identical texts inside one document are kept once."""
    chunks: list[Chunk] = []
    seen: set[str] = set()
    step = chunking.max_words - chunking.overlap
    for heading, section_body in split_sections(clean_markdown(body)):
        words = [w for w in re.split(r"\s+", section_body.strip()) if w]
        for index in range(0, len(words), step):
            segment = " ".join(words[index:index + chunking.max_words]).strip()
            if segment:
                text = f"{heading}. {segment}" if heading else segment
                cid = chunk_id(doc_path, text)
                if cid not in seen:
                    seen.add(cid)
                    chunks.append(Chunk(cid, doc_path, heading, text, chunk_sha256(text)))
            if index + chunking.max_words >= len(words):
                break
    return chunks
