"""Load the knowledge-base markdown files and check their front matter.

A file looks like this::

    ---
    title: Fraud Review Policy
    owner: Compliance
    effective: 2026-07-02
    ---

    # Fraud Review Policy
    ...

The front matter is METADATA (who owns the text, since when it applies). Only the body is
cleaned, chunked and embedded. The document hash covers the whole file, so changing the owner or
the effective date also counts as a change and goes through review."""

from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Files that must never enter the index. An evaluation report contains the test questions
# themselves, so indexing one lets each question retrieve its own answer key.
DEFAULT_EXCLUDE: tuple[str, ...] = (
    "README*", "rag_eval*", "rag_retrieval*", "agent_routing*", "*_eval_results*", "eval_report*",
)
REQUIRED_KEYS = ("title", "owner", "effective")
_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)


class DocError(Exception):
    """The documents cannot be indexed (bad front matter, no files)."""


@dataclass(frozen=True)
class Document:
    path: str            # "docs/fraud_policy.md" -- the id used in chunk ids and the manifest
    sha256: str          # of the whole file
    meta: dict
    body: str
    problems: tuple[str, ...] = field(default=())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_front_matter(raw: str) -> tuple[dict | None, str]:
    """(metadata, body). metadata is None when the file has no front matter block."""
    match = _FRONT_MATTER.match(raw)
    if not match:
        return None, raw
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {"__error__": "front matter is not valid YAML"}, raw[match.end():]
    return (meta if isinstance(meta, dict) else {"__error__": "front matter is not a mapping"}), raw[match.end():]


def _problems(meta: dict | None, body: str) -> list[str]:
    if meta is None:
        return ["no front matter (a block between two '---' lines at the top)"]
    if "__error__" in meta:
        return [meta["__error__"]]
    found = []
    for key in REQUIRED_KEYS:
        value = meta.get(key)
        if value is None or not str(value).strip():
            found.append(f"front matter is missing '{key}'")
    effective = meta.get("effective")
    if effective is not None and str(effective).strip() and not isinstance(effective, dt.date):
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(effective).strip()):
                raise ValueError
            dt.date.fromisoformat(str(effective).strip())
        except ValueError:
            found.append(f"'effective' must be YYYY-MM-DD, got {effective!r}")
    if not body.strip():
        found.append("the document body is empty")
    return found


def load_docs(
    docs_dir: Path | str,
    prefix: str = "docs",
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    strict: bool = True,
) -> list[Document]:
    """Every ``*.md`` under ``docs_dir``, sorted by path.

    ``prefix`` is the folder name used in document paths, so a checkout of main in ``_main/docs`` still
    produces ``docs/faq.md`` and the two builds can be compared. With ``strict`` a document with bad
    front matter raises ``DocError`` listing all of them; the lint command uses ``strict=False``."""
    root = Path(docs_dir)
    if not root.is_dir():
        raise DocError(f"{root} is not a folder")
    docs: list[Document] = []
    for file in sorted(p for p in root.rglob("*.md") if p.is_file()):
        if any(fnmatch.fnmatch(file.name, pattern) for pattern in exclude):
            continue
        raw_bytes = file.read_bytes()
        raw = raw_bytes.decode("utf-8")
        meta, body = split_front_matter(raw)
        rel = file.relative_to(root).as_posix()
        docs.append(Document(
            path=f"{prefix}/{rel}",
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            meta={k: (v.isoformat() if isinstance(v, (dt.date, dt.datetime)) else v)
                  for k, v in (meta or {}).items() if k != "__error__"},
            body=body,
            problems=tuple(_problems(meta, body)),
        ))
    if strict:
        bad = [f"{d.path}: {p}" for d in docs for p in d.problems]
        if bad:
            raise DocError("front matter problems:\n  " + "\n  ".join(bad))
    return docs


def lint_docs(docs_dir: Path | str, prefix: str = "docs") -> list[str]:
    """Human-readable problems, empty when every document is fine."""
    docs = load_docs(docs_dir, prefix=prefix, strict=False)
    problems = [f"{d.path}: {p}" for d in docs for p in d.problems]
    if not docs:
        problems.append(f"no markdown documents found in {docs_dir}")
    titles: dict[str, str] = {}
    for d in docs:
        title = str(d.meta.get("title", "")).strip().lower()
        if title and title in titles:
            problems.append(f"{d.path}: same title as {titles[title]}")
        titles.setdefault(title, d.path)
    return problems
