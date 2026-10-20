"""Never print a 12-digit AWS account id.

Bucket names and role ARNs contain it. Everything the CLI prints goes through ``mask`` first."""

from __future__ import annotations

import re

_ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")


def mask(text: object) -> str:
    return _ACCOUNT_ID.sub("<account-id>", str(text))


def say(*parts: object) -> None:
    """print() with the account id masked."""
    print(mask(" ".join(str(p) for p in parts)), flush=True)
