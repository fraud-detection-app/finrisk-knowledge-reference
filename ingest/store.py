"""Where an index lives: a local folder or ``s3://<kb-bucket>/indexes/<version>/``.

Both look the same to the rest of the code: ``read(name)`` and ``exists(name)``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

VERSION_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{7}$")
INDEXES = "indexes/"
MANIFEST = "manifest.json"


class StoreError(Exception):
    pass


class Store(Protocol):
    def read(self, name: str) -> bytes: ...
    def exists(self, name: str) -> bool: ...


class LocalStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def read(self, name: str) -> bytes:
        return (self.path / name).read_bytes()

    def exists(self, name: str) -> bool:
        return (self.path / name).is_file()

    def __str__(self) -> str:
        return str(self.path)


class S3Store:
    def __init__(self, client, bucket: str, prefix: str) -> None:
        self.client, self.bucket = client, bucket
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"

    def read(self, name: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self.prefix + name)["Body"].read()

    def exists(self, name: str) -> bool:
        listing = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix + name, MaxKeys=1)
        return any(item["Key"] == self.prefix + name for item in listing.get("Contents", []))

    def __str__(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise StoreError(f"not an s3:// uri: {uri}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket:
        raise StoreError(f"no bucket in {uri}")
    return bucket, key


def index_uri(bucket: str, version: str) -> str:
    return f"s3://{bucket}/{INDEXES}{version}/"


def check_version(version: str) -> str:
    if not VERSION_RE.match(version):
        raise StoreError(f"{version!r} is not an index version (expected YYYYMMDD-HHMMSS-<sha7>)")
    return version


def s3_versions(client, bucket: str, base: str = INDEXES) -> list[str]:
    """Every version prefix under ``base``, oldest first. Names sort as dates, so max = latest."""
    versions, token = [], None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": base, "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for item in page.get("CommonPrefixes", []):
            name = item["Prefix"][len(base):].strip("/")
            if VERSION_RE.match(name):
                versions.append(name)
        token = page.get("NextContinuationToken") if page.get("IsTruncated") else None
        if not token:
            return sorted(versions)


def latest_s3_version(client, bucket: str, base: str = INDEXES) -> str | None:
    """The newest version that is COMPLETE (manifest.json is uploaded last)."""
    for version in reversed(s3_versions(client, bucket, base)):
        if S3Store(client, bucket, base + version).exists(MANIFEST):
            return version
    return None


def resolve_store(spec: str | None, bucket: str | None = None, s3_client=None) -> Store | None:
    """Turn ``--previous`` into a store, or None when there is nothing to build on.

    * ``None`` / ``none``                       -> None
    * ``latest``                                -> newest index in ``bucket``
    * ``s3://b`` or ``s3://b/indexes/``         -> newest index under that prefix
    * ``s3://b/indexes/<version>/``             -> exactly that index
    * a folder holding manifest.json            -> that folder
    * a folder holding version sub-folders      -> the newest one
    """
    if not spec or spec == "none":
        return None
    if spec == "latest" or spec.startswith("s3://"):
        if spec == "latest":
            if not bucket:
                raise StoreError("--previous latest needs --bucket (or the KB_BUCKET variable)")
            bkt, key = bucket, INDEXES
        else:
            bkt, key = parse_s3_uri(spec)
        if s3_client is None:
            import boto3
            s3_client = boto3.client("s3")
        key = key if key.endswith("/") or not key else key + "/"
        last = key.rstrip("/").rsplit("/", 1)[-1]
        if VERSION_RE.match(last):
            return S3Store(s3_client, bkt, key)
        base = key or INDEXES
        version = latest_s3_version(s3_client, bkt, base)
        return S3Store(s3_client, bkt, base + version) if version else None
    path = Path(spec)
    if (path / MANIFEST).is_file():
        return LocalStore(path)
    if path.is_dir():
        subs = sorted(p.name for p in path.iterdir() if p.is_dir() and VERSION_RE.match(p.name) and (p / MANIFEST).is_file())
        if subs:
            return LocalStore(path / subs[-1])
    return None
