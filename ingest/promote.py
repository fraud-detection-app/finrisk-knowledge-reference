"""Promote an index version to one environment: write the SSM pointer, then wait for the app.

    /finrisk/<env>/index  =  {"version", "uri", "embeddings", "docs", "chunks", "promoted_at", "run"}

The pointer parameter is created by finrisk-infra with ``{"version": null}``; from then on THIS repo
owns its value. The service re-reads the pointer every POINTER_TTL_SECONDS (60), downloads the new
index in the background and swaps it in. ``/ready`` then reports ``index.version``; the job succeeds
when it equals the version we promoted."""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request
from typing import Callable

from ingest.embeddings import PUBLISHABLE
from ingest.index_format import EVAL_REPORT, MANIFEST
from ingest.safe import say
from ingest.store import INDEXES, S3Store, StoreError, check_version, index_uri, s3_versions

ENVS = ("dev", "sit", "pre-prod", "prod")
READY_TIMEOUT_SECONDS = 600
POLL_SECONDS = 10
# The app only re-reads the pointer every POINTER_TTL_SECONDS (60). A load_error seen before that has
# passed may belong to an EARLIER version, so it is not blamed on this one.
GRACE_SECONDS = 75


class PromoteError(Exception):
    pass


def pointer_name(env: str) -> str:
    return f"/finrisk/{env}/index"


def app_url_name(env: str) -> str:
    return f"/finrisk/{env}/app_url"


def pointer_value(manifest: dict, uri: str, promoted_at: str, run: str) -> dict:
    """The exact JSON the service reads. Key order is part of the contract."""
    return {
        "version": manifest["version"],
        "uri": uri,
        "embeddings": manifest["embeddings"],
        "docs": len(manifest["docs"]),
        "chunks": manifest["chunks"],
        "promoted_at": promoted_at,
        "run": run,
    }


def http_get_json(url: str, timeout: float = 15.0) -> tuple[int, dict | None]:
    """(status, json body). (0, None) when the service cannot be reached yet."""
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "finrisk-knowledge"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:          # noqa: S310 - url comes from SSM
            return response.status, _json(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _json(exc.read())
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None


def _json(raw: bytes) -> dict | None:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except ValueError:
        return None


def index_load_error(ready: dict | None) -> str | None:
    """The app's explanation when it could not load the index, wherever /ready puts it."""
    ready = ready or {}
    index = ready.get("index") or {}
    return index.get("load_error") or index.get("error") or ready.get("index_load_error") or None


def wait_for_index(
    app_url: str,
    version: str,
    timeout: float = READY_TIMEOUT_SECONDS,
    interval: float = POLL_SECONDS,
    grace: float = GRACE_SECONDS,
    http_get: Callable[[str], tuple[int, dict | None]] = http_get_json,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[..., None] = say,
) -> dict:
    url = app_url.rstrip("/") + "/ready"
    started = clock()
    last = "no answer yet"
    while True:
        status, body = http_get(url)
        seen = ((body or {}).get("index") or {}).get("version")
        elapsed = clock() - started
        error = index_load_error(body)
        if seen == version:
            log(f"  /ready reports index.version {version} after {elapsed:.0f}s")
            return body or {}
        if error and elapsed >= grace:
            raise PromoteError(f"the app could not load index {version}: {error}")
        last = f"HTTP {status}, index.version = {seen}" if status else "the service did not answer"
        log(f"  [{elapsed:4.0f}s] waiting for index.version {version}: {last}" + (f" (app reports: {error})" if error else ""))
        if elapsed + interval > timeout:
            raise PromoteError(f"timeout after {timeout:.0f}s: /ready never reported index.version {version} ({last})")
        sleep(interval)


def promote(
    env: str,
    version: str,
    bucket: str,
    ssm,
    s3,
    run: str = "local",
    wait: bool = True,
    now: dt.datetime | None = None,
    summary_path: str | None = None,
    log: Callable[..., None] = say,
    **wait_options,
) -> dict:
    if env not in ENVS:
        raise PromoteError(f"unknown environment {env!r} (one of: {', '.join(ENVS)})")
    try:
        check_version(version)
    except StoreError as exc:
        raise PromoteError(str(exc)) from exc

    store = S3Store(s3, bucket, f"{INDEXES}{version}")
    if not store.exists(MANIFEST):
        known = ", ".join(s3_versions(s3, bucket)[-5:]) or "none"
        raise PromoteError(f"no index {version} in the knowledge bucket (latest versions: {known})")
    manifest = json.loads(store.read(MANIFEST))
    if manifest["embeddings"] not in PUBLISHABLE:
        raise PromoteError(f"embeddings={manifest['embeddings']!r} cannot be served")
    if store.exists(EVAL_REPORT) and not json.loads(store.read(EVAL_REPORT)).get("passed", False):
        raise PromoteError(f"index {version} failed its evaluation gate: not promoting")

    app_url = None
    if wait:
        try:
            app_url = ssm.get_parameter(Name=app_url_name(env))["Parameter"]["Value"].strip()
        except Exception as exc:                                                    # noqa: BLE001
            raise PromoteError(f"cannot read {app_url_name(env)}: deploy the app to {env} first, or use --no-wait") from exc

    before = None
    try:
        before = json.loads(ssm.get_parameter(Name=pointer_name(env))["Parameter"]["Value"]).get("version")
    except Exception:                                                               # noqa: BLE001
        pass

    promoted_at = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    value = pointer_value(manifest, index_uri(bucket, version), promoted_at, run)
    ssm.put_parameter(Name=pointer_name(env), Value=json.dumps(value), Overwrite=True)
    log(f"{env}: {pointer_name(env)} {before or 'unset'} -> {version}  ({value['docs']} docs, {value['chunks']} chunks, {value['embeddings']})")

    if wait:
        log(f"{env}: waiting for the app to report index.version {version} (up to {wait_options.get('timeout', READY_TIMEOUT_SECONDS)}s)")
        wait_for_index(app_url, version, log=log, **wait_options)
        log(f"{env}: OK, the app answers from index {version}")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(f"### promote {env}\n\n| | |\n|---|---|\n| index | `{version}` |\n| replaced | `{before or 'unset'}` |\n"
                         f"| docs / chunks | {value['docs']} / {value['chunks']} |\n| embeddings | {value['embeddings']} |\n"
                         f"| app confirmed | {'yes' if wait else 'not checked (--no-wait)'} |\n\n")
    return value
