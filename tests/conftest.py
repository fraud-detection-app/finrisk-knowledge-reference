"""Shared fakes. Nothing here touches the network: S3, SSM and Bedrock are small in-memory stand-ins."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from ingest.embeddings import TITAN_DIMS, TITAN_MODEL_ID, HashEmbedder

ROOT = Path(__file__).resolve().parents[1]

# Folders that tools generate (or that hold git's own bookkeeping). A scan of "what is committed" must never read them:
# on a CI runner .git/ holds fetched refs and shas; .venv/ holds other people's code; build/ holds the indexes the
# workflow just made; _main/ is the checkout of main the pull-request check compares against.
GENERATED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "_main"}


def repo_files(root: Path = ROOT) -> list[Path]:
    """The files of the repository that git tracks (or would track): `git ls-files` when `root` is a git checkout and git
    is installed, otherwise a walk of the folder that does not enter GENERATED_DIRS or another checkout nested inside
    it. Either way nothing from a generated folder, and never a .pyc."""
    names: list[Path] | None = None
    if (root / ".git").exists() and shutil.which("git"):
        done = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                              capture_output=True, text=True)
        if done.returncode == 0:
            names = [root / n for n in done.stdout.split("\0") if n]
    if names is None:
        names = []
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in GENERATED_DIRS and not (Path(folder) / d / ".git").exists()]   # prune
            names.extend(Path(folder) / f for f in files)
    return sorted(p for p in names if p.is_file() and p.suffix != ".pyc"
                  and not GENERATED_DIRS.intersection(p.relative_to(root).parts))


FRONT = "---\ntitle: {title}\nowner: Compliance\neffective: 2026-10-16\n---\n\n"

DOCS = {
    "fraud_policy.md": ("Fraud Review Policy",
                        "# Fraud Review Policy\n\n## Analyst review\n\nHigh risk transactions must be reviewed by an analyst. "
                        "The model is a prioritisation aid and never an automated decision system.\n\n"
                        "## Advice\n\nThe assistant must not give investment advice.\n"),
    "model_card.md": ("Model Card",
                      "# Model Card\n\n## Excluded fields\n\nThe field is_flagged_fraud is excluded from the features because it "
                      "leaks an older rule. The target is is_fraud.\n\n## Metric\n\nPR-AUC ranks the rare fraud class.\n"),
    "runbook.md": ("Runbook",
                   "# Runbook\n\n## Rollback\n\nTo roll back, run the rollback workflow with the previous commit SHA and approve it.\n"),
}


def write_doc(directory: Path, name: str, body: str, title: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(FRONT.format(title=title or name) + body, encoding="utf-8")
    return path


@pytest.fixture
def docs_dir(tmp_path) -> Path:
    directory = tmp_path / "docs"
    for name, (title, body) in DOCS.items():
        write_doc(directory, name, body, title)
    return directory


class CountingEmbedder(HashEmbedder):
    """Titan stand-in for tests: same 512-dim shape, deterministic, and it counts what it was asked to embed."""

    provider = "titan"
    model = TITAN_MODEL_ID

    def __init__(self) -> None:
        super().__init__(TITAN_DIMS)
        self.batches: list[list[str]] = []

    def embed(self, texts):
        self.batches.append(list(texts))
        return super().embed(texts)

    @property
    def texts(self) -> list[str]:
        return [t for batch in self.batches for t in batch]


@pytest.fixture
def embedder() -> CountingEmbedder:
    return CountingEmbedder()


def client_error(code: str, operation: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


class FakeS3:
    """The few S3 calls the code makes, over a dict. put_object honours IfNoneMatch='*' like the real thing."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_order: list[str] = []

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, **_):
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise client_error("PreconditionFailed", "PutObject")
        self.objects[(Bucket, Key)] = bytes(Body)
        self.put_order.append(Key)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise client_error("NoSuchKey", "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix="", Delimiter=None, MaxKeys=1000, ContinuationToken=None):
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        if Delimiter:
            prefixes = sorted({Prefix + k[len(Prefix):].split(Delimiter)[0] + Delimiter for k in keys if Delimiter in k[len(Prefix):]})
            direct = [k for k in keys if Delimiter not in k[len(Prefix):]]
            return {"CommonPrefixes": [{"Prefix": p} for p in prefixes], "Contents": [{"Key": k} for k in direct][:MaxKeys]}
        return {"Contents": [{"Key": k} for k in keys[:MaxKeys]]}


class FakeSSM:
    def __init__(self, **values: str) -> None:
        self.values = dict(values)
        self.puts: list[dict] = []

    def get_parameter(self, Name):
        if Name not in self.values:
            raise client_error("ParameterNotFound", "GetParameter")
        return {"Parameter": {"Name": Name, "Value": self.values[Name]}}

    def put_parameter(self, Name, Value, Overwrite=False, **kw):
        self.puts.append({"Name": Name, "Value": Value, "Overwrite": Overwrite, **kw})
        self.values[Name] = Value


class FakeBedrock:
    """invoke_model like the real client: body in, {'body': stream} out. ``fail`` is a list of error codes to raise first."""

    def __init__(self, fail: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self.fail = list(fail or [])

    def invoke_model(self, modelId, body, contentType, accept):
        self.calls.append({"modelId": modelId, "body": body, "contentType": contentType, "accept": accept})
        if self.fail:
            raise client_error(self.fail.pop(0), "InvokeModel")
        text = json.loads(body)["inputText"]
        vector = HashEmbedder(TITAN_DIMS).embed([text])[0].tolist()
        return {"body": io.BytesIO(json.dumps({"embedding": vector, "inputTextTokenCount": 3}).encode())}


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3()
