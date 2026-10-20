"""Turn text into vectors.

Three providers, one rule: the question must be embedded EXACTLY like the chunks were.

* ``titan``  -- Amazon Titan Text Embeddings V2 on Bedrock. The real one. 512 dimensions, normalised.
                The risk service sends the same request body for every question (``titan_request_body``).
* ``tfidf``  -- scikit-learn TF-IDF fitted on all chunks. No AWS needed, but it has no stand-alone
                embedder: the fitted vectorizer is part of the index (see ``build.py``).
* ``hash``   -- feature hashing into 512 dimensions. A stand-in for Titan that runs offline and
                needs no fitting, so the CI job and the tests can build a REAL incremental index.
                It is never published or promoted.
"""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
TITAN_DIMS = 512
HASH_DIMS = 512
PROVIDERS = ("titan", "tfidf", "hash")
PUBLISHABLE = ("titan", "tfidf")        # what the risk service can read

# Bedrock errors worth another try. Anything else (validation, access denied) fails at once.
RETRYABLE = {
    "ThrottlingException", "TooManyRequestsException", "ModelTimeoutException",
    "ServiceUnavailableException", "InternalServerException", "ModelNotReadyException",
}


def titan_request_body(text: str) -> str:
    """The InvokeModel body. The app MUST build its query vectors with the same body."""
    return json.dumps({"inputText": text, "dimensions": TITAN_DIMS, "normalize": True})


class TitanEmbedder:
    provider = "titan"
    model = TITAN_MODEL_ID
    dims = TITAN_DIMS

    def __init__(self, client=None, region: str | None = None, max_retries: int = 6,
                 base_delay: float = 0.5, workers: int = 4,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if client is None:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "ap-southeast-2"))
        self.client = client
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.workers = workers
        self._sleep = sleep

    def _embed_one(self, text: str) -> list[float]:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.invoke_model(
                    modelId=TITAN_MODEL_ID, body=titan_request_body(text),
                    contentType="application/json", accept="application/json")
                return json.loads(response["body"].read())["embedding"]
            except Exception as exc:                                  # noqa: BLE001
                code = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
                transient = code in RETRYABLE or type(exc).__name__ in {"ConnectionError", "ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError"}
                if not transient or attempt == self.max_retries:
                    raise
                delay = self.base_delay * (2 ** attempt) * (0.5 + random.random() / 2)
                self._sleep(delay)
        raise RuntimeError("unreachable")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, TITAN_DIMS), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            vectors = list(pool.map(self._embed_one, texts))
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape[1] != TITAN_DIMS:
            raise ValueError(f"Titan returned {matrix.shape[1]} dimensions, expected {TITAN_DIMS}")
        return matrix


class HashEmbedder:
    provider = "hash"
    model = "hash-512"

    def __init__(self, dims: int = HASH_DIMS) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer
        self.dims = dims
        self._vectorizer = HashingVectorizer(n_features=dims, ngram_range=(1, 2), stop_words="english",
                                             norm="l2", alternate_sign=False)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        return self._vectorizer.transform(texts).toarray().astype(np.float32)


def get_embedder(provider: str):
    if provider == "titan":
        return TitanEmbedder()
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"{provider!r} has no stand-alone embedder (tfidf is fitted on the corpus)")
