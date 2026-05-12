from __future__ import annotations

import logging
import math
from typing import Any

import requests

from config import Settings

logger = logging.getLogger(__name__)


class EmbeddingClientError(RuntimeError):
    pass


class EmbeddingsClient:
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.embed_server_url.rstrip("/")
        self.timeout = settings.request_timeout_seconds
        self.embed_max_chars = settings.embed_max_chars

    def embed_query(self, question: str) -> list[float]:
        return self.embed_text(f"[QUERY] {question}")

    def embed_text(self, text: str) -> list[float]:
        max_chars = self.embed_max_chars
        last_error: Exception | None = None
        for attempt in range(5):
            payload = {"input": text[:max_chars], "encoding_format": "float"}
            try:
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise EmbeddingClientError(f"Embedding server unavailable at {self.base_url}: {exc}") from exc

            if response.ok:
                try:
                    vector = response.json()["data"][0]["embedding"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise EmbeddingClientError("Embedding response did not contain data[0].embedding") from exc
                normalized = self._l2_normalize(vector)
                logger.debug("Embedded query text with %s dimensions", len(normalized))
                return normalized

            last_error = EmbeddingClientError(
                f"Embedding server returned {response.status_code} on attempt {attempt + 1}: "
                f"{response.text[:400]}"
            )
            if response.status_code != 500 or max_chars <= 100:
                break
            max_chars //= 2
            logger.warning("Embedding failed with 500; retrying with %s chars", max_chars)

        raise last_error or EmbeddingClientError("Embedding failed for an unknown reason")

    @staticmethod
    def _l2_normalize(vector: list[Any]) -> list[float]:
        floats = [float(v) for v in vector]
        norm = math.sqrt(sum(v * v for v in floats))
        if norm == 0:
            return floats
        return [v / norm for v in floats]
