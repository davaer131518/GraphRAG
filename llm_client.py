from __future__ import annotations

import logging
import re

import requests

from config import Settings

logger = logging.getLogger(__name__)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.llm_server_url.rstrip("/")
        self.timeout = settings.request_timeout_seconds
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMClientError(f"LLM server unavailable at {self.base_url}: {exc}") from exc
        if not response.ok:
            raise LLMClientError(f"LLM server returned {response.status_code}: {response.text[:400]}")
        try:
            raw = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("LLM response did not contain choices[0].message.content") from exc
        cleaned = _THINK_RE.sub("", raw).strip()
        logger.debug("LLM returned %s characters", len(cleaned))
        return cleaned
