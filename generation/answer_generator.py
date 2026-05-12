from __future__ import annotations

import json
import logging
import re
from typing import Any

from evidence.evidence_bundle import AnalystAnswer, EvidenceBundle, SourceCitation
from evidence.trace_formatter import format_trace_steps
from generation.prompts import SYSTEM_PROMPT, build_answer_prompt
from llm_client import LLMClient, LLMClientError

try:
    from json_repair import repair_json as _repair_json  # type: ignore[import]
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, bundle: EvidenceBundle) -> AnalystAnswer:
        if not bundle.final_evidence:
            return self._empty_answer(bundle)
        prompt = build_answer_prompt(bundle)
        try:
            raw = self.llm.chat(SYSTEM_PROMPT, prompt, json_mode=True)
            data = self._parse_json(raw)
            logger.info("Answer JSON parsed successfully")
        except (LLMClientError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Answer generation fallback used: %s", exc)
            data = self._fallback_json(bundle, str(exc))
        return self._to_answer(bundle, data)

    def _parse_json(self, raw: str) -> dict[str, Any]:
        # 1. Standard parse
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # 2. json-repair (handles truncated / malformed JSON from small LLMs)
        if _HAS_JSON_REPAIR:
            try:
                repaired = _repair_json(raw, return_objects=True)
                if isinstance(repaired, dict):
                    logger.debug("JSON repaired successfully")
                    return repaired
            except Exception:
                pass

        # 3. Extract the outermost {...} block and retry
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                if _HAS_JSON_REPAIR:
                    try:
                        repaired = _repair_json(candidate, return_objects=True)
                        if isinstance(repaired, dict):
                            return repaired
                    except Exception:
                        pass

        raise ValueError(f"Could not parse LLM output as JSON. Raw start: {raw[:200]!r}")

    def _to_answer(self, bundle: EvidenceBundle, data: dict[str, Any]) -> AnalystAnswer:
        confidence = str(data.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        sources = []
        for raw_source in data.get("sources") or []:
            if not isinstance(raw_source, dict):
                continue
            sources.append(
                SourceCitation(
                    page=raw_source.get("page"),
                    block_id=str(raw_source.get("block_id") or ""),
                    type=str(raw_source.get("type") or "unknown"),
                    section=raw_source.get("section"),
                    why_relevant=str(raw_source.get("why_relevant") or "Used as evidence."),
                    snippet=str(raw_source.get("snippet") or ""),
                )
            )
        if not sources:
            sources = [
                SourceCitation(
                    page=item.page,
                    block_id=item.block_id,
                    type=item.type,
                    section=item.section,
                    why_relevant=item.why_relevant or "Selected by the retrieval pipeline.",
                    snippet=item.snippet,
                )
                for item in bundle.final_evidence[:5]
            ]

        trace = data.get("trace") if isinstance(data.get("trace"), list) else format_trace_steps(bundle.trace)
        return AnalystAnswer(
            answer=str(data.get("answer") or "I could not produce an answer from the available evidence."),
            confidence=confidence,  # type: ignore[arg-type]
            sources=sources,
            trace=[str(item) for item in trace],
            limitations=str(data.get("limitations") or ""),
            raw_evidence_bundle=bundle,
            raw_answer_json=data,
        )

    def _empty_answer(self, bundle: EvidenceBundle) -> AnalystAnswer:
        data = {
            "answer": "I could not find evidence in the graph for that question.",
            "confidence": "low",
            "sources": [],
            "trace": format_trace_steps(bundle.trace),
            "limitations": "No evidence blocks were retrieved.",
        }
        return self._to_answer(bundle, data)

    def _fallback_json(self, bundle: EvidenceBundle, reason: str) -> dict[str, Any]:
        joined = " ".join(item.snippet for item in bundle.final_evidence[:3])
        return {
            "answer": joined or "I found evidence but could not generate a structured answer.",
            "confidence": "low",
            "sources": [
                {
                    "page": item.page,
                    "block_id": item.block_id,
                    "type": item.type,
                    "section": item.section,
                    "why_relevant": item.why_relevant or "Selected by the retrieval pipeline.",
                    "snippet": item.snippet,
                }
                for item in bundle.final_evidence[:5]
            ],
            "trace": format_trace_steps(bundle.trace),
            "limitations": f"Structured answer generation failed: {reason}",
        }
