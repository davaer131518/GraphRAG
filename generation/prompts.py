from __future__ import annotations

import json

from evidence.evidence_bundle import EvidenceBundle

# Fields the LLM actually needs to produce a cited answer.
# Full `text`, internal retrieval metadata, and the trace are omitted to
# keep the prompt well within the model's context window.
SYSTEM_PROMPT = """You are a traceable PDF analyst.
Answer only from the provided evidence. Do not use outside knowledge.
If the evidence is incomplete, say so in limitations and lower confidence.
Every source must cite page, block_id, type, section, why_relevant, and snippet.
Return only valid JSON with this schema:
{
  "answer": "human-readable answer",
  "confidence": "high | medium | low",
  "sources": [
    {
      "page": 1,
      "block_id": "p0001_b0000",
      "type": "paragraph",
      "section": "Section heading",
      "why_relevant": "why this source supports the answer",
      "snippet": "short quote or table excerpt"
    }
  ],
  "limitations": "state missing or ambiguous evidence, or empty string"
}
"""


def build_answer_prompt(bundle: EvidenceBundle) -> str:
    evidence = []
    for item in bundle.final_evidence:
        evidence.append(
            {
                "block_id": item.block_id,
                "type": item.type,
                "page": item.page,
                "section": item.section,
                "why_relevant": item.why_relevant,
                "snippet": item.snippet,
            }
        )
    payload = {
        "question": bundle.question,
        "evidence": evidence,
    }
    return (
        "Use this evidence to answer the question. "
        "Return only JSON; no markdown.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
