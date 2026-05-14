from __future__ import annotations

import json

from evidence.evidence_bundle import EvidenceBundle, make_snippet

SYSTEM_PROMPT = """You are a traceable PDF analyst.
Answer only from the provided evidence. Do not use outside knowledge.
If the evidence is incomplete, say so in limitations and lower confidence.

IMPORTANT: Read EVERY evidence block before writing the answer. When an evidence entry has "all_block_ids", that entry represents multiple PDF bullet items merged into one snippet — include ALL items from that snippet in your answer and cite each block_id in "all_block_ids" as a separate source.

When an evidence entry's "why_relevant" starts with "Contains question terms:", those exact keywords from the question appear in that block's snippet — read it carefully before concluding the terms are absent.

Every source must cite page, block_id, type, section_title, why_relevant, and snippet.
Return only valid JSON with this schema:
{
  "answer": "human-readable answer",
  "confidence": "high | medium | low",
  "sources": [
    {
      "page": 1,
      "block_id": "p0001_b0000",
      "type": "paragraph",
      "section_title": "Section heading",
      "why_relevant": "why this source supports the answer",
      "snippet": "short quote or table excerpt"
    }
  ],
  "limitations": "state missing or ambiguous evidence, or empty string"
}
"""


_LIST_BLOCK_MAX_CHARS = 350  # blocks shorter than this are candidates for list merging
_STOPWORDS = frozenset(
    "a,an,the,and,or,of,in,on,at,to,for,is,are,was,were,be,been,being,"
    "have,has,had,do,does,did,will,would,could,should,may,might,shall,"
    "this,that,these,those,it,its,with,by,from,as,about,into,through,"
    "which,what,who,how,when,where,why,not,no,nor,so,yet,both,either,neither".split(",")
)


def _question_tokens(question: str) -> list[str]:
    """Return lower-cased tokens > 3 chars that are not stopwords."""
    return [
        w.lower().strip("?.,;:!\"'()")
        for w in question.split()
        if len(w) > 3 and w.lower().strip("?.,;:!\"'()") not in _STOPWORDS
    ]


def build_answer_prompt(bundle: EvidenceBundle, *, prompt_snippet_max_chars: int = 1000) -> str:
    # Use a larger snippet than the UI source card (item.snippet is capped at 360 chars for display).
    # Key phrases often appear mid-block past the 360-char cap, so the LLM needs more context
    # than the display snippet provides.
    items = bundle.final_evidence
    q_tokens = _question_tokens(bundle.question)


    # Detect list groups: 2+ short blocks from the same section anywhere in the evidence set.
    # PDF bullet lists are split one item per Block; gather all of them so the LLM sees the
    # complete list in a single entry, regardless of how the blocks ranked individually.
    section_indices: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        if item.section_id and len(item.text or "") < _LIST_BLOCK_MAX_CHARS:
            section_indices.setdefault(item.section_id, []).append(i)
    list_groups: dict[str, list[int]] = {
        sid: idxs for sid, idxs in section_indices.items() if len(idxs) >= 2
    }

    evidence: list[dict] = []
    emitted_groups: set[str] = set()

    for i, item in enumerate(items):
        sid = item.section_id

        if sid and sid in list_groups and i in list_groups[sid]:
            if sid in emitted_groups:
                continue  # merged into the group entry already emitted
            emitted_groups.add(sid)
            cluster = [items[j] for j in list_groups[sid]]
            merged_snippet = "\n".join(
                "• " + make_snippet(ci.text, max_chars=prompt_snippet_max_chars).lstrip("•").lstrip()
                for ci in cluster
            )
            entry: dict = {
                "block_id": item.block_id,
                "all_block_ids": [ci.block_id for ci in cluster],
                "type": item.type,
                "page": item.page,
                "section_title": item.section_title,
                "why_relevant": item.why_relevant,
                "snippet": merged_snippet,
            }
            if item.section_path and item.section_path != item.section_title:
                entry["section_path"] = item.section_path
            hits = [t for t in q_tokens if t in merged_snippet.lower()]
            if hits:
                terms_str = ", ".join(sorted(set(hits)))
                entry["why_relevant"] = (
                    f"Contains question terms: {terms_str}. "
                    + (entry.get("why_relevant") or "Selected by retrieval pipeline.")
                )
        else:
            snippet = make_snippet(item.text, max_chars=prompt_snippet_max_chars)
            entry = {
                "block_id": item.block_id,
                "type": item.type,
                "page": item.page,
                "section_title": item.section_title,
                "why_relevant": item.why_relevant,
                "snippet": snippet,
            }
            if item.section_path and item.section_path != item.section_title:
                entry["section_path"] = item.section_path
            # Check full text (not truncated snippet) so terms beyond max_chars are detected
            text_lower = (item.text or "").lower()
            hits = [t for t in q_tokens if t in text_lower]
            if hits:
                terms_str = ", ".join(sorted(set(hits)))
                entry["why_relevant"] = (
                    f"Contains question terms: {terms_str}. "
                    + (item.why_relevant or "Selected by retrieval pipeline.")
                )
            if item.mentioned_entities:
                # Cap at 3 entities in the prompt to keep it bounded
                top = item.mentioned_entities[:3]
                entry["entities"] = "; ".join(
                    f"{e.get('name', e.get('entity_name', '?'))} ({e.get('type', e.get('entity_type', '?'))})"
                    for e in top
                )

        evidence.append(entry)

    payload = {
        "question": bundle.question,
        "evidence": evidence,
    }
    return (
        "Use this evidence to answer the question. "
        "Return only JSON; no markdown.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
