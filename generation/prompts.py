from __future__ import annotations

import json

from evidence.evidence_bundle import EvidenceBundle, make_snippet

SYSTEM_PROMPT = """You are a traceable PDF analyst.
Answer only from the provided evidence. Do not use outside knowledge.
If the evidence is incomplete, say so in limitations and lower confidence.

IMPORTANT: Read EVERY evidence block before writing the answer. When an evidence entry has "all_block_ids", that entry represents multiple PDF bullet items merged into one snippet — include ALL items from that snippet in your answer and cite each block_id in "all_block_ids" as a separate source.

IMPORTANT: Evidence blocks with "type": "table" contain real financial or structured data formatted with | (pipe) delimiters. The values between | separators are actual numbers — parse and use them. Never say numerical data is missing if a table block's snippet contains numbers separated by |.

IMPORTANT: When a table snippet shows multiple $ values in a row without explicit column labels, apply the column order established by the nearest paragraph or context evidence entry for that table. For example, if a paragraph states the table covers "2025, 2024 and 2023" and the table row shows "| $ | 416,161 | $ | 391,035 | $ | 383,285", then 2025=$416,161, 2024=$391,035, 2023=$383,285. Do not say a year's value is missing when the value is present and the column order is known from context.

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

COMPARATIVE_SYSTEM_PROMPT = """You are a traceable PDF analyst comparing multiple documents.
Answer only from the provided evidence. Do not use outside knowledge.
The evidence spans MULTIPLE documents; each evidence entry includes a "document" field with the source document label.

IMPORTANT: Evidence blocks with "type": "table" contain real financial or structured data formatted with | (pipe) delimiters. The values between | separators are actual numbers — parse and use them. Never say numerical data is missing if a table block's snippet contains numbers separated by |.

IMPORTANT: When a table snippet shows multiple $ values in a row without explicit column labels, apply the column order established by the nearest paragraph or context evidence entry for that table. For example, if a paragraph states the table covers "2025, 2024 and 2023" and the table row shows "| $ | 416,161 | $ | 391,035 | $ | 383,285", then 2025=$416,161, 2024=$391,035, 2023=$383,285. Do not say a year's value is missing when the value is present and the column order is known from context.

For every claim, state which document it comes from. When documents agree, say so explicitly.
When documents differ, compare and contrast them clearly and attribute each side to its document.
If a document is silent on a point, note that rather than assuming it agrees.

Every source must cite page, block_id, type, section_title, document, why_relevant, and snippet.
Return only valid JSON with this schema:
{
  "answer": "human-readable answer with per-document attribution",
  "confidence": "high | medium | low",
  "sources": [
    {
      "page": 1,
      "block_id": "p0001_b0000",
      "type": "paragraph",
      "section_title": "Section heading",
      "document": "2024_10-K.pdf",
      "why_relevant": "why this source supports the answer",
      "snippet": "short quote or table excerpt"
    }
  ],
  "limitations": "state missing or ambiguous evidence, or empty string"
}
"""


def select_system_prompt(bundle: EvidenceBundle) -> str:
    """Return the comparative prompt when final evidence spans >1 distinct document."""
    distinct_docs = {item.doc_id for item in bundle.final_evidence if item.doc_id}
    return COMPARATIVE_SYSTEM_PROMPT if len(distinct_docs) > 1 else SYSTEM_PROMPT


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
    items = bundle.final_evidence
    q_tokens = _question_tokens(bundle.question)

    # Multi-doc flag: add "document" key to every evidence entry when spans >1 doc
    multi_doc = len({it.doc_id for it in items if it.doc_id}) > 1

    # Detect list groups: 2+ short blocks from the same section anywhere in the evidence set.
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
            if multi_doc and item.doc_label:
                entry["document"] = item.doc_label
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
            if multi_doc and item.doc_label:
                entry["document"] = item.doc_label
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
