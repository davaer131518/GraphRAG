from __future__ import annotations

import json
import re
from html.parser import HTMLParser

from evidence.evidence_bundle import EvidenceBundle, make_snippet

_THOUSANDS_RE = re.compile(r"(\d),(\d{3})(?!\d)")
_FOOTNOTE_RE = re.compile(r"\s*\$\s*\^?\{?\(?\d+\)?\}?\s*\$\s*")


def _strip_number_commas(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _THOUSANDS_RE.sub(r"\1\2", text)
    return text


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._in_cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._row.append(" ".join(self._cell).strip())
            self._in_cell = False
        elif tag == "tr" and self._row:
            self.rows.append(self._row)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell.append(data.strip())


def _html_table_to_llm_text(html: str) -> str:
    """Convert HTML table to a labeled-row text format for LLM consumption.

    Output format:
        Table columns: 2023 | 2022 | 2021
        iPhone: 200583 | 205489 | 191973
        Services: 85200 | 78129 | 68425

    Strips thousands-separator commas from values (200,583 → 200583) and
    footnote markers from row labels (iPhone $ ^{(1)} $ → iPhone).
    Falls back to comma-stripped HTML if parsing yields no rows.
    """
    p = _TableHTMLParser()
    p.feed(html)
    if not p.rows:
        return _strip_number_commas(html)

    max_cols = max(len(r) for r in p.rows)
    rows = [r + [""] * (max_cols - len(r)) for r in p.rows]

    def clean_val(v: str) -> str:
        return _strip_number_commas(v)

    def clean_label(s: str) -> str:
        return _FOOTNOTE_RE.sub("", s).strip()

    headers = [clean_val(c) for c in rows[0][1:]]
    header_str = " | ".join(h for h in headers if h)

    lines = []
    if header_str:
        lines.append(f"Table columns: {header_str}")
    for row in rows[1:]:
        if not any(c.strip() for c in row):
            continue
        label = clean_label(row[0])
        values = " | ".join(clean_val(c) for c in row[1:])
        lines.append(f"{label}: {values}")
    return "\n".join(lines)

SYSTEM_PROMPT = """You are a traceable PDF analyst.
Answer only from the provided evidence. Do not use outside knowledge.
If the evidence is incomplete, say so in limitations and lower confidence.

IMPORTANT: Read EVERY evidence block before writing the answer. When an evidence entry has "all_block_ids", that entry represents multiple PDF bullet items merged into one snippet — include ALL items from that snippet in your answer and cite each block_id in "all_block_ids" as a separate source.

IMPORTANT: Evidence blocks with "type": "table" contain real financial or structured data in one of two formats:
- Labeled-row format (primary): starts with "Table columns: col1 | col2 | ..." then one row per line as "Row label: val1 | val2 | ...". Thousands-separator commas are removed — 200583 means 200,583 (two hundred thousand five hundred eighty-three).
- Pipe-delimited format (fallback): rows of cells separated by | characters, e.g. "| iPhone | 200,583 | 205,489 |". Values may include commas as thousands separators — treat them as such, not as decimal points.
Parse and use these values precisely. Never say numerical data is missing if a table block contains the relevant numbers.

CRITICAL: When finding the highest or lowest value in a table column, you MUST read every row's value and compare them numerically — do NOT rely on memory or general knowledge about which category is largest. Compare by digit count first: 200583 (6 digits) is ALWAYS larger than 85200 (5 digits). A product showing 200583 beats one showing 85200 regardless of their names.

IMPORTANT: When a table snippet shows multiple values in a row without explicit column labels, apply the column order from the "Table columns:" header line or from context evidence. Do not say a year's value is missing when the value is present and the column order is known.

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

IMPORTANT: Evidence blocks with "type": "table" contain real financial or structured data in one of two formats:
- Labeled-row format (primary): starts with "Table columns: col1 | col2 | ..." then one row per line as "Row label: val1 | val2 | ...". Thousands-separator commas are removed — 200583 means 200,583 (two hundred thousand five hundred eighty-three).
- Pipe-delimited format (fallback): rows of cells separated by | characters, e.g. "| iPhone | 200,583 | 205,489 |". Values may include commas as thousands separators — treat them as such, not as decimal points.
Parse and use these values precisely. Never say numerical data is missing if a table block contains the relevant numbers.

CRITICAL: When finding the highest or lowest value in a table column, you MUST read every row's value and compare them numerically — do NOT rely on memory or general knowledge about which category is largest. Compare by digit count first: 200583 (6 digits) is ALWAYS larger than 85200 (5 digits). A product showing 200583 beats one showing 85200 regardless of their names.

IMPORTANT: When a table snippet shows multiple values in a row without explicit column labels, apply the column order from the "Table columns:" header line or from context evidence. Do not say a year's value is missing when the value is present and the column order is known.

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

    # Detect list groups: 2+ short non-table blocks from the same section.
    # Tables are always shown individually so their HTML can be converted to labeled-row format.
    section_indices: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        if (item.section_id and item.type not in ("table", "figure")
                and len(item.text or "") < _LIST_BLOCK_MAX_CHARS):
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
            if item.type == "table" and item.table_html:
                snippet = _html_table_to_llm_text(item.table_html)
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
