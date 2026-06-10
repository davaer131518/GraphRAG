from __future__ import annotations

import json
from html.parser import HTMLParser
from typing import Any


class _TableParser(HTMLParser):
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


def _html_table_to_markdown(html: str) -> str:
    p = _TableParser()
    p.feed(html)
    if not p.rows:
        return ""
    max_cols = max(len(r) for r in p.rows)
    rows = [r + [""] * (max_cols - len(r)) for r in p.rows]

    def row_md(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    lines = [row_md(rows[0]), "| " + " | ".join(["---"] * max_cols) + " |"]
    lines += [row_md(r) for r in rows[1:]]
    return "\n".join(lines)

from evidence.evidence_bundle import (
    AnalystAnswer,
    DocumentMapResult,
    EvidenceBundle,
    EvidenceItem,
    TableExplorerResult,
    TraceStep,
)


def format_trace_steps(steps: list[TraceStep]) -> list[str]:
    lines: list[str] = []
    for step in steps:
        score = f" (score={step.score:.3f})" if step.score is not None else ""
        relation = f" via {step.relationship}" if step.relationship else ""
        page = f", page {step.page}" if step.page is not None else ""
        if step.to_id:
            suffix = f" -> {step.to_id}{relation}{page}{score}"
        elif step.from_id:
            suffix = f" [{step.from_id}]"
        else:
            suffix = score
        lines.append(f"{step.action}{suffix}: {step.description}")
    return lines


def format_bundle_trace(bundle: EvidenceBundle) -> str:
    lines = ["Question"]
    if bundle.trace:
        for step in bundle.trace:
            target = step.to_id or step.from_id or ""
            label = step.relationship or step.method or step.action
            detail = f"Block {target}" if target else step.description
            if step.page is not None:
                detail += f", page {step.page}"
            if step.section:
                detail += f", section {step.section}"
            lines.append(f"  -> {label} -> {detail}")
    else:
        for item in bundle.final_evidence:
            path = " -> ".join(item.relationship_path)
            lines.append(f"  -> {path} -> Block {item.block_id}, page {item.page}")
    return "\n".join(lines)


def format_sources_markdown(
    sources: list[dict[str, Any]] | list[Any],
    *,
    show_document: bool = False,
) -> str:
    if not sources:
        return "No sources returned."
    blocks: list[str] = []
    for source in sources:
        if hasattr(source, "to_dict"):
            source = source.to_dict()
        page = source.get("page", "unknown")
        block_id = source.get("block_id", "unknown")
        btype = source.get("type", "unknown")
        section_title = source.get("section_title") or source.get("section") or "Unknown section"
        section_path = source.get("section_path") or ""
        why = source.get("why_relevant") or "Selected by the retrieval pipeline."
        snippet = source.get("snippet") or ""
        table_html = source.get("table_html") or ""
        mentioned_entities = source.get("mentioned_entities") or []
        doc_label = source.get("doc_label") or source.get("doc_id")

        lines = [
            f"**Page {page} - Block `{block_id}` - {btype}**",
            "",
            f"Section: {section_title}",
        ]
        if section_path and section_path != section_title:
            lines.append(f"`{section_path}`")
        if show_document and doc_label:
            lines.append(f"Document: {doc_label}")
        if mentioned_entities:
            top = mentioned_entities[:5]
            entity_str = ", ".join(
                f"{e.get('name', e.get('entity_name', '?'))} ({e.get('type', e.get('entity_type', '?'))})"
                + (f", ×{e['count']}" if e.get("count") else "")
                for e in top
            )
            lines.append(f"Mentions: {entity_str}")
        lines += ["", f"Why relevant: {why}", ""]
        if btype == "table" and table_html:
            md_table = _html_table_to_markdown(table_html)
            lines.append(md_table if md_table else f'Snippet: "{snippet}"')
        else:
            lines.append(f'Snippet: "{snippet}"')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_answer_markdown(answer: AnalystAnswer) -> str:
    bundle = answer.raw_evidence_bundle
    answering_docs = bundle.answering_doc_ids() if hasattr(bundle, "answering_doc_ids") else []
    show_document = len(answering_docs) > 1

    sections = []

    # Scope banner: input scope (how retrieval was scoped) + post-retrieval note (what actually answered)
    if bundle.scope_rationale:
        sections.append(f"> Scope: {bundle.scope_rationale}")
    if bundle.answering_scope_note:
        sections.append(f"> {bundle.answering_scope_note}")

    sections += [
        f"### Answer\n\n{answer.answer}",
        f"### Confidence\n\n{answer.confidence}",
        f"### Sources\n\n{format_sources_markdown(answer.sources, show_document=show_document)}",
    ]
    if answer.limitations:
        sections.append(f"### Limitations\n\n{answer.limitations}")
    return "\n\n".join(sections)


def format_table_explorer(result: TableExplorerResult) -> str:
    title = f"### Table `{result.table_id}` Relationships"
    if result.relation_filter:
        title += f" ({result.relation_filter})"
    if not result.related_tables:
        return title + "\n\nNo related tables found."
    cards = []
    for rel in result.related_tables:
        cross_marker = " *(cross-document)*" if rel.is_cross_doc else ""
        header = f"**{rel.relation}: `{rel.source_block_id}` -> `{rel.target_block_id}`**{cross_marker}"
        tgt_doc_line = f"Target document: {rel.target_doc_label or rel.target_doc_id}" if rel.is_cross_doc and (rel.target_doc_label or rel.target_doc_id) else None
        lines = [
            header,
            "",
            f"Source: page {rel.source_page}, section {rel.source_section or 'Unknown'}",
            f"Target: page {rel.target_page}, section {rel.target_section or 'Unknown'}",
        ]
        if tgt_doc_line:
            lines.append(tgt_doc_line)
        lines += [
            f"Reason: {rel.reason or 'No relationship reason stored.'}",
            "",
            f'Target snippet: "{rel.target_snippet}"',
        ]
        cards.append("\n".join(lines))
    trace = "\n".join(result.traces)
    return title + "\n\n" + "\n\n".join(cards) + (f"\n\n```text\n{trace}\n```" if trace else "")


def format_document_map(result: DocumentMapResult) -> str:
    return result.markdown


def format_related_documents(rows: list[dict[str, Any]], *, source_label: str = "") -> str:
    if not rows:
        return "No related documents found."
    header = f"### Related Documents"
    if source_label:
        header += f" for **{source_label}**"
    cards = []
    for row in rows:
        fname = row.get("filename") or row.get("doc_id") or "Unknown"
        doc_id = row.get("doc_id") or ""
        score = row.get("score")
        evidence_summary = row.get("evidence_summary") or ""
        shared_entities = row.get("shared_canonical_entity_count") or 0
        hv_entities = row.get("high_value_shared_canonical_entity_count") or 0
        similar_sections = row.get("similar_section_count") or 0
        schema_matches = row.get("schema_match_count") or 0
        metrics = row.get("reports_same_metric_count") or 0
        score_str = f"score={score:.2f}" if score is not None else ""
        lines = [
            f"**{fname}** (`{doc_id}`)" + (f" — {score_str}" if score_str else ""),
        ]
        if evidence_summary:
            lines.append(f"Evidence: {evidence_summary}")
        counts = []
        if shared_entities:
            counts.append(f"{shared_entities} shared entities ({hv_entities} high-value)")
        if similar_sections:
            counts.append(f"{similar_sections} similar sections")
        if schema_matches:
            counts.append(f"{schema_matches} table schema matches")
        if metrics:
            counts.append(f"{metrics} shared metrics")
        if counts:
            lines.append("Signals: " + ", ".join(counts))
        family = row.get("doc_family")
        version = row.get("version_id")
        pub = row.get("published_at")
        meta = ", ".join(p for p in [family, version, pub] if p)
        if meta:
            lines.append(f"Metadata: {meta}")
        scope_hint = f"`/scope {doc_id}`"
        lines.append(f"Tip: use {scope_hint} to scope to this document, or `/scope <this_id>,{doc_id}` to compare.")
        cards.append("\n".join(lines))
    return header + "\n\n" + "\n\n".join(cards)


def format_debug_json(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```"
