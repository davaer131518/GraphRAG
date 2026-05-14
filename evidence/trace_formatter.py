from __future__ import annotations

import json
from typing import Any

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


def format_sources_markdown(sources: list[dict[str, Any]] | list[Any]) -> str:
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
        mentioned_entities = source.get("mentioned_entities") or []

        lines = [
            f"**Page {page} - Block `{block_id}` - {btype}**",
            "",
            f"Section: {section_title}",
        ]
        if section_path and section_path != section_title:
            lines.append(f"`{section_path}`")
        if mentioned_entities:
            top = mentioned_entities[:5]
            entity_str = ", ".join(
                f"{e.get('name', e.get('entity_name', '?'))} ({e.get('type', e.get('entity_type', '?'))})"
                + (f", ×{e['count']}" if e.get("count") else "")
                for e in top
            )
            lines.append(f"Mentions: {entity_str}")
        lines += [
            "",
            f"Why relevant: {why}",
            "",
            f'Snippet: "{snippet}"',
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_answer_markdown(answer: AnalystAnswer) -> str:
    sections = [
        f"### Answer\n\n{answer.answer}",
        f"### Confidence\n\n{answer.confidence}",
        f"### Sources\n\n{format_sources_markdown(answer.sources)}",
        "### Evidence Trace\n\n```text\n" + "\n".join(answer.trace) + "\n```",
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
        cards.append(
            f"**{rel.relation}: `{rel.source_block_id}` -> `{rel.target_block_id}`**\n\n"
            f"Source: page {rel.source_page}, section {rel.source_section or 'Unknown'}\n\n"
            f"Target: page {rel.target_page}, section {rel.target_section or 'Unknown'}\n\n"
            f"Reason: {rel.reason or 'No relationship reason stored.'}\n\n"
            f"Target snippet: \"{rel.target_snippet}\""
        )
    trace = "\n".join(result.traces)
    return title + "\n\n" + "\n\n".join(cards) + (f"\n\n```text\n{trace}\n```" if trace else "")


def format_document_map(result: DocumentMapResult) -> str:
    return result.markdown


def format_debug_json(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```"
