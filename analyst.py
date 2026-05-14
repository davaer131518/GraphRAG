from __future__ import annotations

import logging
from collections import Counter

from config import Settings
from evidence.evidence_bundle import (
    AnalystAnswer,
    DocumentMapResult,
    TableExplorerResult,
    TableRelationship,
    make_snippet,
)
from generation.answer_generator import AnswerGenerator
from neo4j_client import Neo4jClient, TABLE_RELATION_TYPES
from retrievers.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class TraceablePDFAnalyst:
    def __init__(
        self,
        retriever: HybridRetriever,
        answer_generator: AnswerGenerator,
        neo4j: Neo4jClient,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.neo4j = neo4j
        self.settings = settings
        self.last_bundle = None
        self.last_answer = None

    def ask(self, question: str) -> AnalystAnswer:
        logger.info("Analyst question: %s", question)
        bundle = self.retriever.retrieve(question)
        answer = self.answer_generator.generate(bundle)
        self.last_bundle = bundle
        self.last_answer = answer
        return answer

    def table_explorer(self, table_id: str, relation: str | None = None) -> TableExplorerResult:
        relation = relation.upper() if relation else None
        if relation is not None and relation not in TABLE_RELATION_TYPES:
            raise ValueError(f"Relation must be one of: {', '.join(sorted(TABLE_RELATION_TYPES))}")
        rows = self.neo4j.get_table_relationships(table_id, relation)
        related = [
            TableRelationship(
                source_block_id=row["source_block_id"],
                source_page=row.get("source_page"),
                source_section=row.get("source_section"),
                source_snippet=make_snippet(row.get("source_text")),
                target_block_id=row["target_block_id"],
                target_page=row.get("target_page"),
                target_section=row.get("target_section"),
                target_snippet=make_snippet(row.get("target_text")),
                relation=row["relation"],
                reason=row.get("reason"),
            )
            for row in rows
        ]
        traces = [
            f"{rel.source_block_id} -{rel.relation}-> {rel.target_block_id}"
            for rel in related
        ]
        return TableExplorerResult(table_id=table_id, relation_filter=relation, related_tables=related, traces=traces)

    def document_map(self) -> DocumentMapResult:
        rows = self.neo4j.get_document_map_hierarchical(
            self.settings.document_id,
            term_doc_freq_filter=self.settings.term_doc_freq_filter,
        )
        markdown = self._document_map_markdown(rows)
        return DocumentMapResult(markdown=markdown, sections=rows)

    def _document_map_markdown(self, rows: list[dict]) -> str:
        if not rows:
            return "# Document Map\n\nNo sections found in the graph."

        parts = ["# Document Map"]
        for row in rows:
            title = " ".join((row.get("title") or "Untitled section").split())
            path = row.get("path") or ""
            level = int(row.get("level") or 1)
            page_start = row.get("page_start")
            page_end = row.get("page_end")
            block_count = row.get("block_count")
            blocks = row.get("blocks") or []

            # Heading depth: level 1 → ##, level 2 → ###, etc. (clamped at ######)
            heading = "#" * min(level + 1, 6)
            path_display = f" `{path}`" if path and path != title else ""
            parts.append(f"\n{heading} {title}{path_display}")

            # Page range
            if page_start is not None:
                page_info = f"Pages {page_start}–{page_end}" if page_end else f"Page {page_start}"
                extras = []
                if block_count:
                    extras.append(f"{block_count} blocks")
                if extras:
                    page_info += f", {', '.join(extras)}"
                parts.append(f"- {page_info}")

            # Top entities across all blocks in this section
            entity_counts: Counter[tuple[str, str]] = Counter()
            for block in blocks:
                for ent in (block.get("entities") or []):
                    name = ent.get("name") or ""
                    etype = ent.get("type") or ""
                    cnt = int(ent.get("count") or 1)
                    if name:
                        entity_counts[(name, etype)] += cnt
            top_entities = entity_counts.most_common(5)
            if top_entities:
                entity_str = ", ".join(f"{name} ({etype}, {count})" for (name, etype), count in top_entities)
                parts.append(f"- Top entities: {entity_str}")

            # Block summary
            type_counts = Counter(b.get("type") for b in blocks if b.get("type"))
            if type_counts:
                summary = ", ".join(f"{count} {kind}" for kind, count in sorted(type_counts.items()))
                parts.append(f"- Block summary: {summary}")

            # Key blocks
            key_paragraphs = [b for b in blocks if b.get("type") in {"paragraph", "list_item"}][:3]
            tables = [b for b in blocks if b.get("type") == "table"][:5]
            captions = [b for b in blocks if b.get("type") == "caption"][:3]

            if key_paragraphs:
                parts.append("- Key paragraphs:")
                for block in key_paragraphs:
                    parts.append(
                        f"  - Page {block.get('page')}, `{block.get('block_id')}`: "
                        f"{make_snippet(block.get('text'), 180)}"
                    )
            if tables:
                parts.append("- Tables:")
                for block in tables:
                    parts.append(
                        f"  - Page {block.get('page')}, `{block.get('block_id')}`: "
                        f"{make_snippet(block.get('text'), 160)}"
                    )
            if captions:
                parts.append("- Captions:")
                for block in captions:
                    parts.append(
                        f"  - Page {block.get('page')}, `{block.get('block_id')}`: "
                        f"{make_snippet(block.get('text'), 160)}"
                    )

        return "\n".join(parts)
