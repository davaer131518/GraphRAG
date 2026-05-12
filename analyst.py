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
        rows = self.neo4j.get_document_map_rows(self.settings.document_id)
        markdown = self._document_map_markdown(rows)
        return DocumentMapResult(markdown=markdown, sections=rows)

    def _document_map_markdown(self, rows: list[dict]) -> str:
        if not rows:
            return "# Document Map\n\nNo sections found in the graph."
        parts = ["# Document Map"]
        for row in rows:
            section = " ".join((row.get("section") or "Untitled section").split())
            page = row.get("section_page")
            blocks = row.get("blocks") or []
            counts = Counter(block.get("type") for block in blocks)
            parts.append(f"\n## {section}\n")
            if page is not None:
                parts.append(f"- Starts on page {page}")
            if counts:
                summary = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()) if kind)
                parts.append(f"- Block summary: {summary}")
            key_paragraphs = [b for b in blocks if b.get("type") in {"paragraph", "list_item"}][:3]
            tables = [b for b in blocks if b.get("type") == "table"][:5]
            captions = [b for b in blocks if b.get("type") == "caption"][:3]
            if key_paragraphs:
                parts.append("- Key paragraphs:")
                for block in key_paragraphs:
                    parts.append(f"  - Page {block.get('page')}, `{block.get('block_id')}`: {make_snippet(block.get('text'), 180)}")
            if tables:
                parts.append("- Tables:")
                for block in tables:
                    parts.append(f"  - Page {block.get('page')}, `{block.get('block_id')}`: {make_snippet(block.get('text'), 160)}")
            if captions:
                parts.append("- Captions:")
                for block in captions:
                    parts.append(f"  - Page {block.get('page')}, `{block.get('block_id')}`: {make_snippet(block.get('text'), 160)}")
            relationships = self._flatten_relationships(row.get("relationship_groups") or [])
            rel_counts = Counter(rel.get("rel") for rel in relationships if rel.get("rel"))
            if rel_counts:
                rel_summary = ", ".join(f"{rel}: {count}" for rel, count in sorted(rel_counts.items()))
                parts.append(f"- Relationship summary: {rel_summary}")
        return "\n".join(parts)

    @staticmethod
    def _flatten_relationships(groups: list) -> list[dict]:
        flattened: list[dict] = []
        for group in groups:
            if isinstance(group, list):
                flattened.extend(rel for rel in group if isinstance(rel, dict))
            elif isinstance(group, dict):
                flattened.append(group)
        return flattened
