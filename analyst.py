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
from retrievers.scope import RetrievalScope
from retrievers.scope_resolver import ScopeResolver

logger = logging.getLogger(__name__)


class TraceablePDFAnalyst:
    def __init__(
        self,
        retriever: HybridRetriever,
        answer_generator: AnswerGenerator,
        neo4j: Neo4jClient,
        settings: Settings,
        scope_resolver: ScopeResolver | None = None,
    ) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.neo4j = neo4j
        self.settings = settings
        self.scope_resolver = scope_resolver
        self.last_bundle = None
        self.last_answer = None
        self._doc_cache: list[dict] | None = None

    def _documents(self) -> list[dict]:
        """Lazily cached list of all Document nodes."""
        if self._doc_cache is None:
            self._doc_cache = self.neo4j.list_documents()
        return self._doc_cache

    def _resolve_scope(self, question: str, scope: RetrievalScope | None) -> RetrievalScope:
        """Apply per-turn scope precedence: sticky > query-cues > whole corpus."""
        # 1. Sticky session scope wins (set by /use or /scope command, or DOCUMENT_ID seed).
        #    /scope all sends a whole_corpus() scope whose is_scoped is False, so it falls through.
        if scope is not None and scope.is_scoped:
            return scope

        # 2. Query-derived scope cues
        documents = self._documents()
        corpus_id = documents[0].get("corpus_id") if documents else None
        if self.scope_resolver is not None:
            resolved = self.scope_resolver.resolve(question, documents, corpus_id=corpus_id)
            if resolved.is_scoped:
                return resolved

        # 3. Whole-corpus default
        corpus_id = corpus_id or (documents[0].get("corpus_id") if documents else None)
        return RetrievalScope.whole_corpus(corpus_id=corpus_id)

    def ask(self, question: str, scope: RetrievalScope | None = None) -> AnalystAnswer:
        logger.info("Analyst question: %s", question)
        resolved = self._resolve_scope(question, scope)
        bundle = self.retriever.retrieve(question, resolved)
        answer = self.answer_generator.generate(bundle)
        self.last_bundle = bundle
        self.last_answer = answer
        return answer

    def table_explorer(
        self,
        table_id: str,
        relation: str | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> TableExplorerResult:
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

        # Cross-doc table matches (only when multi-doc scope and no same-doc filter)
        if scope is not None and scope.is_multi_doc and relation is None:
            cross_rows = self.neo4j.get_cross_doc_table_matches(
                table_id,
                document_ids=scope.doc_id_list,
                limit=self.settings.cross_doc_table_limit,
            )
            for row in cross_rows:
                schema_score = row.get("schema_score")
                metric_score = row.get("metric_score")
                reason_parts = []
                if schema_score is not None:
                    reason_parts.append(f"schema_score={schema_score:.2f}")
                if metric_score is not None:
                    reason_parts.append(f"metric_score={metric_score:.2f}")
                related.append(
                    TableRelationship(
                        source_block_id=row["source_block_id"],
                        source_page=row.get("source_page"),
                        source_section=row.get("source_section"),
                        source_snippet=make_snippet(row.get("source_text")),
                        target_block_id=row["target_block_id"],
                        target_page=row.get("target_page"),
                        target_section=row.get("target_section"),
                        target_snippet=make_snippet(row.get("target_text")),
                        relation=row["relationship"],
                        reason=", ".join(reason_parts) or None,
                        is_cross_doc=True,
                        target_doc_id=row.get("doc_id"),
                        target_doc_label=row.get("filename"),
                    )
                )

        traces = [
            f"{rel.source_block_id} -{rel.relation}-> {rel.target_block_id}"
            + (f" [{rel.target_doc_label or rel.target_doc_id}]" if rel.is_cross_doc else "")
            for rel in related
        ]
        return TableExplorerResult(table_id=table_id, relation_filter=relation, related_tables=related, traces=traces)

    def related_documents(
        self,
        doc_id: str,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[dict]:
        """Return RELATED_DOCUMENT neighbors for the given doc_id."""
        return self.neo4j.list_related_documents(
            doc_id,
            document_ids=(scope.doc_id_list if scope else None),
        )

    def document_map(self, scope: RetrievalScope | None = None) -> DocumentMapResult:
        # Determine which doc_ids to scope to
        if scope is not None and scope.is_scoped:
            doc_ids = scope.doc_id_list
        elif self.settings.document_id:
            doc_ids = [self.settings.document_id]
        else:
            doc_ids = None
        rows = self.neo4j.get_document_map_hierarchical(
            doc_ids,
            term_doc_freq_filter=self.settings.term_doc_freq_filter,
        )
        markdown = self._document_map_markdown(rows)
        return DocumentMapResult(markdown=markdown, sections=rows)

    def _document_map_markdown(self, rows: list[dict]) -> str:
        if not rows:
            return "# Document Map\n\nNo sections found in the graph."

        # Group by doc_id to support multi-doc rendering
        doc_order: list[str] = []
        doc_groups: dict[str, list[dict]] = {}
        for row in rows:
            did = row.get("doc_id") or "__unknown__"
            if did not in doc_groups:
                doc_order.append(did)
                doc_groups[did] = []
            doc_groups[did].append(row)

        parts = ["# Document Map"]

        # Single group → today's exact output (no per-doc header — no regression)
        if len(doc_order) <= 1:
            for row in rows:
                parts.extend(self._section_markdown_parts(row))
        else:
            # Multiple groups → per-doc header
            for did in doc_order:
                group = doc_groups[did]
                fname = group[0].get("filename") or did
                parts.append(f"\n## Document: {fname} (`{did}`)")
                for row in group:
                    parts.extend(self._section_markdown_parts(row))

        return "\n".join(parts)

    def _section_markdown_parts(self, row: dict) -> list[str]:
        parts: list[str] = []
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

        if page_start is not None:
            page_info = f"Pages {page_start}–{page_end}" if page_end else f"Page {page_start}"
            extras = []
            if block_count:
                extras.append(f"{block_count} blocks")
            if extras:
                page_info += f", {', '.join(extras)}"
            parts.append(f"- {page_info}")

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

        type_counts = Counter(b.get("type") for b in blocks if b.get("type"))
        if type_counts:
            summary = ", ".join(f"{count} {kind}" for kind, count in sorted(type_counts.items()))
            parts.append(f"- Block summary: {summary}")

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

        return parts
