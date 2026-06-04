from __future__ import annotations

import logging
from collections import Counter

from config import Settings
from evidence.evidence_bundle import EvidenceItem, TraceStep
from neo4j_client import Neo4jClient
from retrievers.keyword_retriever import KeywordRetriever
from retrievers.scope import RetrievalScope

logger = logging.getLogger(__name__)


class EntityRetriever:
    """Seed retriever: matches question terms against :Entity nodes via MENTIONS edges."""

    def __init__(self, neo4j: Neo4jClient, settings: Settings) -> None:
        self.neo4j = neo4j
        self.settings = settings

    def retrieve(
        self, question: str, scope: RetrievalScope | None = None
    ) -> tuple[list[EvidenceItem], list[TraceStep]]:
        scope = scope or RetrievalScope.whole_corpus()
        terms = KeywordRetriever.extract_terms(question)
        terms_lower = list({t.lower() for t in terms})
        rows = self.neo4j.entity_search_blocks(
            terms_lower,
            top_k=self.settings.entity_top_k,
            term_doc_freq_filter=self.settings.term_doc_freq_filter,
            document_ids=scope.doc_id_list,
        )
        items = [self._row_to_item(row) for row in rows]

        match_type_counts: Counter[str] = Counter(r.get("entity_match_type", "?") for r in rows)
        unique_entities = len({r.get("entity_id") for r in rows if r.get("entity_id")})
        trace = [
            TraceStep(
                action="entity_search",
                description=(
                    f"Entity search returned {len(items)} seed blocks "
                    f"across {unique_entities} entities "
                    f"({', '.join(f'{v} {k}' for k, v in sorted(match_type_counts.items()))})."
                ),
                method="entity",
                metadata={
                    "terms": terms_lower,
                    "top_k": self.settings.entity_top_k,
                    "match_type_counts": dict(match_type_counts),
                },
            )
        ]
        logger.info("Entity seeds: %s blocks, %s entities", len(items), unique_entities)
        return items, trace

    def _row_to_item(self, row: dict) -> EvidenceItem:
        entity_name = row.get("entity_name") or ""
        entity_type = row.get("entity_type") or ""
        match_type = row.get("entity_match_type", "partial")
        match_score = float(row.get("entity_match_score") or 0.6)

        item = EvidenceItem.from_row(
            row,
            retrieval_method="entity",
            relationship_path=[
                f"query_entity_match({entity_name!r}, {match_type}) -> {row.get('block_id')}"
            ],
            why_relevant=f"Mentions entity '{entity_name}' ({entity_type}); match: {match_type}.",
            metadata={
                "matched_entity_id": row.get("entity_id"),
                "matched_entity_name": entity_name,
                "matched_entity_type": entity_type,
                "entity_match_type": match_type,
                "entity_match_score": match_score,
                "mention_count": row.get("mention_count"),
                "mention_confidence": row.get("mention_confidence"),
                "mention_methods": list(row.get("mention_methods") or []),
            },
        )
        item.matched_entities = [
            {
                "entity_id": row.get("entity_id"),
                "entity_name": entity_name,
                "entity_type": entity_type,
                "entity_match_type": match_type,
                "entity_match_score": match_score,
                "mention_confidence": row.get("mention_confidence"),
                "mention_count": row.get("mention_count"),
            }
        ]
        return item
