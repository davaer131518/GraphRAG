"""Shared test fakes for the retrieval pipeline.

FakeNeo4j covers every method called by the analyst/retrievers.
Default return values use new-schema row shapes (Section nodes, Entity nodes).
Individual tests override specific methods by subclassing or monkey-patching.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Canonical new-schema row shapes (minimal, valid defaults)
# ---------------------------------------------------------------------------

def _block_row(
    block_id: str = "p0001_b0000",
    btype: str = "paragraph",
    page: int = 1,
    text: str = "Example block text.",
    score: float | None = None,
    section_id: str | None = "sec_001",
    section_title: str | None = "Risk Factors",
    section_path: str | None = "Part I / Risk Factors",
    section_level: int | None = 2,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "block_id": block_id,
        "type": btype,
        "page": page,
        "reading_order": 0,
        "text": text,
        "section_id": section_id,
        "section_title": section_title,
        "section_path": section_path,
        "section_level": section_level,
    }
    if score is not None:
        row["score"] = score
    row.update(extra)
    return row


def _entity_block_row(
    block_id: str = "p0001_b0000",
    entity_id: str = "ent_001",
    entity_name: str = "App Store",
    entity_type: str = "ORG",
    entity_match_type: str = "exact",
    entity_match_score: float = 1.0,
    mention_count: int = 5,
    mention_confidence: float = 0.9,
    **extra: Any,
) -> dict[str, Any]:
    row = _block_row(block_id=block_id, **extra)
    row.update({
        "entity_id": entity_id,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "entity_confidence": 0.9,
        "entity_match_type": entity_match_type,
        "entity_match_score": entity_match_score,
        "mention_count": mention_count,
        "mention_confidence": mention_confidence,
        "mention_methods": ["ner"],
    })
    return row


def _section_map_row(
    section_id: str = "sec_001",
    title: str = "Risk Factors",
    path: str = "Part I / Risk Factors",
    level: int = 2,
    page_start: int = 10,
    page_end: int = 25,
    block_count: int = 3,
    blocks: list[dict] | None = None,
    child_section_ids: list[str] | None = None,
) -> dict[str, Any]:
    if blocks is None:
        blocks = [
            {
                "block_id": "p0010_b0001",
                "type": "paragraph",
                "page": 10,
                "text": "App Store risk text.",
                "entities": [
                    {"entity_id": "ent_001", "name": "App Store", "type": "ORG", "count": 8},
                ],
            },
            {
                "block_id": "p0011_b0003",
                "type": "table",
                "page": 11,
                "text": "Revenue breakdown table.",
                "entities": [],
            },
        ]
    return {
        "section_id": section_id,
        "title": title,
        "path": path,
        "level": level,
        "page_start": page_start,
        "page_end": page_end,
        "block_count": block_count,
        "blocks": blocks,
        "child_section_ids": child_section_ids or [],
    }


def _table_rel_row(
    source_block_id: str = "p0029_b0007",
    target_block_id: str = "p0030_b0003",
    relation: str = "SUPPLEMENTS",
    reason: str = "Related table reason",
) -> dict[str, Any]:
    return {
        "source_block_id": source_block_id,
        "source_page": 29,
        "source_text": "Source table text.",
        "source_section": "Repurchases",
        "target_block_id": target_block_id,
        "target_page": 30,
        "target_text": "Target table text.",
        "target_section": "Stock Performance",
        "relation": relation,
        "reason": reason,
    }


def _expand_row(
    block_id: str = "p0002_b0000",
    relationship: str = "REFERS_TO",
    score: float | None = None,
    confidence: float | None = None,
    scope: str | None = None,
    methods: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    row = _block_row(block_id=block_id, score=score, **extra)
    row.update({
        "relationship": relationship,
        "methods": methods,
        "mention": None,
        "reason": None,
        "confidence": confidence,
        "scope": scope,
    })
    return row


def _entity_expand_row(
    block_id: str = "p0003_b0000",
    entity_id: str = "ent_001",
    entity_name: str = "App Store",
    entity_type: str = "ORG",
    mention_count: int = 3,
    mention_confidence: float = 0.8,
    **extra: Any,
) -> dict[str, Any]:
    row = _block_row(block_id=block_id, **extra)
    row.update({
        "entity_id": entity_id,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "mention_count": mention_count,
        "mention_confidence": mention_confidence,
    })
    return row


def _section_expand_row(
    block_id: str = "p0001_b0005",
    section_distance: int = 2,
    structural_boost: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    row = _block_row(block_id=block_id, **extra)
    row.update({
        "section_distance": section_distance,
        "structural_boost": structural_boost,
    })
    return row


# ---------------------------------------------------------------------------
# FakeNeo4j
# ---------------------------------------------------------------------------

class FakeNeo4j:
    """Drop-in replacement for Neo4jClient in tests.

    All methods return minimal new-schema rows by default.
    Override any method in a subclass or via monkey-patching for specific test needs.
    """

    # Called by analyst / retrievers ------------------------------------------

    def vector_search_blocks(
        self, embedding: list[float], *, top_k: int, document_id: str | None
    ) -> list[dict[str, Any]]:
        return [_block_row(block_id="p0001_b0000", score=0.85)]

    def keyword_search_blocks(
        self,
        query_text: str,
        *,
        terms: list[str],
        top_k: int,
        document_id: str | None,
        use_fulltext: bool,
    ) -> list[dict[str, Any]]:
        return [_block_row(block_id="p0001_b0001", score=1.0)]

    def section_title_search_blocks(
        self,
        terms: list[str],
        *,
        top_k: int,
        document_id: str | None,
        term_min_len: int = 4,
    ) -> list[dict[str, Any]]:
        return [_block_row(
            block_id="p0010_b0001",
            section_title="Third Quarter of 2023",
            section_path="Part I / Third Quarter of 2023",
            text="MacBook Air 15-inch, Mac Studio, Mac Pro, Apple Vision Pro.",
            score=3.0,
        )]

    def entity_search_blocks(
        self,
        terms_lower: list[str],
        *,
        top_k: int,
        term_doc_freq_filter: float,
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        return [_entity_block_row(block_id="p0001_b0002")]

    def expand_block(
        self,
        block_id: str,
        *,
        block_type: str,
        semantic_similarity_threshold: float,
        limit: int,
        global_threshold: float = 2.0,
    ) -> list[dict[str, Any]]:
        return [_expand_row(block_id="p0002_b0000", relationship="REFERS_TO")]

    def expand_block_via_entities(
        self,
        block_id: str,
        *,
        entities_per_seed: int,
        blocks_per_entity: int,
        term_doc_freq_filter: float,
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        return [_entity_expand_row(block_id="p0003_b0000")]

    def expand_block_via_section(
        self,
        block_id: str,
        *,
        limit: int,
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        return [_section_expand_row(block_id="p0001_b0005", section_distance=1)]

    def get_table_relationships(
        self, table_id: str, relation: str | None = None
    ) -> list[dict[str, Any]]:
        return [_table_rel_row(source_block_id=table_id, relation=relation or "SUPPLEMENTS")]

    def get_document_map_hierarchical(
        self,
        document_id: str | None,
        *,
        term_doc_freq_filter: float = 0.25,
    ) -> list[dict[str, Any]]:
        return [_section_map_row()]

    # Infrastructure ----------------------------------------------------------

    def has_index(self, name: str) -> bool:
        return True

    def verify_connectivity(self) -> None:
        pass

    def ensure_fulltext_index(self) -> None:
        pass

    def close(self) -> None:
        pass
