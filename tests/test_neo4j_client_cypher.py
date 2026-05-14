"""Cypher-shape guard tests.

These tests verify that the Cypher in neo4j_client.py has been updated
to the current KGBuilder schema. They fail loudly if a future change
accidentally reintroduces old patterns (IN_SECTION to a Block, directed
SEMANTICALLY_SIMILAR, missing TABLE_RELATES_TO, etc.).
"""
from __future__ import annotations

import pathlib
import re

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "neo4j_client.py"


def _source() -> str:
    return SOURCE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# IN_SECTION must target :Section, never :Block or a plain alias
# ---------------------------------------------------------------------------

def test_in_section_targets_section_node() -> None:
    src = _source()
    # Must contain at least one IN_SECTION targeting :Section
    assert re.search(r"IN_SECTION\]->\(s?:Section\)", src), (
        "Expected IN_SECTION to target :Section nodes"
    )


def test_no_in_section_to_block() -> None:
    src = _source()
    # Must NOT contain IN_SECTION targeting a :Block
    assert not re.search(r"IN_SECTION\]->\([a-z]*:Block", src), (
        "Found IN_SECTION targeting :Block — should target :Section"
    )


def test_no_in_section_to_heading_block() -> None:
    src = _source()
    # Must NOT reference IN_SECTION on a heading alias like h:Block
    assert not re.search(r"IN_SECTION\]->\([a-z]*:Block\s*\{[^}]*type.*heading", src), (
        "Found IN_SECTION targeting a heading :Block — should target :Section"
    )


# ---------------------------------------------------------------------------
# SEMANTICALLY_SIMILAR must be undirected
# ---------------------------------------------------------------------------

def test_semantically_similar_is_undirected() -> None:
    src = _source()
    # All occurrences of the match pattern should be undirected (no -> after the rel)
    directed = re.findall(r"-\[r:SEMANTICALLY_SIMILAR\]->", src)
    assert not directed, (
        f"Found {len(directed)} directed SEMANTICALLY_SIMILAR match(es) — must be undirected"
    )


def test_semantically_similar_uses_scope_filter() -> None:
    src = _source()
    assert "r.scope" in src, "Expected r.scope filtering for SEMANTICALLY_SIMILAR"


# ---------------------------------------------------------------------------
# TABLE_RELATES_TO must be present and normalised
# ---------------------------------------------------------------------------

def test_table_relates_to_in_expand_table() -> None:
    src = _source()
    assert "TABLE_RELATES_TO" in src, (
        "Expected TABLE_RELATES_TO in the table expansion union"
    )


def test_coalesce_label_normalisation() -> None:
    src = _source()
    assert "coalesce(r.label, type(r))" in src, (
        "Expected coalesce(r.label, type(r)) to normalise TABLE_RELATES_TO label"
    )


# ---------------------------------------------------------------------------
# Section node: :Section must appear in RETURN projections
# ---------------------------------------------------------------------------

def test_section_node_projected_in_returns() -> None:
    src = _source()
    assert "s.section_id AS section_id" in src, (
        "Expected s.section_id AS section_id in query RETURN projections"
    )
    assert "s.title AS section_title" in src, (
        "Expected s.title AS section_title in query RETURN projections"
    )
    assert "s.path AS section_path" in src, (
        "Expected s.path AS section_path in query RETURN projections"
    )


# ---------------------------------------------------------------------------
# REFERS_TO: confidence and scope must be projected
# ---------------------------------------------------------------------------

def test_refers_to_confidence_and_scope_projected() -> None:
    src = _source()
    assert "r.confidence" in src, "Expected r.confidence to be projected from REFERS_TO"
    assert "r.scope" in src, "Expected r.scope to be projected from REFERS_TO"


# ---------------------------------------------------------------------------
# Entity search: entity_match_type and entity_match_score must exist
# ---------------------------------------------------------------------------

def test_entity_search_match_type_and_score() -> None:
    src = _source()
    assert "entity_match_type" in src, "Expected entity_match_type in entity_search_blocks"
    assert "entity_match_score" in src, "Expected entity_match_score in entity_search_blocks"


# ---------------------------------------------------------------------------
# Document map: must use HAS_SECTION/HAS_SUBSECTION (hierarchical)
# ---------------------------------------------------------------------------

def test_document_map_uses_has_section() -> None:
    src = _source()
    assert "HAS_SECTION" in src, (
        "Expected HAS_SECTION in get_document_map_hierarchical"
    )
    assert "HAS_SUBSECTION" in src, (
        "Expected HAS_SUBSECTION traversal in get_document_map_hierarchical"
    )


def test_no_old_get_document_map_rows() -> None:
    src = _source()
    assert "get_document_map_rows" not in src, (
        "Old get_document_map_rows still present — should be replaced by get_document_map_hierarchical"
    )
