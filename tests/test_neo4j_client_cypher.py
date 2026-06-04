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
    assert re.search(r"IN_SECTION\]->\(s?:Section\)", src), (
        "Expected IN_SECTION to target :Section nodes"
    )


def test_no_in_section_to_block() -> None:
    src = _source()
    assert not re.search(r"IN_SECTION\]->\([a-z]*:Block", src), (
        "Found IN_SECTION targeting :Block — should target :Section"
    )


def test_no_in_section_to_heading_block() -> None:
    src = _source()
    assert not re.search(r"IN_SECTION\]->\([a-z]*:Block\s*\{[^}]*type.*heading", src), (
        "Found IN_SECTION targeting a heading :Block — should target :Section"
    )


# ---------------------------------------------------------------------------
# SEMANTICALLY_SIMILAR must be undirected
# ---------------------------------------------------------------------------

def test_semantically_similar_is_undirected() -> None:
    src = _source()
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


# ---------------------------------------------------------------------------
# Multi-doc: document_ids IN filter (old equality form must be gone)
# ---------------------------------------------------------------------------

def test_document_ids_in_filter_present() -> None:
    src = _source()
    assert "$document_ids IS NULL OR d.doc_id IN $document_ids" in src, (
        "Expected $document_ids IS NULL OR d.doc_id IN $document_ids filter"
    )
    assert "$document_ids IS NULL OR e.doc_id IN $document_ids" in src, (
        "Expected $document_ids IS NULL OR e.doc_id IN $document_ids filter"
    )
    assert "$document_ids IS NULL OR s.doc_id IN $document_ids" in src, (
        "Expected $document_ids IS NULL OR s.doc_id IN $document_ids filter"
    )


def test_old_equality_document_id_filter_gone() -> None:
    src = _source()
    assert "= $document_id" not in src, (
        "Old '= $document_id' equality filter still present — must be replaced with IN $document_ids"
    )


# ---------------------------------------------------------------------------
# Multi-doc: doc_id and filename projected from seed queries
# ---------------------------------------------------------------------------

def test_doc_id_and_filename_projected() -> None:
    src = _source()
    assert "d.doc_id AS doc_id" in src, (
        "Expected d.doc_id AS doc_id in seed/expansion RETURN projections"
    )
    assert "d.filename AS filename" in src, (
        "Expected d.filename AS filename in seed/expansion RETURN projections"
    )


# ---------------------------------------------------------------------------
# list_documents: discovery query projections
# ---------------------------------------------------------------------------

def test_list_documents_projects_required_fields() -> None:
    src = _source()
    assert "def list_documents" in src, "Expected list_documents method"
    assert "d.corpus_id" in src, "Expected d.corpus_id in list_documents"
    assert "d.doc_family" in src, "Expected d.doc_family in list_documents"
    assert "d.logical_doc_key" in src, "Expected d.logical_doc_key in list_documents"
    assert "d.version_id" in src, "Expected d.version_id in list_documents"
    assert "d.published_at" in src, "Expected d.published_at in list_documents"


# ---------------------------------------------------------------------------
# Cross-doc edges: must be matched undirected (no closing arrow)
# ---------------------------------------------------------------------------

def test_similar_section_is_undirected() -> None:
    src = _source()
    # Must NOT have ]-> immediately after :SIMILAR_SECTION
    directed = re.findall(r"-\[r:SIMILAR_SECTION\]->", src)
    assert not directed, (
        f"Found {len(directed)} directed SIMILAR_SECTION match(es) — must be undirected"
    )


def test_schema_match_reports_same_metric_undirected() -> None:
    src = _source()
    # The cross-doc table query uses alternation — must not be directed
    directed = re.findall(r"-\[r:SCHEMA_MATCH\|REPORTS_SAME_METRIC\]->", src)
    assert not directed, (
        f"Found {len(directed)} directed SCHEMA_MATCH|REPORTS_SAME_METRIC match(es) — must be undirected"
    )


def test_related_document_undirected() -> None:
    src = _source()
    directed = re.findall(r"-\[r:RELATED_DOCUMENT\]->", src)
    assert not directed, (
        f"Found {len(directed)} directed RELATED_DOCUMENT match(es) — must be undirected"
    )


# ---------------------------------------------------------------------------
# Cross-doc edges: decision filter present
# ---------------------------------------------------------------------------

def test_cross_doc_edges_filter_accepted() -> None:
    src = _source()
    assert "coalesce(r.decision, 'accepted') = 'accepted'" in src, (
        "Expected coalesce(r.decision,'accepted')='accepted' filter on cross-doc edges"
    )


# ---------------------------------------------------------------------------
# CanonicalEntity: correct property names (display_name, NOT canonical_name as field)
# ---------------------------------------------------------------------------

def test_canonical_entity_uses_display_name() -> None:
    src = _source()
    assert "ce.display_name" in src, (
        "Expected ce.display_name (not ce.canonical_name) — display_name is the actual KGBuilder property"
    )
    assert "ce.canonical_id" in src, (
        "Expected ce.canonical_id in cross-doc entity expansion"
    )


# ---------------------------------------------------------------------------
# RESOLVES_TO must be directed; e2.doc_id <> e.doc_id must be present
# ---------------------------------------------------------------------------

def test_resolves_to_is_directed() -> None:
    src = _source()
    # Must have at least one directed RESOLVES_TO (Entity->CanonicalEntity)
    assert re.search(r"-\[:RESOLVES_TO\]->", src), (
        "Expected directed -[:RESOLVES_TO]-> in cross-doc entity expansion"
    )


def test_cross_doc_entity_filters_same_doc() -> None:
    src = _source()
    assert "e2.doc_id <> e.doc_id" in src, (
        "Expected e2.doc_id <> e.doc_id cross-doc guard in expand_block_via_canonical_entities"
    )


# ---------------------------------------------------------------------------
# Cross-doc table query uses type(r) for relationship name
# ---------------------------------------------------------------------------

def test_cross_doc_table_uses_type_r() -> None:
    src = _source()
    assert "type(r) AS relationship" in src, (
        "Expected type(r) AS relationship in get_cross_doc_table_matches"
    )
