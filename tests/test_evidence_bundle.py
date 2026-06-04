from evidence.evidence_bundle import EvidenceBundle, EvidenceItem, SourceCitation, TraceStep, make_snippet
from evidence.trace_formatter import format_bundle_trace


def test_make_snippet_collapses_whitespace_and_truncates() -> None:
    snippet = make_snippet("hello\n\nworld " * 40, max_chars=30)
    assert "\n" not in snippet
    assert len(snippet) <= 30
    assert snippet.endswith("...")


def test_evidence_bundle_serializes_required_fields() -> None:
    item = EvidenceItem(
        block_id="p0001_b0001",
        type="paragraph",
        page=1,
        text="Example evidence text",
        section_title="Risk Factors",
        retrieval_method="vector",
    )
    bundle = EvidenceBundle(
        question="What are the risks?",
        document_ids=["doc"],
        seed_blocks=[item],
        final_evidence=[item],
        trace=[TraceStep(action="vector_search", description="matched", to_id=item.block_id)],
    )
    data = bundle.to_dict()
    assert data["question"] == "What are the risks?"
    assert data["final_evidence"][0]["block_id"] == "p0001_b0001"
    assert "vector_search" in format_bundle_trace(bundle)


def test_evidence_item_carries_section_path_and_entities() -> None:
    row = {
        "block_id": "p0002_b0003",
        "type": "paragraph",
        "page": 5,
        "text": "GDPR compliance text.",
        "section_id": "sec_002",
        "section_title": "Regulatory Risks",
        "section_path": "Part I / Regulatory Risks",
        "section_level": 2,
        "confidence": 0.85,
        "scope": "reference",
        "methods": ["citation", "url"],
    }
    item = EvidenceItem.from_row(
        row,
        retrieval_method="graph_expansion",
        metadata={"entity_id": "ent_001", "entity_name": "GDPR", "entity_match_type": "exact"},
    )
    assert item.section_path == "Part I / Regulatory Risks"
    assert item.section_level == 2
    assert item.section_title == "Regulatory Risks"
    assert item.section_id == "sec_002"
    assert item.relationship_confidence == 0.85
    assert item.relationship_scope == "reference"
    assert item.relationship_methods == ["citation", "url"]
    assert item.metadata["entity_match_type"] == "exact"


def test_evidence_item_from_row_defaults_missing_section_fields() -> None:
    row = {"block_id": "p0003_b0000", "type": "table", "page": 3, "text": "Table."}
    item = EvidenceItem.from_row(row, retrieval_method="keyword")
    assert item.section_path is None
    assert item.section_level is None
    assert item.section_id is None
    assert item.mentioned_entities == []
    assert item.matched_entities == []
    assert item.relationship_methods == []


def test_source_citation_carries_section_path_and_entities() -> None:
    src = SourceCitation(
        page=3,
        block_id="p0003_b0000",
        type="table",
        section_title="Data Tables",
        section_path="Part II / Data Tables",
        why_relevant="Contains revenue data.",
        snippet="Revenue: $100M",
        mentioned_entities=[{"entity_id": "ent_rev", "name": "Revenue", "type": "METRIC", "count": 2}],
    )
    data = src.to_dict()
    assert data["section_title"] == "Data Tables"
    assert data["section_path"] == "Part II / Data Tables"
    assert data["mentioned_entities"][0]["name"] == "Revenue"
