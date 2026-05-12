from evidence.evidence_bundle import EvidenceBundle, EvidenceItem, TraceStep, make_snippet
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
        section="Risk Factors",
        retrieval_method="vector",
    )
    bundle = EvidenceBundle(
        question="What are the risks?",
        document_id="doc",
        seed_blocks=[item],
        final_evidence=[item],
        trace=[TraceStep(action="vector_search", description="matched", to_id=item.block_id)],
    )
    data = bundle.to_dict()
    assert data["question"] == "What are the risks?"
    assert data["final_evidence"][0]["block_id"] == "p0001_b0001"
    assert "vector_search" in format_bundle_trace(bundle)
