from __future__ import annotations

from analyst import TraceablePDFAnalyst
from evidence.trace_formatter import format_table_explorer
from tests.fakes import FakeNeo4j


class FakeSettings:
    document_id = None
    term_doc_freq_filter = 0.25


def test_table_explorer_formats_relationships() -> None:
    analyst = TraceablePDFAnalyst(None, None, FakeNeo4j(), FakeSettings())  # type: ignore[arg-type]
    result = analyst.table_explorer("p0029_b0007", "SUPPLEMENTS")
    markdown = format_table_explorer(result)
    assert "SUPPLEMENTS" in markdown
    assert "p0030_b0003" in markdown


def test_table_explorer_normalises_table_relates_to_label() -> None:
    """TABLE_RELATES_TO rows must render as their logical label (SUPPLEMENTS, etc.)."""

    class TableRelatesToFakeNeo4j(FakeNeo4j):
        def get_table_relationships(self, table_id, relation=None):
            return [
                {
                    "source_block_id": table_id,
                    "source_page": 5,
                    "source_text": "Source text.",
                    "source_section": "Method",
                    "target_block_id": "p0006_b0001",
                    "target_page": 6,
                    "target_text": "Target text.",
                    "target_section": "Results",
                    # coalesce(r.label, type(r)) returns the logical label
                    "relation": "SUPPLEMENTS",
                    "reason": "Supplements the source table.",
                }
            ]

    analyst = TraceablePDFAnalyst(None, None, TableRelatesToFakeNeo4j(), FakeSettings())  # type: ignore[arg-type]
    result = analyst.table_explorer("p0005_b0000")
    markdown = format_table_explorer(result)
    assert "SUPPLEMENTS" in markdown
    assert "TABLE_RELATES_TO" not in markdown


def test_document_map_returns_markdown() -> None:
    analyst = TraceablePDFAnalyst(None, None, FakeNeo4j(), FakeSettings())  # type: ignore[arg-type]
    result = analyst.document_map()
    assert "# Document Map" in result.markdown
    assert "Risk Factors" in result.markdown


def test_document_map_renders_section_path_and_level() -> None:
    analyst = TraceablePDFAnalyst(None, None, FakeNeo4j(), FakeSettings())  # type: ignore[arg-type]
    result = analyst.document_map()
    # FakeNeo4j returns a level-2 section; heading should be ### (level+1=3)
    assert "### Risk Factors" in result.markdown
    # section path should appear as a breadcrumb
    assert "Part I / Risk Factors" in result.markdown


def test_document_map_renders_top_entities() -> None:
    analyst = TraceablePDFAnalyst(None, None, FakeNeo4j(), FakeSettings())  # type: ignore[arg-type]
    result = analyst.document_map()
    # FakeNeo4j blocks include entity "App Store"
    assert "App Store" in result.markdown
