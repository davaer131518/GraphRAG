from analyst import TraceablePDFAnalyst
from evidence.trace_formatter import format_table_explorer


class FakeNeo4j:
    def get_table_relationships(self, table_id: str, relation: str | None = None) -> list[dict]:
        return [
            {
                "source_block_id": table_id,
                "source_page": 29,
                "source_text": "Source table text",
                "source_section": "Repurchases",
                "target_block_id": "p0030_b0003",
                "target_page": 30,
                "target_text": "Target table text",
                "target_section": "Stock Performance",
                "relation": relation or "SUPPLEMENTS",
                "reason": "Related table reason",
            }
        ]

    def get_document_map_rows(self, document_id: str | None) -> list[dict]:
        return [
            {
                "section": "Risk Factors",
                "section_page": 10,
                "blocks": [
                    {"block_id": "p0017_b0000", "type": "paragraph", "page": 17, "text": "App Store risk text"},
                    {"block_id": "p0029_b0007", "type": "table", "page": 29, "text": "Table text"},
                ],
                "relationship_groups": [[{"rel": "REFERS_TO"}]],
            }
        ]


class FakeSettings:
    document_id = None


def test_table_explorer_formats_relationships() -> None:
    analyst = TraceablePDFAnalyst(None, None, FakeNeo4j(), FakeSettings())  # type: ignore[arg-type]
    result = analyst.table_explorer("p0029_b0007", "SUPPLEMENTS")
    markdown = format_table_explorer(result)
    assert "SUPPLEMENTS" in markdown
    assert "p0030_b0003" in markdown


def test_document_map_returns_markdown() -> None:
    analyst = TraceablePDFAnalyst(None, None, FakeNeo4j(), FakeSettings())  # type: ignore[arg-type]
    result = analyst.document_map()
    assert "# Document Map" in result.markdown
    assert "Risk Factors" in result.markdown
