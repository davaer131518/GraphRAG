from __future__ import annotations

import logging
import re

from config import Settings
from evidence.evidence_bundle import EvidenceItem, TraceStep
from neo4j_client import Neo4jClient
from retrievers.scope import RetrievalScope

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "does",
    "for",
    "from",
    "how",
    "into",
    "related",
    "say",
    "says",
    "the",
    "this",
    "that",
    "what",
    "when",
    "where",
    "which",
    "with",
}

_PHRASE_RE = re.compile(r"[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.'-]{2,}|\$?\d+(?:\.\d+)?%?")


class KeywordRetriever:
    def __init__(self, neo4j: Neo4jClient, settings: Settings) -> None:
        self.neo4j = neo4j
        self.settings = settings

    def retrieve(
        self, question: str, scope: RetrievalScope | None = None
    ) -> tuple[list[EvidenceItem], list[TraceStep]]:
        scope = scope or RetrievalScope.whole_corpus()
        terms = self.extract_terms(question)
        use_fulltext = self.neo4j.has_index("block_text_fulltext")
        rows = self.neo4j.keyword_search_blocks(
            self._fulltext_query(terms) if use_fulltext else question,
            terms=terms,
            top_k=self.settings.keyword_top_k,
            document_ids=scope.doc_id_list,
            use_fulltext=use_fulltext,
        )
        items = [
            EvidenceItem.from_row(
                row,
                retrieval_method="keyword",
                relationship_path=[f"query_keyword_match -> {row['block_id']}"],
                why_relevant="Contains exact or near-exact terms from the question.",
                metadata={"terms": terms},
            )
            for row in rows
        ]

        section_title_items: list[EvidenceItem] = []
        if self.settings.enable_section_title_search:
            seen_ids = {row["block_id"] for row in rows}
            st_rows = self.neo4j.section_title_search_blocks(
                terms,
                top_k=self.settings.keyword_top_k,
                document_ids=scope.doc_id_list,
            )
            for row in st_rows:
                if row["block_id"] not in seen_ids:
                    section_title_items.append(
                        EvidenceItem.from_row(
                            row,
                            retrieval_method="section_title",
                            relationship_path=[
                                f"query_section_title_match({row.get('section_title')!r}) "
                                f"-> {row['block_id']}"
                            ],
                            why_relevant=(
                                f"Found within section '{row.get('section_title')}' "
                                "whose title matches the question."
                            ),
                            metadata={"terms": terms},
                        )
                    )
                    seen_ids.add(row["block_id"])

        all_items = items + section_title_items
        trace = [
            TraceStep(
                action="keyword_search",
                description=(
                    f"Keyword search returned {len(items)} blocks; "
                    f"section-title search returned {len(section_title_items)} additional blocks."
                ),
                method="keyword",
                metadata={
                    "terms": terms,
                    "used_fulltext": use_fulltext,
                    "section_title_items": len(section_title_items),
                },
            )
        ]
        logger.info("Keyword terms: %s; section-title extras: %s", terms, len(section_title_items))
        return all_items, trace

    def retrieve_tables(
        self, question: str, scope: RetrievalScope | None = None
    ) -> tuple[list[EvidenceItem], list[TraceStep]]:
        """Seed table blocks directly from question terms, bypassing prose competition."""
        scope = scope or RetrievalScope.whole_corpus()
        terms = self.extract_terms(question)
        use_fulltext = self.neo4j.has_index("block_text_fulltext")
        rows = self.neo4j.table_keyword_search(
            self._fulltext_query(terms) if use_fulltext else question,
            terms=terms,
            top_k=self.settings.table_top_k,
            document_ids=scope.doc_id_list,
            use_fulltext=use_fulltext,
        )
        items = [
            EvidenceItem.from_row(
                row,
                retrieval_method="table_keyword",
                relationship_path=[f"query_table_keyword_match -> {row['block_id']}"],
                why_relevant="Table block whose row labels match question terms.",
                metadata={"terms": terms},
            )
            for row in rows
        ]
        trace = [
            TraceStep(
                action="table_search",
                description=f"Table keyword search returned {len(items)} table blocks.",
                method="table_keyword",
                metadata={"terms": terms, "used_fulltext": use_fulltext},
            )
        ]
        logger.info("Table seed search: %s table blocks", len(items))
        return items, trace

    @staticmethod
    def extract_terms(question: str) -> list[str]:
        phrases = [p.strip() for p in _PHRASE_RE.findall(question) if len(p.strip()) > 3]
        tokens = [
            t.strip()
            for t in _TOKEN_RE.findall(question)
            if t.strip().lower() not in _STOPWORDS and len(t.strip()) > 2
        ]
        ordered: list[str] = []
        for term in [*phrases, *tokens]:
            if term.lower() not in {existing.lower() for existing in ordered}:
                ordered.append(term)
        return ordered[:12]

    @staticmethod
    def _fulltext_query(terms: list[str]) -> str:
        if not terms:
            return ""
        formatted = []
        for term in terms:
            escaped = term.replace('"', '\\"')
            if " " in term:
                formatted.append(f'"{escaped}"')
            else:
                formatted.append(escaped)
        return " OR ".join(formatted)
