from __future__ import annotations

import logging
import re

from config import Settings
from evidence.evidence_bundle import EvidenceItem, TraceStep
from neo4j_client import Neo4jClient

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

    def retrieve(self, question: str) -> tuple[list[EvidenceItem], list[TraceStep]]:
        terms = self.extract_terms(question)
        use_fulltext = self.neo4j.has_index("block_text_fulltext")
        rows = self.neo4j.keyword_search_blocks(
            self._fulltext_query(terms) if use_fulltext else question,
            terms=terms,
            top_k=self.settings.keyword_top_k,
            document_id=self.settings.document_id,
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
        trace = [
            TraceStep(
                action="keyword_search",
                description=f"Keyword search returned {len(items)} seed blocks.",
                method="keyword",
                metadata={"terms": terms, "used_fulltext": use_fulltext},
            )
        ]
        logger.info("Keyword terms: %s", terms)
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
