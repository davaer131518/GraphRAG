from __future__ import annotations

import logging

from config import Settings
from embeddings_client import EmbeddingsClient
from evidence.evidence_bundle import EvidenceItem, TraceStep
from neo4j_client import Neo4jClient
from retrievers.scope import RetrievalScope

logger = logging.getLogger(__name__)


class SemanticRetriever:
    def __init__(self, neo4j: Neo4jClient, embeddings: EmbeddingsClient, settings: Settings) -> None:
        self.neo4j = neo4j
        self.embeddings = embeddings
        self.settings = settings

    def retrieve(
        self, question: str, scope: RetrievalScope | None = None
    ) -> tuple[list[EvidenceItem], list[TraceStep]]:
        scope = scope or RetrievalScope.whole_corpus()
        query_embedding = self.embeddings.embed_query(question)
        rows = self.neo4j.vector_search_blocks(
            query_embedding,
            top_k=self.settings.vector_top_k,
            document_ids=scope.doc_id_list,
        )
        items = [
            EvidenceItem.from_row(
                row,
                retrieval_method="vector",
                relationship_path=[f"query_vector_match -> {row['block_id']}"],
                why_relevant="Semantically similar to the question.",
            )
            for row in rows
        ]
        trace = [
            TraceStep(
                action="vector_search",
                description=f"Vector search returned {len(items)} seed blocks.",
                method="vector",
                metadata={"top_k": self.settings.vector_top_k},
            )
        ]
        logger.info("Semantic seeds: %s", [item.block_id for item in items])
        return items, trace
