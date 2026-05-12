from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from config import Settings
from evidence.evidence_bundle import make_snippet

logger = logging.getLogger(__name__)


TABLE_RELATION_TYPES = {"COMPARES", "SUPPLEMENTS", "CONTRASTS", "ABLATES"}


class Neo4jClient:
    def __init__(self, settings: Settings) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Install the neo4j package to use the Traceable PDF Analyst.") from exc
        self.settings = settings
        self.database = settings.neo4j_database
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()
        logger.info("Connected to Neo4j")

    def run(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return session.run(query, params or {}).data()

    def list_indexes(self) -> list[dict[str, Any]]:
        try:
            return self.run(
                """
                SHOW INDEXES YIELD name, type, state
                RETURN name, type, state
                ORDER BY name
                """
            )
        except Exception as exc:
            logger.warning("Could not list Neo4j indexes: %s", exc)
            return []

    def has_index(self, name: str) -> bool:
        return any(row.get("name") == name and row.get("state") == "ONLINE" for row in self.list_indexes())

    def ensure_fulltext_index(self) -> None:
        if not self.settings.create_fulltext_index:
            return
        self.run(
            """
            CREATE FULLTEXT INDEX block_text_fulltext IF NOT EXISTS
            FOR (b:Block) ON EACH [b.text]
            """
        )
        logger.info("Ensured full-text index block_text_fulltext")

    def vector_search_blocks(
        self,
        embedding: list[float],
        *,
        top_k: int,
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        rows = self.run(
            """
            CALL db.index.vector.queryNodes('block_embedding_index', $top_k, $embedding)
            YIELD node, score
            MATCH (node)-[:ON_PAGE]->(p:Page)-[:PART_OF]->(d:Document)
            WHERE $document_id IS NULL OR d.doc_id = $document_id
            OPTIONAL MATCH (node)-[:IN_SECTION]->(section:Block {type:'heading'})
            RETURN node.block_id AS block_id,
                   node.type AS type,
                   node.page_number AS page,
                   node.reading_order AS reading_order,
                   node.text AS text,
                   section.block_id AS section_id,
                   section.text AS section,
                   score AS score
            ORDER BY score DESC
            """,
            {"top_k": top_k, "embedding": embedding, "document_id": document_id},
        )
        logger.info("Vector search returned %s blocks", len(rows))
        return rows

    def keyword_search_blocks(
        self,
        query_text: str,
        *,
        terms: list[str],
        top_k: int,
        document_id: str | None,
        use_fulltext: bool,
    ) -> list[dict[str, Any]]:
        if use_fulltext:
            try:
                rows = self.run(
                    """
                    CALL db.index.fulltext.queryNodes('block_text_fulltext', $query, {limit: $top_k})
                    YIELD node, score
                    MATCH (node)-[:ON_PAGE]->(:Page)-[:PART_OF]->(d:Document)
                    WHERE $document_id IS NULL OR d.doc_id = $document_id
                    OPTIONAL MATCH (node)-[:IN_SECTION]->(section:Block {type:'heading'})
                    RETURN node.block_id AS block_id,
                           node.type AS type,
                           node.page_number AS page,
                           node.reading_order AS reading_order,
                           node.text AS text,
                           section.block_id AS section_id,
                           section.text AS section,
                           score AS score
                    ORDER BY score DESC
                    """,
                    {"query": query_text, "top_k": top_k, "document_id": document_id},
                )
                logger.info("Full-text search returned %s blocks", len(rows))
                return rows
            except Exception as exc:
                logger.warning("Full-text search failed; falling back to CONTAINS search: %s", exc)

        return self._fallback_keyword_search(terms, top_k=top_k, document_id=document_id)

    def _fallback_keyword_search(
        self,
        terms: list[str],
        *,
        top_k: int,
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        if not terms:
            return []
        rows = self.run(
            """
            MATCH (b:Block)-[:ON_PAGE]->(:Page)-[:PART_OF]->(d:Document)
            WHERE ($document_id IS NULL OR d.doc_id = $document_id)
              AND any(term IN $terms WHERE toLower(b.text) CONTAINS toLower(term))
            OPTIONAL MATCH (b)-[:IN_SECTION]->(section:Block {type:'heading'})
            WITH b, section,
                 reduce(score = 0.0, term IN $terms |
                     score + CASE WHEN toLower(b.text) CONTAINS toLower(term) THEN 1.0 ELSE 0.0 END
                 ) AS score
            RETURN b.block_id AS block_id,
                   b.type AS type,
                   b.page_number AS page,
                   b.reading_order AS reading_order,
                   b.text AS text,
                   section.block_id AS section_id,
                   section.text AS section,
                   score AS score
            ORDER BY score DESC, b.page_number, b.reading_order
            LIMIT $top_k
            """,
            {"terms": terms, "top_k": top_k, "document_id": document_id},
        )
        logger.info("Fallback keyword search returned %s blocks", len(rows))
        return rows

    def get_block(self, block_id: str) -> dict[str, Any] | None:
        rows = self.run(
            """
            MATCH (b:Block {block_id:$block_id})
            OPTIONAL MATCH (b)-[:IN_SECTION]->(section:Block {type:'heading'})
            RETURN b.block_id AS block_id,
                   b.type AS type,
                   b.page_number AS page,
                   b.reading_order AS reading_order,
                   b.text AS text,
                   section.block_id AS section_id,
                   section.text AS section
            """,
            {"block_id": block_id},
        )
        return rows[0] if rows else None

    def expand_block(
        self,
        block_id: str,
        *,
        block_type: str,
        semantic_similarity_threshold: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        if block_type == "table":
            return self._expand_table(block_id, limit=limit)
        if block_type == "heading":
            return self._expand_heading(block_id, limit=limit)
        return self._expand_text_block(
            block_id,
            semantic_similarity_threshold=semantic_similarity_threshold,
            limit=limit,
        )

    def _expand_text_block(
        self,
        block_id: str,
        *,
        semantic_similarity_threshold: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (seed:Block {block_id:$block_id})
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:REFERS_TO]->(target:Block)
                RETURN collect(DISTINCT {node:target, rel:type(r), score:null, methods:r.methods, mention:r.mention, reason:null}) AS refs
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:SEMANTICALLY_SIMILAR]->(similar:Block {type:'table'})
                WHERE r.score >= $threshold
                RETURN collect(DISTINCT {node:similar, rel:type(r), score:r.score, methods:null, mention:null, reason:null}) AS similar
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (prev:Block)-[r:PRECEDES]->(seed)
                WHERE prev.page_number = seed.page_number
                RETURN collect(DISTINCT {node:prev, rel:type(r), score:null, methods:null, mention:null, reason:null}) AS prevs
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:PRECEDES]->(next:Block)
                WHERE next.page_number = seed.page_number
                RETURN collect(DISTINCT {node:next, rel:type(r), score:null, methods:null, mention:null, reason:null}) AS nexts
            }
            WITH refs + similar + prevs + nexts AS candidates
            UNWIND candidates AS item
            WITH item.node AS n, item.rel AS relationship, item.score AS score,
                 item.methods AS methods, item.mention AS mention, item.reason AS reason
            WHERE n IS NOT NULL
            OPTIONAL MATCH (n)-[:IN_SECTION]->(section:Block {type:'heading'})
            RETURN n.block_id AS block_id,
                   n.type AS type,
                   n.page_number AS page,
                   n.reading_order AS reading_order,
                   n.text AS text,
                   section.block_id AS section_id,
                   section.text AS section,
                   score,
                   relationship,
                   methods,
                   mention,
                   reason
            ORDER BY page, reading_order
            LIMIT $limit
            """,
            {"block_id": block_id, "threshold": semantic_similarity_threshold, "limit": limit},
        )
        return rows

    def _expand_table(self, block_id: str, *, limit: int) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (seed:Block {block_id:$block_id, type:'table'})
            CALL {
                WITH seed
                OPTIONAL MATCH (before)-[r:CONTEXT_BEFORE]->(seed)
                RETURN collect(DISTINCT {node:before, rel:type(r), score:null, methods:null, mention:null, reason:null}) AS before_rows
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:CONTEXT_AFTER]->(after)
                RETURN collect(DISTINCT {node:after, rel:type(r), score:null, methods:null, mention:null, reason:null}) AS after_rows
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (referer)-[r:REFERS_TO]->(seed)
                RETURN collect(DISTINCT {node:referer, rel:type(r), score:null, methods:r.methods, mention:r.mention, reason:null}) AS referer_rows
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (caption)-[r:DESCRIBES]->(seed)
                RETURN collect(DISTINCT {node:caption, rel:type(r), score:null, methods:null, mention:null, reason:null}) AS caption_rows
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:COMPARES|SUPPLEMENTS|CONTRASTS|ABLATES]-(related:Block {type:'table'})
                RETURN collect(DISTINCT {node:related, rel:type(r), score:null, methods:null, mention:null, reason:r.reason}) AS table_rows
            }
            WITH before_rows + after_rows + referer_rows + caption_rows + table_rows AS candidates
            UNWIND candidates AS item
            WITH item.node AS n, item.rel AS relationship, item.score AS score,
                 item.methods AS methods, item.mention AS mention, item.reason AS reason
            WHERE n IS NOT NULL
            OPTIONAL MATCH (n)-[:IN_SECTION]->(section:Block {type:'heading'})
            RETURN n.block_id AS block_id,
                   n.type AS type,
                   n.page_number AS page,
                   n.reading_order AS reading_order,
                   n.text AS text,
                   section.block_id AS section_id,
                   section.text AS section,
                   score,
                   relationship,
                   methods,
                   mention,
                   reason
            ORDER BY page, reading_order
            LIMIT $limit
            """,
            {"block_id": block_id, "limit": limit},
        )
        return rows

    def _expand_heading(self, block_id: str, *, limit: int) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (seed:Block {block_id:$block_id, type:'heading'})
            OPTIONAL MATCH (seed)-[r:INTRODUCES]->(introduced:Block)
            WHERE introduced.type IN ['paragraph','table','caption','figure','formula','list_item']
            OPTIONAL MATCH (introduced)-[:IN_SECTION]->(section:Block {type:'heading'})
            RETURN introduced.block_id AS block_id,
                   introduced.type AS type,
                   introduced.page_number AS page,
                   introduced.reading_order AS reading_order,
                   introduced.text AS text,
                   section.block_id AS section_id,
                   section.text AS section,
                   null AS score,
                   type(r) AS relationship,
                   null AS methods,
                   null AS mention,
                   null AS reason
            ORDER BY page, reading_order
            LIMIT $limit
            """,
            {"block_id": block_id, "limit": limit},
        )
        return rows

    def get_table_relationships(self, table_id: str, relation: str | None = None) -> list[dict[str, Any]]:
        if relation is not None and relation not in TABLE_RELATION_TYPES:
            raise ValueError(f"Unsupported table relation {relation}")
        return self.run(
            """
            MATCH (source:Block {block_id:$table_id, type:'table'})-[r:COMPARES|SUPPLEMENTS|CONTRASTS|ABLATES]-(target:Block {type:'table'})
            WHERE $relation IS NULL OR type(r) = $relation
            OPTIONAL MATCH (source)-[:IN_SECTION]->(source_section:Block {type:'heading'})
            OPTIONAL MATCH (target)-[:IN_SECTION]->(target_section:Block {type:'heading'})
            RETURN source.block_id AS source_block_id,
                   source.page_number AS source_page,
                   source.text AS source_text,
                   source_section.text AS source_section,
                   target.block_id AS target_block_id,
                   target.page_number AS target_page,
                   target.text AS target_text,
                   target_section.text AS target_section,
                   type(r) AS relation,
                   r.reason AS reason
            ORDER BY target.page_number, target.reading_order
            """,
            {"table_id": table_id, "relation": relation},
        )

    def get_document_map_rows(self, document_id: str | None) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (h:Block {type:'heading'})<-[:IN_SECTION]-(b:Block)
            MATCH (b)-[:ON_PAGE]->(:Page)-[:PART_OF]->(d:Document)
            WHERE ($document_id IS NULL OR d.doc_id = $document_id)
              AND b.type IN ['paragraph','table','caption','figure','formula','list_item']
            OPTIONAL MATCH (b)-[r:COMPARES|SUPPLEMENTS|CONTRASTS|ABLATES|REFERS_TO|DESCRIBES|SEMANTICALLY_SIMILAR]-(related:Block)
            WITH h, b, collect(DISTINCT {from:b.block_id, rel:type(r), to:related.block_id}) AS relationships
            ORDER BY h.page_number, h.reading_order, b.page_number, b.reading_order
            RETURN h.block_id AS section_id,
                   h.text AS section,
                   h.page_number AS section_page,
                   collect(DISTINCT {
                       block_id:b.block_id,
                       type:b.type,
                       page:b.page_number,
                       text:left(b.text, 240)
                   }) AS blocks,
                   collect(DISTINCT relationships) AS relationship_groups
            ORDER BY section_page
            """,
            {"document_id": document_id},
        )


def rows_to_table_relationships(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        converted.append(
            {
                **row,
                "source_snippet": make_snippet(row.get("source_text")),
                "target_snippet": make_snippet(row.get("target_text")),
            }
        )
    return converted
