from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from config import Settings
from evidence.evidence_bundle import make_snippet

logger = logging.getLogger(__name__)


TABLE_RELATION_TYPES = {"COMPARES", "SUPPLEMENTS", "CONTRASTS", "ABLATES"}

# Cypher fragment reused in every query that fetches section info.
# Targets the new :Section node; projects the same column names consumed by EvidenceItem.from_row.
_SECTION_MATCH = """
    OPTIONAL MATCH ({alias})-[:IN_SECTION]->(s:Section)
"""

_SECTION_RETURN = (
    "s.section_id AS section_id, s.title AS section_title, "
    "s.path AS section_path, s.level AS section_level"
)


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
            OPTIONAL MATCH (node)-[:IN_SECTION]->(s:Section)
            RETURN node.block_id AS block_id,
                   node.type AS type,
                   node.page_number AS page,
                   node.reading_order AS reading_order,
                   node.text AS text,
                   s.section_id AS section_id,
                   s.title AS section_title,
                   s.path AS section_path,
                   s.level AS section_level,
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
                    OPTIONAL MATCH (node)-[:IN_SECTION]->(s:Section)
                    RETURN node.block_id AS block_id,
                           node.type AS type,
                           node.page_number AS page,
                           node.reading_order AS reading_order,
                           node.text AS text,
                           s.section_id AS section_id,
                           s.title AS section_title,
                           s.path AS section_path,
                           s.level AS section_level,
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
            OPTIONAL MATCH (b)-[:IN_SECTION]->(s:Section)
            WITH b, s,
                 reduce(score = 0.0, term IN $terms |
                     score + CASE WHEN toLower(b.text) CONTAINS toLower(term) THEN 1.0 ELSE 0.0 END
                 ) AS score
            RETURN b.block_id AS block_id,
                   b.type AS type,
                   b.page_number AS page,
                   b.reading_order AS reading_order,
                   b.text AS text,
                   s.section_id AS section_id,
                   s.title AS section_title,
                   s.path AS section_path,
                   s.level AS section_level,
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
            OPTIONAL MATCH (b)-[:IN_SECTION]->(s:Section)
            RETURN b.block_id AS block_id,
                   b.type AS type,
                   b.page_number AS page,
                   b.reading_order AS reading_order,
                   b.text AS text,
                   s.section_id AS section_id,
                   s.title AS section_title,
                   s.path AS section_path,
                   s.level AS section_level
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
        global_threshold: float = 2.0,
    ) -> list[dict[str, Any]]:
        if block_type == "table":
            return self._expand_table(block_id, limit=limit)
        if block_type == "heading":
            return self._expand_heading(block_id, limit=limit)
        return self._expand_text_block(
            block_id,
            table_threshold=semantic_similarity_threshold,
            global_threshold=global_threshold,
            limit=limit,
        )

    def _expand_text_block(
        self,
        block_id: str,
        *,
        table_threshold: float,
        global_threshold: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (seed:Block {block_id:$block_id})
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:REFERS_TO]->(target:Block)
                RETURN collect(DISTINCT {
                    node:target, rel:type(r), score:null,
                    methods:r.methods, mention:r.mention, reason:null,
                    confidence:r.confidence, scope:r.scope
                }) AS refs
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:SEMANTICALLY_SIMILAR]-(similar:Block {type:'table'})
                WHERE r.scope = 'table' AND r.score >= $table_threshold
                RETURN collect(DISTINCT {
                    node:similar, rel:type(r), score:r.score,
                    methods:r.methods, mention:null, reason:null,
                    confidence:r.confidence, scope:r.scope
                }) AS table_similar
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:SEMANTICALLY_SIMILAR]-(global_block:Block)
                WHERE r.scope = 'global' AND r.score >= $global_threshold
                RETURN collect(DISTINCT {
                    node:global_block, rel:type(r), score:r.score,
                    methods:r.methods, mention:null, reason:null,
                    confidence:r.confidence, scope:r.scope
                }) AS global_similar
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (prev:Block)-[r:PRECEDES]->(seed)
                WHERE prev.page_number = seed.page_number
                RETURN collect(DISTINCT {
                    node:prev, rel:type(r), score:null,
                    methods:null, mention:null, reason:null,
                    confidence:null, scope:null
                }) AS prevs
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:PRECEDES]->(next:Block)
                WHERE next.page_number = seed.page_number
                RETURN collect(DISTINCT {
                    node:next, rel:type(r), score:null,
                    methods:null, mention:null, reason:null,
                    confidence:null, scope:null
                }) AS nexts
            }
            WITH refs + table_similar + global_similar + prevs + nexts AS candidates
            UNWIND candidates AS item
            WITH item.node AS n, item.rel AS relationship, item.score AS score,
                 item.methods AS methods, item.mention AS mention, item.reason AS reason,
                 item.confidence AS confidence, item.scope AS scope
            WHERE n IS NOT NULL
            OPTIONAL MATCH (n)-[:IN_SECTION]->(s:Section)
            RETURN n.block_id AS block_id,
                   n.type AS type,
                   n.page_number AS page,
                   n.reading_order AS reading_order,
                   n.text AS text,
                   s.section_id AS section_id,
                   s.title AS section_title,
                   s.path AS section_path,
                   s.level AS section_level,
                   score,
                   relationship,
                   methods,
                   mention,
                   reason,
                   confidence,
                   scope
            ORDER BY page, reading_order
            LIMIT $limit
            """,
            {
                "block_id": block_id,
                "table_threshold": table_threshold,
                "global_threshold": global_threshold,
                "limit": limit,
            },
        )
        return rows

    def _expand_table(self, block_id: str, *, limit: int) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (seed:Block {block_id:$block_id, type:'table'})
            CALL {
                WITH seed
                OPTIONAL MATCH (before)-[r:CONTEXT_BEFORE]->(seed)
                RETURN collect(DISTINCT {
                    node:before, rel:type(r), score:null,
                    methods:null, mention:null, reason:null,
                    confidence:null, scope:null
                }) AS before_rows
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:CONTEXT_AFTER]->(after)
                RETURN collect(DISTINCT {
                    node:after, rel:type(r), score:null,
                    methods:null, mention:null, reason:null,
                    confidence:null, scope:null
                }) AS after_rows
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (referer)-[r:REFERS_TO]->(seed)
                RETURN collect(DISTINCT {
                    node:referer, rel:type(r), score:null,
                    methods:r.methods, mention:r.mention, reason:null,
                    confidence:r.confidence, scope:r.scope
                }) AS referer_rows
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (caption)-[r:DESCRIBES]->(seed)
                RETURN collect(DISTINCT {
                    node:caption, rel:type(r), score:null,
                    methods:null, mention:null, reason:null,
                    confidence:null, scope:null
                }) AS caption_rows
            }
            CALL {
                WITH seed
                OPTIONAL MATCH (seed)-[r:COMPARES|SUPPLEMENTS|CONTRASTS|ABLATES|TABLE_RELATES_TO]-(related:Block {type:'table'})
                RETURN collect(DISTINCT {
                    node:related,
                    rel:coalesce(r.label, type(r)),
                    score:null,
                    methods:r.methods, mention:null, reason:r.reason,
                    confidence:r.confidence, scope:r.scope
                }) AS table_rows
            }
            WITH before_rows + after_rows + referer_rows + caption_rows + table_rows AS candidates
            UNWIND candidates AS item
            WITH item.node AS n, item.rel AS relationship, item.score AS score,
                 item.methods AS methods, item.mention AS mention, item.reason AS reason,
                 item.confidence AS confidence, item.scope AS scope
            WHERE n IS NOT NULL
            OPTIONAL MATCH (n)-[:IN_SECTION]->(s:Section)
            RETURN n.block_id AS block_id,
                   n.type AS type,
                   n.page_number AS page,
                   n.reading_order AS reading_order,
                   n.text AS text,
                   s.section_id AS section_id,
                   s.title AS section_title,
                   s.path AS section_path,
                   s.level AS section_level,
                   score,
                   relationship,
                   methods,
                   mention,
                   reason,
                   confidence,
                   scope
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
            OPTIONAL MATCH (introduced)-[:IN_SECTION]->(s:Section)
            RETURN introduced.block_id AS block_id,
                   introduced.type AS type,
                   introduced.page_number AS page,
                   introduced.reading_order AS reading_order,
                   introduced.text AS text,
                   s.section_id AS section_id,
                   s.title AS section_title,
                   s.path AS section_path,
                   s.level AS section_level,
                   null AS score,
                   type(r) AS relationship,
                   null AS methods,
                   null AS mention,
                   null AS reason,
                   null AS confidence,
                   null AS scope
            ORDER BY page, reading_order
            LIMIT $limit
            """,
            {"block_id": block_id, "limit": limit},
        )
        return rows

    def expand_block_via_entities(
        self,
        block_id: str,
        *,
        entities_per_seed: int,
        blocks_per_entity: int,
        term_doc_freq_filter: float,
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (seed:Block {block_id:$block_id})-[m_seed:MENTIONS]->(e:Entity)
            WHERE NOT (e.type = 'TERM' AND coalesce(e.doc_frequency_ratio, 0.0) > $term_doc_freq_filter)
            WITH seed, e, m_seed
            ORDER BY m_seed.confidence DESC, m_seed.count DESC
            LIMIT $entities_per_seed
            MATCH (e)<-[m:MENTIONS]-(related:Block)-[:ON_PAGE]->(:Page)-[:PART_OF]->(d:Document)
            WHERE related.block_id <> seed.block_id
              AND ($document_id IS NULL OR d.doc_id = $document_id)
            OPTIONAL MATCH (related)-[:IN_SECTION]->(s:Section)
            WITH e, related, s, m
            ORDER BY m.confidence DESC, m.count DESC
            WITH e, collect(DISTINCT {block:related, section:s, mention:m})[..$blocks_per_entity] AS top_related
            UNWIND top_related AS hit
            RETURN hit.block.block_id AS block_id,
                   hit.block.type AS type,
                   hit.block.page_number AS page,
                   hit.block.reading_order AS reading_order,
                   hit.block.text AS text,
                   hit.section.section_id AS section_id,
                   hit.section.title AS section_title,
                   hit.section.path AS section_path,
                   hit.section.level AS section_level,
                   e.entity_id AS entity_id,
                   e.canonical_name AS entity_name,
                   e.type AS entity_type,
                   hit.mention.count AS mention_count,
                   hit.mention.confidence AS mention_confidence
            """,
            {
                "block_id": block_id,
                "entities_per_seed": entities_per_seed,
                "blocks_per_entity": blocks_per_entity,
                "term_doc_freq_filter": term_doc_freq_filter,
                "document_id": document_id,
            },
        )
        return rows

    def expand_block_via_section(
        self,
        block_id: str,
        *,
        limit: int,
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (seed:Block {block_id:$block_id})-[:IN_SECTION]->(s:Section)
            WHERE $document_id IS NULL OR s.doc_id = $document_id
            MATCH (s)<-[:IN_SECTION]-(sibling:Block)
            WHERE sibling.block_id <> seed.block_id
            WITH seed, s, sibling,
                 abs(coalesce(sibling.reading_order, 0) - coalesce(seed.reading_order, 0)) AS section_distance,
                 CASE WHEN sibling.type IN ['table','figure','caption'] THEN 1 ELSE 0 END AS structural_boost
            ORDER BY section_distance ASC, sibling.reading_order ASC
            LIMIT $limit
            RETURN sibling.block_id AS block_id,
                   sibling.type AS type,
                   sibling.page_number AS page,
                   sibling.reading_order AS reading_order,
                   sibling.text AS text,
                   s.section_id AS section_id,
                   s.title AS section_title,
                   s.path AS section_path,
                   s.level AS section_level,
                   section_distance,
                   structural_boost
            """,
            {"block_id": block_id, "limit": limit, "document_id": document_id},
        )
        return rows

    def section_title_search_blocks(
        self,
        terms: list[str],
        *,
        top_k: int,
        document_id: str | None,
        term_min_len: int = 4,
    ) -> list[dict[str, Any]]:
        """Return blocks whose containing :Section title or path matches any query term.

        This catches questions like "What was announced in the third quarter of 2023?"
        where the answer block contains only product names with no query-term overlap,
        but lives inside a :Section whose title IS the temporal anchor.
        """
        qualifying = [t for t in terms if len(t) >= term_min_len]
        if not qualifying:
            return []
        rows = self.run(
            """
            MATCH (s:Section)
            WHERE ($document_id IS NULL OR s.doc_id = $document_id)
              AND any(term IN $terms
                      WHERE toLower(s.title) CONTAINS toLower(term)
                         OR toLower(coalesce(s.path, '')) CONTAINS toLower(term))
            MATCH (b:Block)-[:IN_SECTION]->(s)
            WHERE b.type IN ['paragraph','table','caption','figure','formula','list_item']
            MATCH (b)-[:ON_PAGE]->(:Page)-[:PART_OF]->(d:Document)
            WHERE $document_id IS NULL OR d.doc_id = $document_id
            WITH b, s,
                 reduce(score = 0.0, term IN $terms |
                     score + CASE
                         WHEN toLower(s.title) CONTAINS toLower(term)
                           OR toLower(coalesce(s.path, '')) CONTAINS toLower(term)
                         THEN 1.0 ELSE 0.0 END
                 ) AS score
            RETURN b.block_id AS block_id,
                   b.type AS type,
                   b.page_number AS page,
                   b.reading_order AS reading_order,
                   b.text AS text,
                   s.section_id AS section_id,
                   s.title AS section_title,
                   s.path AS section_path,
                   s.level AS section_level,
                   score AS score
            ORDER BY score DESC, b.reading_order ASC
            LIMIT $top_k
            """,
            {"terms": qualifying, "top_k": top_k, "document_id": document_id},
        )
        logger.info("Section-title search returned %s blocks", len(rows))
        return rows

    def entity_search_blocks(
        self,
        terms_lower: list[str],
        *,
        top_k: int,
        term_doc_freq_filter: float,
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        if not terms_lower:
            return []
        rows = self.run(
            """
            MATCH (e:Entity)
            WHERE ($document_id IS NULL OR e.doc_id = $document_id)
              AND NOT (e.type = 'TERM' AND coalesce(e.doc_frequency_ratio, 0.0) > $term_doc_freq_filter)
            WITH e,
                 (toLower(e.canonical_name) IN $terms_lower
                  OR toLower(e.normalized_name) IN $terms_lower) AS is_exact,
                 any(alias IN coalesce(e.aliases, [])
                     WHERE size(alias) >= 3 AND toLower(alias) IN $terms_lower) AS is_alias_exact,
                 any(t IN $terms_lower
                     WHERE size(t) >= 4 AND toLower(e.normalized_name) CONTAINS t) AS is_partial
            WHERE is_exact OR is_alias_exact OR is_partial
            WITH e,
                 CASE WHEN is_exact THEN 'exact'
                      WHEN is_alias_exact THEN 'alias'
                      ELSE 'partial' END AS entity_match_type,
                 CASE WHEN is_exact THEN 1.0
                      WHEN is_alias_exact THEN 0.9
                      ELSE 0.6 END AS entity_match_score
            MATCH (b:Block)-[m:MENTIONS]->(e)
            MATCH (b)-[:ON_PAGE]->(:Page)-[:PART_OF]->(d:Document)
            WHERE $document_id IS NULL OR d.doc_id = $document_id
            OPTIONAL MATCH (b)-[:IN_SECTION]->(s:Section)
            RETURN b.block_id AS block_id,
                   b.type AS type,
                   b.page_number AS page,
                   b.reading_order AS reading_order,
                   b.text AS text,
                   s.section_id AS section_id,
                   s.title AS section_title,
                   s.path AS section_path,
                   s.level AS section_level,
                   e.entity_id AS entity_id,
                   e.canonical_name AS entity_name,
                   e.type AS entity_type,
                   e.confidence AS entity_confidence,
                   entity_match_type,
                   entity_match_score,
                   m.count AS mention_count,
                   m.confidence AS mention_confidence,
                   m.methods AS mention_methods
            ORDER BY entity_match_score DESC, mention_confidence DESC, mention_count DESC
            LIMIT $top_k
            """,
            {
                "terms_lower": terms_lower,
                "top_k": top_k,
                "term_doc_freq_filter": term_doc_freq_filter,
                "document_id": document_id,
            },
        )
        logger.info("Entity search returned %s blocks", len(rows))
        return rows

    def get_table_relationships(self, table_id: str, relation: str | None = None) -> list[dict[str, Any]]:
        if relation is not None and relation not in TABLE_RELATION_TYPES:
            raise ValueError(f"Relation must be one of: {', '.join(sorted(TABLE_RELATION_TYPES))}")
        return self.run(
            """
            MATCH (source:Block {block_id:$table_id, type:'table'})
                  -[r:COMPARES|SUPPLEMENTS|CONTRASTS|ABLATES|TABLE_RELATES_TO]-
                  (target:Block {type:'table'})
            WITH source, target, r, coalesce(r.label, type(r)) AS rel_label
            WHERE $relation IS NULL OR rel_label = $relation
            OPTIONAL MATCH (source)-[:IN_SECTION]->(source_section:Section)
            OPTIONAL MATCH (target)-[:IN_SECTION]->(target_section:Section)
            RETURN source.block_id AS source_block_id,
                   source.page_number AS source_page,
                   source.text AS source_text,
                   source_section.title AS source_section,
                   target.block_id AS target_block_id,
                   target.page_number AS target_page,
                   target.text AS target_text,
                   target_section.title AS target_section,
                   rel_label AS relation,
                   r.reason AS reason
            ORDER BY target.page_number, target.reading_order
            """,
            {"table_id": table_id, "relation": relation},
        )

    def get_document_map_hierarchical(
        self,
        document_id: str | None,
        *,
        term_doc_freq_filter: float = 0.25,
    ) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (d:Document)
            WHERE $document_id IS NULL OR d.doc_id = $document_id
            MATCH (d)-[:HAS_SECTION]->(top:Section)
            OPTIONAL MATCH path_to = (top)-[:HAS_SUBSECTION*0..]->(s:Section)
            WITH s
            WHERE s IS NOT NULL
            OPTIONAL MATCH (s)<-[:IN_SECTION]-(b:Block)
            WHERE b.type IN ['paragraph','table','caption','figure','formula','list_item']
            OPTIONAL MATCH (b)-[m:MENTIONS]->(e:Entity)
            WHERE NOT (e.type = 'TERM' AND coalesce(e.doc_frequency_ratio, 0.0) > $term_doc_freq_filter)
            WITH s, b,
                 collect(DISTINCT {
                     entity_id: e.entity_id,
                     name: e.canonical_name,
                     type: e.type,
                     count: m.count
                 }) AS block_entities
            WITH s,
                 collect(DISTINCT {
                     block_id: b.block_id,
                     type: b.type,
                     page: b.page_number,
                     text: left(coalesce(b.text, ''), 240),
                     entities: block_entities
                 }) AS blocks
            OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(child:Section)
            RETURN s.section_id AS section_id,
                   s.title AS title,
                   s.path AS path,
                   s.level AS level,
                   s.page_start AS page_start,
                   s.page_end AS page_end,
                   s.block_count AS block_count,
                   blocks,
                   collect(DISTINCT child.section_id) AS child_section_ids
            ORDER BY s.page_start, s.level
            """,
            {"document_id": document_id, "term_doc_freq_filter": term_doc_freq_filter},
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
