from __future__ import annotations

import logging

from config import Settings
from evidence.evidence_bundle import EvidenceItem, TraceStep
from neo4j_client import Neo4jClient
from retrievers.scope import RetrievalScope

logger = logging.getLogger(__name__)


class GraphExpander:
    def __init__(self, neo4j: Neo4jClient, settings: Settings) -> None:
        self.neo4j = neo4j
        self.settings = settings

    def expand(
        self, seeds: list[EvidenceItem], scope: RetrievalScope | None = None
    ) -> tuple[list[EvidenceItem], list[TraceStep]]:
        scope = scope or RetrievalScope.whole_corpus()
        expanded: list[EvidenceItem] = []
        trace: list[TraceStep] = []
        for seed in seeds[: self.settings.graph_expansion_limit]:
            # Arm 1: relationship-typed expansion (single-doc, no scope filter needed — block_id-based)
            global_threshold = (
                self.settings.global_similarity_threshold
                if self.settings.enable_global_similarity_expansion
                else 2.0
            )
            rows = self.neo4j.expand_block(
                seed.block_id,
                block_type=seed.type,
                semantic_similarity_threshold=self.settings.semantic_similarity_threshold,
                limit=self.settings.graph_expansion_limit * 4,
                global_threshold=global_threshold,
            )
            logger.info("Expanded %s via %s rows", seed.block_id, len(rows))
            arm1_items: list[EvidenceItem] = []
            rel_counts: dict[str, int] = {}
            for row in rows:
                relationship = row.get("relationship") or "RELATED"
                rel_counts[relationship] = rel_counts.get(relationship, 0) + 1
                item = EvidenceItem.from_row(
                    row,
                    retrieval_method="graph_expansion",
                    relationship_path=[
                        *seed.relationship_path,
                        f"{seed.block_id} -{relationship}-> {row['block_id']}",
                    ],
                    source_relationship=relationship,
                    source_block_id=seed.block_id,
                    why_relevant=self._why_relevant(relationship, row),
                    metadata={
                        "methods": row.get("methods"),
                        "mention": row.get("mention"),
                        "reason": row.get("reason"),
                        "confidence": row.get("confidence"),
                        "scope": row.get("scope"),
                    },
                )
                arm1_items.append(item)
            expanded.extend(arm1_items)
            rel_summary = ", ".join(f"{r} ×{n}" for r, n in rel_counts.items()) if rel_counts else "none"
            trace.append(
                TraceStep(
                    action="graph_expansion",
                    description=(
                        f"Expanded {seed.block_id} through explicit graph relationships "
                        f"→ {len(arm1_items)} blocks ({rel_summary})."
                    ),
                    from_id=seed.block_id,
                    method="graph_expansion",
                    metadata={"rows": len(rows), "seed_type": seed.type, "rel_counts": rel_counts},
                )
            )

            # Arm 2: entity-mediated expansion (same-doc)
            if self.settings.enable_entity_expansion:
                entity_rows = self.neo4j.expand_block_via_entities(
                    seed.block_id,
                    entities_per_seed=self.settings.entity_expansion_entities_per_seed,
                    blocks_per_entity=self.settings.entity_expansion_blocks_per_entity,
                    term_doc_freq_filter=self.settings.term_doc_freq_filter,
                    document_ids=scope.doc_id_list,
                )
                logger.info("Entity expansion of %s: %s rows", seed.block_id, len(entity_rows))
                arm2_items: list[EvidenceItem] = []
                entity_names: list[str] = []
                seen_entity_names: set[str] = set()
                for row in entity_rows:
                    entity_name = row.get("entity_name") or ""
                    entity_type = row.get("entity_type") or ""
                    if entity_name and entity_name not in seen_entity_names:
                        entity_names.append(entity_name)
                        seen_entity_names.add(entity_name)
                    item = EvidenceItem.from_row(
                        row,
                        retrieval_method="entity_expansion",
                        relationship_path=[
                            *seed.relationship_path,
                            f"{seed.block_id} -[via Entity {entity_name!r}]-> {row['block_id']}",
                        ],
                        source_relationship="MENTIONS_SHARED",
                        source_block_id=seed.block_id,
                        why_relevant=self._why_relevant("MENTIONS_SHARED", row),
                        metadata={
                            "entity_id": row.get("entity_id"),
                            "entity_name": entity_name,
                            "entity_type": entity_type,
                            "mention_count": row.get("mention_count"),
                            "mention_confidence": row.get("mention_confidence"),
                        },
                    )
                    arm2_items.append(item)
                expanded.extend(arm2_items)
                entities_label = (
                    ", ".join(f"'{n}'" for n in entity_names[:3])
                    + (" …" if len(entity_names) > 3 else "")
                ) if entity_names else "none"
                trace.append(
                    TraceStep(
                        action="entity_expansion",
                        description=(
                            f"Expanded {seed.block_id} via shared entities ({entities_label}) "
                            f"→ {len(arm2_items)} blocks."
                        ),
                        from_id=seed.block_id,
                        method="entity_expansion",
                        metadata={"rows": len(entity_rows), "entities": entity_names},
                    )
                )

            # Arm 3: section-aware expansion (same-doc)
            if self.settings.enable_section_expansion:
                section_rows = self.neo4j.expand_block_via_section(
                    seed.block_id,
                    limit=self.settings.section_expansion_limit,
                    document_ids=scope.doc_id_list,
                )
                logger.info("Section expansion of %s: %s rows", seed.block_id, len(section_rows))
                arm3_items: list[EvidenceItem] = []
                section_label = ""
                for row in section_rows:
                    section_title = row.get("section_title") or ""
                    if not section_label and section_title:
                        section_label = section_title
                    structural_boost = int(row.get("structural_boost") or 0)
                    item = EvidenceItem.from_row(
                        row,
                        retrieval_method="section_expansion",
                        relationship_path=[
                            *seed.relationship_path,
                            f"{seed.block_id} -[SAME_SECTION]-> {row['block_id']}",
                        ],
                        source_relationship="SAME_SECTION",
                        source_block_id=seed.block_id,
                        why_relevant=self._why_relevant("SAME_SECTION", row),
                        metadata={
                            "section_distance": row.get("section_distance"),
                            "structural_boost": structural_boost,
                        },
                    )
                    arm3_items.append(item)
                expanded.extend(arm3_items)
                sec_detail = f" in '{section_label}'" if section_label else ""
                trace.append(
                    TraceStep(
                        action="section_expansion",
                        description=(
                            f"Expanded {seed.block_id} via sibling blocks{sec_detail} "
                            f"→ {len(arm3_items)} blocks."
                        ),
                        from_id=seed.block_id,
                        method="section_expansion",
                        metadata={"rows": len(section_rows), "section": section_label},
                    )
                )

            # ── Cross-doc arms (only fire when scope spans multiple docs) ──────

            # Arm 4: cross-doc entity expansion via CanonicalEntity bridge
            if self.settings.enable_cross_doc_entity_expansion and scope.is_multi_doc:
                cdce_rows = self.neo4j.expand_block_via_canonical_entities(
                    seed.block_id,
                    entities_per_seed=self.settings.cross_doc_entity_entities_per_seed,
                    blocks_per_entity=self.settings.cross_doc_entity_blocks_per_entity,
                    term_doc_freq_filter=self.settings.term_doc_freq_filter,
                    document_ids=scope.doc_id_list,
                    corpus_id=scope.corpus_id,
                )
                logger.info("Cross-doc entity expansion of %s: %s rows", seed.block_id, len(cdce_rows))
                arm4_items: list[EvidenceItem] = []
                canon_names: list[str] = []
                seen_canon: set[str] = set()
                for row in cdce_rows:
                    canon = row.get("canonical_name") or ""
                    if canon and canon not in seen_canon:
                        canon_names.append(canon)
                        seen_canon.add(canon)
                    seed_doc = seed.doc_label or seed.doc_id or "?"
                    tgt_doc = row.get("filename") or row.get("doc_id") or "?"
                    item = EvidenceItem.from_row(
                        row,
                        retrieval_method="cross_doc_entity_expansion",
                        relationship_path=[
                            *seed.relationship_path,
                            (
                                f"{seed.block_id} ({seed_doc})"
                                f" -[via CanonicalEntity {canon!r}]->"
                                f" {row['block_id']} ({tgt_doc})"
                            ),
                        ],
                        source_relationship="MENTIONS_SHARED_CANONICAL",
                        source_block_id=seed.block_id,
                        why_relevant=self._why_relevant("MENTIONS_SHARED_CANONICAL", row),
                        metadata={
                            "canonical_id": row.get("canonical_id"),
                            "canonical_name": canon,
                            "canonical_type": row.get("canonical_type"),
                            "source_doc_id": seed.doc_id,
                            "target_doc_id": row.get("doc_id"),
                            "mention_count": row.get("mention_count"),
                            "mention_confidence": row.get("mention_confidence"),
                        },
                    )
                    arm4_items.append(item)
                expanded.extend(arm4_items)
                canon_label = (
                    ", ".join(f"'{n}'" for n in canon_names[:3])
                    + (" …" if len(canon_names) > 3 else "")
                ) if canon_names else "none"
                trace.append(
                    TraceStep(
                        action="cross_doc_entity_expansion",
                        description=(
                            f"Bridged {seed.block_id} to other documents via shared canonical "
                            f"entities ({canon_label}) → {len(arm4_items)} blocks."
                        ),
                        from_id=seed.block_id,
                        method="cross_doc_entity_expansion",
                        metadata={"rows": len(cdce_rows), "canonical_entities": canon_names},
                    )
                )

            # Arm 5: cross-doc section expansion via SIMILAR_SECTION
            if self.settings.enable_cross_doc_section_expansion and scope.is_multi_doc:
                ss_rows = self.neo4j.expand_block_via_similar_sections(
                    seed.block_id,
                    similar_sections_per_seed=self.settings.cross_doc_similar_sections_per_seed,
                    blocks_per_section=self.settings.cross_doc_blocks_per_similar_section,
                    document_ids=scope.doc_id_list,
                )
                logger.info("Cross-doc section expansion of %s: %s rows", seed.block_id, len(ss_rows))
                arm5_items: list[EvidenceItem] = []
                for row in ss_rows:
                    src_sec = row.get("source_section_title") or (seed.section_title or "section")
                    tgt_sec = row.get("section_title") or "analogous section"
                    seed_doc = seed.doc_label or seed.doc_id or "?"
                    tgt_doc = row.get("filename") or row.get("doc_id") or "?"
                    item = EvidenceItem.from_row(
                        row,
                        retrieval_method="cross_doc_section_expansion",
                        relationship_path=[
                            *seed.relationship_path,
                            (
                                f"{seed.block_id} ({seed_doc})"
                                f" -[SIMILAR_SECTION {src_sec!r}≈{tgt_sec!r}]->"
                                f" {row['block_id']} ({tgt_doc})"
                            ),
                        ],
                        source_relationship="SIMILAR_SECTION",
                        source_block_id=seed.block_id,
                        why_relevant=self._why_relevant("SIMILAR_SECTION", row),
                        metadata={
                            "source_section_title": src_sec,
                            "target_section_title": tgt_sec,
                            "target_doc_id": row.get("doc_id"),
                            "similar_section_score": row.get("score"),
                        },
                    )
                    arm5_items.append(item)
                expanded.extend(arm5_items)
                trace.append(
                    TraceStep(
                        action="cross_doc_section_expansion",
                        description=(
                            f"Followed SIMILAR_SECTION from {seed.block_id} to analogous sections "
                            f"in other documents → {len(arm5_items)} blocks."
                        ),
                        from_id=seed.block_id,
                        method="cross_doc_section_expansion",
                        metadata={"rows": len(ss_rows)},
                    )
                )

            # Arm 6: cross-doc table expansion (table seeds only)
            if (
                self.settings.enable_cross_doc_table_expansion
                and scope.is_multi_doc
                and seed.type == "table"
            ):
                ct_rows = self.neo4j.get_cross_doc_table_matches(
                    seed.block_id,
                    document_ids=scope.doc_id_list,
                    limit=self.settings.cross_doc_table_limit,
                )
                logger.info("Cross-doc table expansion of %s: %s rows", seed.block_id, len(ct_rows))
                arm6_items: list[EvidenceItem] = []
                ct_rel_counts: dict[str, int] = {}
                for row in ct_rows:
                    rel = row.get("relationship") or "SCHEMA_MATCH"
                    ct_rel_counts[rel] = ct_rel_counts.get(rel, 0) + 1
                    seed_doc = seed.doc_label or seed.doc_id or "?"
                    tgt_doc = row.get("filename") or row.get("doc_id") or "?"
                    item = EvidenceItem.from_row(
                        row,
                        retrieval_method="cross_doc_table_expansion",
                        relationship_path=[
                            *seed.relationship_path,
                            f"{seed.block_id} ({seed_doc}) -[{rel}]-> {row['block_id']} ({tgt_doc})",
                        ],
                        source_relationship=rel,
                        source_block_id=seed.block_id,
                        why_relevant=self._why_relevant(rel, row),
                        metadata={
                            "schema_score": row.get("schema_score"),
                            "metric_score": row.get("metric_score"),
                            "target_doc_id": row.get("doc_id"),
                        },
                    )
                    arm6_items.append(item)
                expanded.extend(arm6_items)
                trace.append(
                    TraceStep(
                        action="cross_doc_table_expansion",
                        description=(
                            f"Matched table {seed.block_id} to tables in other documents → "
                            f"{len(arm6_items)} tables ({ct_rel_counts or 'none'})."
                        ),
                        from_id=seed.block_id,
                        method="cross_doc_table_expansion",
                        metadata={"rows": len(ct_rows), "rel_counts": ct_rel_counts},
                    )
                )

        return expanded, trace

    @staticmethod
    def _why_relevant(relationship: str, row: dict) -> str:
        if relationship == "REFERS_TO":
            methods = row.get("methods") or []
            detail = f" using {', '.join(methods)}" if methods else ""
            return f"Discusses or explicitly references the seed block{detail}."
        if relationship in {"CONTEXT_BEFORE", "CONTEXT_AFTER", "PRECEDES"}:
            return "Provides local reading-order context for the seed block."
        if relationship == "DESCRIBES":
            return "Caption or descriptive block attached to the seed block."
        if relationship == "SEMANTICALLY_SIMILAR":
            scope = row.get("scope") or ""
            scope_label = f" ({scope} scope)" if scope else ""
            return f"Connected by stored semantic similarity in the KG{scope_label}."
        if relationship in {"COMPARES", "SUPPLEMENTS", "CONTRASTS", "ABLATES"}:
            reason = row.get("reason")
            return reason or f"Table relationship labelled as {relationship}."
        if relationship == "INTRODUCES":
            return "Introduced by the matched heading."
        if relationship == "MENTIONS_SHARED":
            entity_name = row.get("entity_name") or ""
            entity_type = row.get("entity_type") or ""
            detail = f" '{entity_name}' ({entity_type})" if entity_name else ""
            return f"Shares entity{detail} with the seed block."
        if relationship == "SAME_SECTION":
            section_title = row.get("section_title") or ""
            detail = f" '{section_title}'" if section_title else ""
            return f"Same section{detail} as the seed block."
        if relationship == "MENTIONS_SHARED_CANONICAL":
            canon = row.get("canonical_name") or ""
            ctype = row.get("canonical_type") or ""
            tgt = row.get("filename") or row.get("doc_id") or "another document"
            detail = f" '{canon}' ({ctype})" if canon else ""
            return f"Discusses the same corpus entity{detail} as the seed, in {tgt}."
        if relationship == "SIMILAR_SECTION":
            tgt_sec = row.get("section_title") or ""
            tgt = row.get("filename") or row.get("doc_id") or "another document"
            detail = f" '{tgt_sec}'" if tgt_sec else ""
            return f"From the analogous section{detail} in {tgt} (matched by stored SIMILAR_SECTION)."
        if relationship in {"SCHEMA_MATCH", "REPORTS_SAME_METRIC"}:
            tgt = row.get("filename") or row.get("doc_id") or "another document"
            if relationship == "REPORTS_SAME_METRIC":
                return f"Reports the same metric as the seed table, in {tgt} (cross-document)."
            return f"Has a matching table schema to the seed table, in {tgt} (cross-document)."
        return "Connected to the seed block by the KG."
