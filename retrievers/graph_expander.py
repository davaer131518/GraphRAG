from __future__ import annotations

import logging

from config import Settings
from evidence.evidence_bundle import EvidenceItem, TraceStep
from neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphExpander:
    def __init__(self, neo4j: Neo4jClient, settings: Settings) -> None:
        self.neo4j = neo4j
        self.settings = settings

    def expand(self, seeds: list[EvidenceItem]) -> tuple[list[EvidenceItem], list[TraceStep]]:
        expanded: list[EvidenceItem] = []
        trace: list[TraceStep] = []
        for seed in seeds[: self.settings.graph_expansion_limit]:
            # Arm 1: relationship-typed expansion
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

            # Arm 2: entity-mediated expansion
            if self.settings.enable_entity_expansion:
                entity_rows = self.neo4j.expand_block_via_entities(
                    seed.block_id,
                    entities_per_seed=self.settings.entity_expansion_entities_per_seed,
                    blocks_per_entity=self.settings.entity_expansion_blocks_per_entity,
                    term_doc_freq_filter=self.settings.term_doc_freq_filter,
                    document_id=self.settings.document_id,
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

            # Arm 3: section-aware expansion
            if self.settings.enable_section_expansion:
                section_rows = self.neo4j.expand_block_via_section(
                    seed.block_id,
                    limit=self.settings.section_expansion_limit,
                    document_id=self.settings.document_id,
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
        return "Connected to the seed block by the KG."
