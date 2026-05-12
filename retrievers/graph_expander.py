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
            rows = self.neo4j.expand_block(
                seed.block_id,
                block_type=seed.type,
                semantic_similarity_threshold=self.settings.semantic_similarity_threshold,
                limit=self.settings.graph_expansion_limit * 4,
            )
            logger.info("Expanded %s via %s rows", seed.block_id, len(rows))
            trace.append(
                TraceStep(
                    action="graph_expansion",
                    description=f"Expanded {seed.block_id} through explicit graph relationships.",
                    from_id=seed.block_id,
                    method="graph_expansion",
                    metadata={"rows": len(rows), "seed_type": seed.type},
                )
            )
            for row in rows:
                relationship = row.get("relationship") or "RELATED"
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
                    },
                )
                expanded.append(item)
                trace.append(
                    TraceStep(
                        action="relationship_expand",
                        description=f"Added {item.block_id} from {seed.block_id}.",
                        from_id=seed.block_id,
                        to_id=item.block_id,
                        relationship=relationship,
                        method="graph_expansion",
                        score=item.score,
                        page=item.page,
                        section=item.section,
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
            return "Connected by stored semantic similarity in the KG."
        if relationship in {"COMPARES", "SUPPLEMENTS", "CONTRASTS", "ABLATES"}:
            reason = row.get("reason")
            return reason or f"Table relationship labelled as {relationship}."
        if relationship == "INTRODUCES":
            return "Introduced by the matched heading."
        return "Connected to the seed block by the KG."
