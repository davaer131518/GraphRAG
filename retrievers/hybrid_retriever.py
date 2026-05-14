from __future__ import annotations

import logging
from collections import defaultdict

from config import Settings
from evidence.evidence_bundle import EvidenceBundle, EvidenceItem, TraceStep
from retrievers.entity_retriever import EntityRetriever
from retrievers.graph_expander import GraphExpander
from retrievers.keyword_retriever import KeywordRetriever
from retrievers.semantic_retriever import SemanticRetriever

logger = logging.getLogger(__name__)


RELATION_WEIGHTS = {
    "REFERS_TO": 0.28,
    "DESCRIBES": 0.24,
    "CONTEXT_BEFORE": 0.18,
    "CONTEXT_AFTER": 0.18,
    "SUPPLEMENTS": 0.22,
    "COMPARES": 0.22,
    "CONTRASTS": 0.18,
    "ABLATES": 0.18,
    "SEMANTICALLY_SIMILAR": 0.12,
    "PRECEDES": 0.05,
    "INTRODUCES": 0.08,
    "MENTIONS_SHARED": 0.14,
    "SAME_SECTION": 0.10,
}

TYPE_WEIGHTS = {
    "paragraph": 0.12,
    "table": 0.12,
    "caption": 0.08,
    "list_item": 0.08,
    "heading": 0.04,
    "figure": 0.03,
    "formula": 0.03,
}


class HybridRetriever:
    def __init__(
        self,
        semantic: SemanticRetriever,
        keyword: KeywordRetriever,
        entity_retriever: EntityRetriever,
        graph_expander: GraphExpander,
        settings: Settings,
    ) -> None:
        self.semantic = semantic
        self.keyword = keyword
        self.entity_retriever = entity_retriever
        self.graph_expander = graph_expander
        self.settings = settings

    def retrieve(self, question: str) -> EvidenceBundle:
        semantic_items, semantic_trace = self.semantic.retrieve(question)
        keyword_items, keyword_trace = self.keyword.retrieve(question)

        entity_items: list[EvidenceItem] = []
        entity_trace: list[TraceStep] = []
        if self.settings.enable_entity_retriever:
            entity_items, entity_trace = self.entity_retriever.retrieve(question)

        seeds = self._merge_seeds([semantic_items, keyword_items, entity_items])
        expanded_items, expansion_trace = self.graph_expander.expand(seeds)
        final_evidence, ranking_debug = self._rank_and_dedupe(question, seeds, expanded_items)

        trace = [
            *semantic_trace,
            *keyword_trace,
            *entity_trace,
            TraceStep(
                action="merge_seeds",
                description=(
                    f"Merged {len(semantic_items)} vector, {len(keyword_items)} keyword, "
                    f"and {len(entity_items)} entity hits into {len(seeds)} seeds."
                ),
                method="hybrid",
            ),
            *expansion_trace,
            TraceStep(
                action="rank_evidence",
                description=f"Selected {len(final_evidence)} final evidence blocks.",
                method="hybrid",
                metadata={"ranking_debug": ranking_debug},
            ),
        ]
        return EvidenceBundle(
            question=question,
            document_id=self.settings.document_id,
            seed_blocks=seeds,
            expanded_blocks=expanded_items,
            final_evidence=final_evidence,
            trace=trace,
            ranking_debug=ranking_debug,
        )

    def _merge_seeds(self, seed_lists: list[list[EvidenceItem]]) -> list[EvidenceItem]:
        merged: dict[str, EvidenceItem] = {}
        for items in seed_lists:
            for item in items:
                existing = merged.get(item.block_id)
                if existing is None:
                    merged[item.block_id] = item
                    continue
                existing.retrieval_method = "+".join(
                    sorted(set(existing.retrieval_method.split("+") + [item.retrieval_method]))
                )
                existing.score = max(existing.score or 0.0, item.score or 0.0)
                existing.relationship_path.extend(
                    path for path in item.relationship_path if path not in existing.relationship_path
                )
                # Merge matched_entities (union by entity_id)
                existing_ids = {e.get("entity_id") for e in existing.matched_entities}
                for ent in item.matched_entities:
                    if ent.get("entity_id") not in existing_ids:
                        existing.matched_entities.append(ent)
                        existing_ids.add(ent.get("entity_id"))
                if "vector" in existing.retrieval_method and "keyword" in existing.retrieval_method:
                    existing.why_relevant = "Matched by both vector and keyword retrieval."
        return sorted(merged.values(), key=lambda i: (-(i.score or 0.0), i.page or 10**9))

    def _rank_and_dedupe(
        self,
        question: str,
        seeds: list[EvidenceItem],
        expanded: list[EvidenceItem],
    ) -> tuple[list[EvidenceItem], list[dict]]:
        lower_question = question.lower()
        question_tokens = {t for t in lower_question.split() if len(t) > 3}
        seed_section_ids = {item.section_id for item in seeds if item.section_id}

        grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in [*seeds, *expanded]:
            grouped[item.block_id].append(item)

        ranked: list[tuple[float, EvidenceItem, dict]] = []
        for block_id, items in grouped.items():
            best = items[0]

            # --- existing bonuses ---
            method_bonus = 0.0
            if any("vector" in item.retrieval_method for item in items):
                method_bonus += 0.32
            if any("keyword" in item.retrieval_method for item in items):
                method_bonus += 0.30
            if any(item.retrieval_method == "section_title" for item in items):
                method_bonus += 0.28
            if any(item.retrieval_method in {"graph_expansion", "entity_expansion", "section_expansion"}
                   for item in items):
                method_bonus += 0.12

            relation_bonus = max(
                (RELATION_WEIGHTS.get(item.source_relationship or "", 0.0) for item in items),
                default=0.0,
            )
            type_bonus = TYPE_WEIGHTS.get(best.type, 0.0)
            score_bonus = max((item.score or 0.0 for item in items), default=0.0) * 0.35
            exact_bonus = self._exact_match_bonus(lower_question, best.text)

            # --- new bonuses (each individually capped) ---

            # Entity match bonus — scaled by entity_match_score (exact=1.0, alias=0.9, partial=0.6)
            all_matched: list[dict] = []
            for item in items:
                all_matched.extend(item.matched_entities)
            entity_match_score = max(
                (e.get("entity_match_score") or 0.0 for e in all_matched), default=0.0
            )
            entity_match_bonus = (
                min(self.settings.entity_match_bonus * entity_match_score, self.settings.entity_match_bonus)
                if all_matched else 0.0
            )

            # Entity confidence bonus — × max MENTIONS confidence
            max_mention_conf = max(
                (e.get("mention_confidence") or 0.0 for e in all_matched), default=0.0
            )
            entity_confidence_bonus = min(
                max_mention_conf * self.settings.entity_confidence_bonus_weight,
                self.settings.entity_match_bonus,
            )

            # Same-section bonus
            same_section_bonus = (
                self.settings.same_section_bonus
                if best.section_id and best.section_id in seed_section_ids
                else 0.0
            )

            # Section path/title token match
            section_path_match_bonus = 0.0
            combined_section_text = " ".join(filter(None, [best.section_title, best.section_path])).lower()
            if combined_section_text and any(tok in combined_section_text for tok in question_tokens):
                section_path_match_bonus = self.settings.section_path_match_bonus

            # Structural bonus for section-expansion tables/figures/captions
            section_structural_bonus = 0.0
            if any(
                item.retrieval_method == "section_expansion"
                and item.metadata.get("structural_boost") == 1
                for item in items
            ):
                section_structural_bonus = self.settings.section_structural_bonus

            # Global similarity bonus
            global_similarity_bonus = min(
                max(
                    (
                        (item.score or 0.0)
                        for item in items
                        if item.relationship_scope == "global"
                    ),
                    default=0.0,
                ) * self.settings.global_similarity_bonus_weight,
                0.20,
            )

            # Relationship confidence bonus
            relationship_confidence_bonus = min(
                max(
                    (item.relationship_confidence or 0.0 for item in items),
                    default=0.0,
                ) * self.settings.relationship_confidence_bonus_weight,
                0.10,
            )

            total = (
                method_bonus
                + relation_bonus
                + type_bonus
                + score_bonus
                + exact_bonus
                + entity_match_bonus
                + entity_confidence_bonus
                + same_section_bonus
                + section_path_match_bonus
                + section_structural_bonus
                + global_similarity_bonus
                + relationship_confidence_bonus
            )

            best.rank_features = {
                "method_bonus": method_bonus,
                "relation_bonus": relation_bonus,
                "type_bonus": type_bonus,
                "score_bonus": score_bonus,
                "exact_bonus": exact_bonus,
                "entity_match_bonus": entity_match_bonus,
                "entity_confidence_bonus": entity_confidence_bonus,
                "same_section_bonus": same_section_bonus,
                "section_path_match_bonus": section_path_match_bonus,
                "section_structural_bonus": section_structural_bonus,
                "global_similarity_bonus": global_similarity_bonus,
                "relationship_confidence_bonus": relationship_confidence_bonus,
                "total": total,
            }

            # Merge relationship paths from duplicates
            if len(items) > 1:
                for item in items[1:]:
                    for path in item.relationship_path:
                        if path not in best.relationship_path:
                            best.relationship_path.append(path)
                    # Merge matched_entities
                    existing_ids = {e.get("entity_id") for e in best.matched_entities}
                    for ent in item.matched_entities:
                        if ent.get("entity_id") not in existing_ids:
                            best.matched_entities.append(ent)
                            existing_ids.add(ent.get("entity_id"))

            ranked.append((total, best, {"block_id": block_id, **best.rank_features}))

        ranked.sort(key=lambda entry: (-entry[0], entry[1].page or 10**9))
        final = [item for _, item, _ in ranked[: self.settings.final_evidence_limit]]

        # Guarantee short sibling blocks from section_title-matched sections are included.
        # A section whose title matches the question may hold a list split across several short
        # Blocks. One block can rank into the top-N via additional vector/keyword bonuses while
        # its siblings don't — include all short siblings so the LLM sees the complete list.
        section_title_section_ids = {
            item.section_id
            for item in seeds
            if item.retrieval_method == "section_title" and item.section_id
        }
        if section_title_section_ids:
            final_block_ids = {item.block_id for item in final}
            list_siblings = [
                item
                for _, item, _ in ranked[self.settings.final_evidence_limit:]
                if (
                    item.section_id in section_title_section_ids
                    and len(item.text or "") < 400
                    and item.block_id not in final_block_ids
                )
            ][: self.settings.section_expansion_limit]
            final = final + list_siblings

        debug = [debug for _, _, debug in ranked]
        logger.info("Final evidence blocks: %s", [item.block_id for item in final])
        return final, debug

    def _exact_match_bonus(self, lower_question: str, text: str) -> float:
        lower_text = (text or "").lower()
        question_terms = {term for term in lower_question.split() if len(term) > 4}
        matched = sum(1 for term in question_terms if term in lower_text)
        return min(matched * self.settings.keyword_term_boost, 0.25)
