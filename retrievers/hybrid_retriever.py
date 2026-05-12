from __future__ import annotations

import logging
from collections import defaultdict

from config import Settings
from evidence.evidence_bundle import EvidenceBundle, EvidenceItem, TraceStep
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
        graph_expander: GraphExpander,
        settings: Settings,
    ) -> None:
        self.semantic = semantic
        self.keyword = keyword
        self.graph_expander = graph_expander
        self.settings = settings

    def retrieve(self, question: str) -> EvidenceBundle:
        semantic_items, semantic_trace = self.semantic.retrieve(question)
        keyword_items, keyword_trace = self.keyword.retrieve(question)
        seeds = self._merge_seeds(semantic_items, keyword_items)
        expanded_items, expansion_trace = self.graph_expander.expand(seeds)
        final_evidence, ranking_debug = self._rank_and_dedupe(question, seeds, expanded_items)
        trace = [
            *semantic_trace,
            *keyword_trace,
            TraceStep(
                action="merge_seeds",
                description=f"Merged {len(semantic_items)} vector and {len(keyword_items)} keyword hits into {len(seeds)} seeds.",
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

    def _merge_seeds(self, semantic_items: list[EvidenceItem], keyword_items: list[EvidenceItem]) -> list[EvidenceItem]:
        merged: dict[str, EvidenceItem] = {}
        for item in [*semantic_items, *keyword_items]:
            existing = merged.get(item.block_id)
            if existing is None:
                merged[item.block_id] = item
                continue
            existing.retrieval_method = "+".join(sorted(set(existing.retrieval_method.split("+") + [item.retrieval_method])))
            existing.score = max(existing.score or 0.0, item.score or 0.0)
            existing.relationship_path.extend(path for path in item.relationship_path if path not in existing.relationship_path)
            existing.why_relevant = "Matched by both vector and keyword retrieval."
        return sorted(merged.values(), key=lambda item: (-(item.score or 0.0), item.page or 10**9))

    def _rank_and_dedupe(
        self,
        question: str,
        seeds: list[EvidenceItem],
        expanded: list[EvidenceItem],
    ) -> tuple[list[EvidenceItem], list[dict]]:
        lower_question = question.lower()
        grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in [*seeds, *expanded]:
            grouped[item.block_id].append(item)

        ranked: list[tuple[float, EvidenceItem, dict]] = []
        for block_id, items in grouped.items():
            best = items[0]
            method_bonus = 0.0
            if any("vector" in item.retrieval_method for item in items):
                method_bonus += 0.32
            if any("keyword" in item.retrieval_method for item in items):
                method_bonus += 0.30
            if any(item.retrieval_method == "graph_expansion" for item in items):
                method_bonus += 0.12
            relation_bonus = max(
                (RELATION_WEIGHTS.get(item.source_relationship or "", 0.0) for item in items),
                default=0.0,
            )
            type_bonus = TYPE_WEIGHTS.get(best.type, 0.0)
            score_bonus = max((item.score or 0.0 for item in items), default=0.0) * 0.35
            exact_bonus = self._exact_match_bonus(lower_question, best.text)
            total = method_bonus + relation_bonus + type_bonus + score_bonus + exact_bonus
            best.rank_features = {
                "method_bonus": method_bonus,
                "relation_bonus": relation_bonus,
                "type_bonus": type_bonus,
                "score_bonus": score_bonus,
                "exact_bonus": exact_bonus,
                "total": total,
            }
            if len(items) > 1:
                for item in items[1:]:
                    for path in item.relationship_path:
                        if path not in best.relationship_path:
                            best.relationship_path.append(path)
            ranked.append((total, best, {"block_id": block_id, **best.rank_features}))

        ranked.sort(key=lambda entry: (-entry[0], entry[1].page or 10**9))
        final = [item for _, item, _ in ranked[: self.settings.final_evidence_limit]]
        debug = [debug for _, _, debug in ranked]
        logger.info("Final evidence blocks: %s", [item.block_id for item in final])
        return final, debug

    def _exact_match_bonus(self, lower_question: str, text: str) -> float:
        lower_text = (text or "").lower()
        bonus = 0.0
        for phrase in ("app store", "digital markets act", "risk factors", "litigation"):
            if phrase in lower_question and phrase in lower_text:
                bonus += self.settings.keyword_exact_boost
        question_terms = {term for term in lower_question.split() if len(term) > 4}
        matched = sum(1 for term in question_terms if term in lower_text)
        return bonus + min(matched * self.settings.keyword_term_boost, 0.25)
