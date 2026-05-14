from __future__ import annotations

from config import Settings
from evidence.evidence_bundle import EvidenceItem
from retrievers.entity_retriever import EntityRetriever
from retrievers.graph_expander import GraphExpander
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.keyword_retriever import KeywordRetriever
from tests.fakes import FakeNeo4j, _entity_block_row, _section_expand_row


def _settings(**overrides) -> Settings:
    base = dict(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        neo4j_database=None,
        embed_server_url="http://localhost:8091",
        llm_server_url="http://localhost:8092",
        document_id=None,
        vector_top_k=3,
        keyword_top_k=3,
        graph_expansion_limit=2,
        final_evidence_limit=3,
        semantic_similarity_threshold=0.5,
        llm_max_tokens=100,
        llm_temperature=0.0,
        embed_max_chars=6000,
        request_timeout_seconds=10,
        log_level="INFO",
        keyword_term_boost=0.05,
        create_fulltext_index=False,
        llama_server_exe=None,
        embed_model_path="",
        embed_server_port=8091,
        embed_n_ctx=8192,
        llm_model_path="",
        llm_server_port=8092,
        llm_n_ctx=4096,
        llama_health_timeout=120,
        auto_start_servers=False,
        # Entity retrieval bounds
        entity_top_k=4,
        entity_expansion_entities_per_seed=2,
        entity_expansion_blocks_per_entity=3,
        section_expansion_limit=4,
        global_similarity_threshold=0.65,
        term_doc_freq_filter=0.25,
        mentioned_entities_per_block=5,
        # Ranker bonuses
        entity_match_bonus=0.18,
        entity_confidence_bonus_weight=0.10,
        same_section_bonus=0.08,
        section_path_match_bonus=0.10,
        section_structural_bonus=0.05,
        global_similarity_bonus_weight=0.20,
        relationship_confidence_bonus_weight=0.10,
        # Feature flags
        enable_entity_retriever=True,
        enable_entity_expansion=True,
        enable_section_expansion=True,
        enable_global_similarity_expansion=True,
        enable_section_title_search=True,
        prompt_evidence_max_chars=1000,
    )
    base.update(overrides)
    return Settings(**base)


def _item(block_id: str, method: str, score: float, text: str = "App Store risk") -> EvidenceItem:
    return EvidenceItem(
        block_id=block_id,
        type="paragraph",
        page=1,
        text=text,
        score=score,
        retrieval_method=method,
        relationship_path=[method],
    )


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def test_keyword_extracts_specific_phrases() -> None:
    terms = KeywordRetriever.extract_terms(
        "What does Apple say about App Store and Digital Markets Act risks?"
    )
    assert "App Store" in terms
    assert "Digital Markets Act" in terms
    assert "risks" in terms


# ---------------------------------------------------------------------------
# Seed merging
# ---------------------------------------------------------------------------

def test_hybrid_merge_dedupes_seed_blocks() -> None:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.settings = _settings()
    merged = retriever._merge_seeds(
        [
            [_item("p1", "vector", 0.8)],
            [_item("p1", "keyword", 2.0), _item("p2", "keyword", 1.0)],
        ]
    )
    assert [c.block_id for c in merged] == ["p1", "p2"]
    assert merged[0].retrieval_method == "keyword+vector"


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_hybrid_ranking_prefers_exact_keyword_seed() -> None:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.settings = _settings()
    seeds = [_item("p1", "vector", 0.7, "Generic financial text")]
    expanded = [_item("p2", "keyword", 1.0, "App Store Digital Markets Act litigation risk")]
    final, debug = retriever._rank_and_dedupe(
        "App Store Digital Markets Act risk", seeds, expanded
    )
    assert final[0].block_id == "p2"
    assert debug[0]["total"] >= debug[1]["total"]


def test_ranking_includes_entity_and_section_bonuses() -> None:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.settings = _settings()

    seed = _item("seed1", "vector", 0.8)
    seed.section_id = "sec_001"

    entity_item = _item("ent1", "entity", 0.0)
    entity_item.section_id = "sec_001"
    entity_item.matched_entities = [
        {
            "entity_id": "e1",
            "entity_match_score": 1.0,
            "mention_confidence": 0.9,
        }
    ]

    seeds = [seed]
    expanded = [entity_item]
    final, debug = retriever._rank_and_dedupe("test question", seeds, expanded)

    # entity_item should have non-zero entity and section bonuses
    ent_debug = next(d for d in debug if d["block_id"] == "ent1")
    assert ent_debug["entity_match_bonus"] > 0.0
    assert ent_debug["entity_match_bonus"] <= retriever.settings.entity_match_bonus
    assert ent_debug["same_section_bonus"] > 0.0


def test_ranking_partial_entity_match_scored_lower_than_exact() -> None:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.settings = _settings()

    exact_item = _item("exact1", "entity", 0.0)
    exact_item.matched_entities = [
        {"entity_id": "e1", "entity_match_score": 1.0, "mention_confidence": 0.9}
    ]
    partial_item = _item("partial1", "entity", 0.0)
    partial_item.matched_entities = [
        {"entity_id": "e2", "entity_match_score": 0.6, "mention_confidence": 0.9}
    ]

    _, debug = retriever._rank_and_dedupe("test", [], [exact_item, partial_item])
    exact_debug = next(d for d in debug if d["block_id"] == "exact1")
    partial_debug = next(d for d in debug if d["block_id"] == "partial1")
    assert exact_debug["entity_match_bonus"] > partial_debug["entity_match_bonus"]


def test_ranking_caps_global_similarity_bonus() -> None:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.settings = _settings()

    global_item = _item("glob1", "graph_expansion", 0.0)
    global_item.relationship_scope = "global"
    global_item.score = 1.0  # max possible similarity score

    _, debug = retriever._rank_and_dedupe("test", [], [global_item])
    glob_debug = next(d for d in debug if d["block_id"] == "glob1")
    assert glob_debug["global_similarity_bonus"] <= 0.20


def test_ranking_uses_refers_to_confidence() -> None:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.settings = _settings()

    conf_item = _item("conf1", "graph_expansion", 0.0)
    conf_item.relationship_confidence = 1.0

    _, debug = retriever._rank_and_dedupe("test", [], [conf_item])
    conf_debug = next(d for d in debug if d["block_id"] == "conf1")
    assert conf_debug["relationship_confidence_bonus"] > 0.0
    assert conf_debug["relationship_confidence_bonus"] <= 0.10


# ---------------------------------------------------------------------------
# EntityRetriever
# ---------------------------------------------------------------------------

def test_entity_retriever_returns_blocks_via_mentions() -> None:
    neo4j = FakeNeo4j()
    settings = _settings()
    retriever = EntityRetriever(neo4j, settings)
    items, trace = retriever.retrieve("What are the App Store risks?")
    assert len(items) > 0
    assert items[0].metadata["entity_match_type"] == "exact"
    assert items[0].metadata["entity_match_score"] == 1.0
    assert len(trace) == 1
    assert trace[0].action == "entity_search"


def test_entity_retriever_partial_match_lower_score() -> None:
    class PartialFakeNeo4j(FakeNeo4j):
        def entity_search_blocks(self, terms_lower, *, top_k, term_doc_freq_filter, document_id):
            return [_entity_block_row(
                block_id="p0005_b0001",
                entity_match_type="partial",
                entity_match_score=0.6,
            )]

    neo4j = PartialFakeNeo4j()
    settings = _settings()
    retriever = EntityRetriever(neo4j, settings)
    items, _ = retriever.retrieve("EU Digital Markets Act")
    assert items[0].metadata["entity_match_type"] == "partial"
    assert items[0].metadata["entity_match_score"] == 0.6


def test_entity_retriever_filters_high_freq_terms() -> None:
    captured: dict = {}

    class CaptureFakeNeo4j(FakeNeo4j):
        def entity_search_blocks(self, terms_lower, *, top_k, term_doc_freq_filter, document_id):
            captured["term_doc_freq_filter"] = term_doc_freq_filter
            return []

    settings = _settings(term_doc_freq_filter=0.30)
    retriever = EntityRetriever(CaptureFakeNeo4j(), settings)
    retriever.retrieve("test question about something")
    assert captured.get("term_doc_freq_filter") == 0.30


def test_entity_retriever_disabled_by_flag() -> None:
    neo4j = FakeNeo4j()
    settings = _settings(enable_entity_retriever=False)

    class FakeSemanticRetriever:
        def retrieve(self, question):
            return [_item("sem1", "vector", 0.8)], []

    class FakeKeywordRetriever:
        def retrieve(self, question):
            return [], []

    retriever = HybridRetriever(
        FakeSemanticRetriever(),  # type: ignore[arg-type]
        FakeKeywordRetriever(),  # type: ignore[arg-type]
        EntityRetriever(neo4j, settings),
        GraphExpander(neo4j, settings),
        settings,
    )
    bundle = retriever.retrieve("test question")
    # No block should have entity as its retrieval method
    entity_seeds = [b for b in bundle.seed_blocks if "entity" in b.retrieval_method]
    assert entity_seeds == []


# ---------------------------------------------------------------------------
# Graph expander: entity-mediated and section-aware arms
# ---------------------------------------------------------------------------

def test_entity_expansion_emits_items_via_shared_entity() -> None:
    neo4j = FakeNeo4j()
    settings = _settings()
    expander = GraphExpander(neo4j, settings)
    seed = _item("p0001_b0000", "vector", 0.8)
    expanded, trace = expander.expand([seed])

    entity_items = [i for i in expanded if i.retrieval_method == "entity_expansion"]
    assert len(entity_items) > 0
    assert entity_items[0].source_relationship == "MENTIONS_SHARED"

    entity_trace = [t for t in trace if t.action == "entity_expansion"]
    assert len(entity_trace) > 0


def test_section_expansion_emits_sibling_blocks() -> None:
    neo4j = FakeNeo4j()
    settings = _settings()
    expander = GraphExpander(neo4j, settings)
    seed = _item("p0001_b0000", "vector", 0.8)
    expanded, trace = expander.expand([seed])

    section_items = [i for i in expanded if i.retrieval_method == "section_expansion"]
    assert len(section_items) > 0
    assert section_items[0].source_relationship == "SAME_SECTION"


def test_section_expansion_structural_boost_in_metadata() -> None:
    class StructuralFakeNeo4j(FakeNeo4j):
        def expand_block_via_section(self, block_id, *, limit, document_id):
            return [_section_expand_row(block_id="p0001_b0010", structural_boost=1, btype="table")]

    settings = _settings()
    expander = GraphExpander(StructuralFakeNeo4j(), settings)
    seed = _item("p0001_b0000", "vector", 0.8)
    expanded, _ = expander.expand([seed])

    structural_items = [
        i for i in expanded
        if i.retrieval_method == "section_expansion" and i.metadata.get("structural_boost") == 1
    ]
    assert len(structural_items) > 0


# ---------------------------------------------------------------------------
# Section-title search
# ---------------------------------------------------------------------------

def test_section_title_search_returns_blocks_from_matching_section() -> None:
    """Blocks from sections whose title matches query terms become seeds."""
    from retrievers.keyword_retriever import KeywordRetriever

    neo4j = FakeNeo4j()
    settings = _settings()
    retriever = KeywordRetriever(neo4j, settings)
    items, trace = retriever.retrieve("Which products were announced in the third quarter of 2023?")

    section_title_items = [i for i in items if i.retrieval_method == "section_title"]
    assert len(section_title_items) > 0
    assert section_title_items[0].section_title == "Third Quarter of 2023"


def test_section_title_search_dedupes_against_text_search() -> None:
    """A block found by both text search and section-title search appears only once."""
    from retrievers.keyword_retriever import KeywordRetriever
    from tests.fakes import _block_row

    SHARED_ID = "p0010_b0001"

    class OverlapFakeNeo4j(FakeNeo4j):
        def keyword_search_blocks(self, query_text, *, terms, top_k, document_id, use_fulltext):
            return [_block_row(block_id=SHARED_ID, score=2.0)]

        def section_title_search_blocks(self, terms, *, top_k, document_id, term_min_len=4):
            return [_block_row(block_id=SHARED_ID, score=3.0,
                               section_title="Third Quarter of 2023")]

    settings = _settings()
    retriever = KeywordRetriever(OverlapFakeNeo4j(), settings)
    items, _ = retriever.retrieve("third quarter of 2023")
    ids = [i.block_id for i in items]
    assert ids.count(SHARED_ID) == 1


def test_section_title_method_bonus_higher_than_expansion() -> None:
    """A section_title seed scores method_bonus 0.28, above graph expansion (0.12)."""
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.settings = _settings()

    st_item = _item("st1", "section_title", 0.0)
    exp_item = _item("exp1", "graph_expansion", 0.0)

    _, debug = retriever._rank_and_dedupe("third quarter 2023", [], [st_item, exp_item])
    st_debug = next(d for d in debug if d["block_id"] == "st1")
    exp_debug = next(d for d in debug if d["block_id"] == "exp1")
    assert st_debug["method_bonus"] > exp_debug["method_bonus"]


def test_section_title_search_disabled_by_flag() -> None:
    from retrievers.keyword_retriever import KeywordRetriever

    captured: dict = {}

    class CaptureFakeNeo4j(FakeNeo4j):
        def section_title_search_blocks(self, terms, *, top_k, document_id, term_min_len=4):
            captured["called"] = True
            return []

    settings = _settings(enable_section_title_search=False)
    retriever = KeywordRetriever(CaptureFakeNeo4j(), settings)
    retriever.retrieve("third quarter of 2023 products")
    assert "called" not in captured


def test_global_similarity_arm_disabled_by_flag() -> None:
    captured: dict = {}

    class CaptureFakeNeo4j(FakeNeo4j):
        def expand_block(self, block_id, *, block_type, semantic_similarity_threshold, limit, global_threshold=2.0):
            captured["global_threshold"] = global_threshold
            return []

    settings = _settings(enable_global_similarity_expansion=False)
    expander = GraphExpander(CaptureFakeNeo4j(), settings)
    seed = _item("p0001_b0000", "vector", 0.8)
    expander.expand([seed])
    assert captured.get("global_threshold") == 2.0
