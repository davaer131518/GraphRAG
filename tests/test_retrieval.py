from config import Settings
from evidence.evidence_bundle import EvidenceItem
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.keyword_retriever import KeywordRetriever


def settings() -> Settings:
    return Settings(
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
        keyword_exact_boost=0.25,
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
    )


def item(block_id: str, method: str, score: float, text: str = "App Store risk") -> EvidenceItem:
    return EvidenceItem(
        block_id=block_id,
        type="paragraph",
        page=1,
        text=text,
        score=score,
        retrieval_method=method,
        relationship_path=[method],
    )


def test_keyword_extracts_specific_phrases() -> None:
    terms = KeywordRetriever.extract_terms("What does Apple say about App Store and Digital Markets Act risks?")
    assert "App Store" in terms
    assert "Digital Markets Act" in terms
    assert "risks" in terms


def test_hybrid_merge_dedupes_seed_blocks() -> None:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.settings = settings()
    merged = retriever._merge_seeds(
        [item("p1", "vector", 0.8)],
        [item("p1", "keyword", 2.0), item("p2", "keyword", 1.0)],
    )
    assert [candidate.block_id for candidate in merged] == ["p1", "p2"]
    assert merged[0].retrieval_method == "keyword+vector"


def test_hybrid_ranking_prefers_exact_keyword_seed() -> None:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.settings = settings()
    seeds = [item("p1", "vector", 0.7, "Generic financial text")]
    expanded = [item("p2", "keyword", 1.0, "App Store Digital Markets Act litigation risk")]
    final, debug = retriever._rank_and_dedupe("App Store Digital Markets Act risk", seeds, expanded)
    assert final[0].block_id == "p2"
    assert debug[0]["total"] >= debug[1]["total"]
