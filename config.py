from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ModuleNotFoundError:
    pass  # python-dotenv not installed; fall back to system environment variables


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str | None
    embed_server_url: str
    llm_server_url: str
    document_id: str | None
    vector_top_k: int
    keyword_top_k: int
    graph_expansion_limit: int
    final_evidence_limit: int
    semantic_similarity_threshold: float
    llm_max_tokens: int
    llm_temperature: float
    embed_max_chars: int
    request_timeout_seconds: int
    log_level: str
    keyword_exact_boost: float
    keyword_term_boost: float
    create_fulltext_index: bool
    # llama.cpp server auto-management
    llama_server_exe: str | None
    embed_model_path: str
    embed_server_port: int
    embed_n_ctx: int
    llm_model_path: str
    llm_server_port: int
    llm_n_ctx: int
    llama_health_timeout: int
    auto_start_servers: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j")),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            neo4j_database=os.getenv("NEO4J_DATABASE") or None,
            embed_server_url=os.getenv("EMBED_SERVER_URL", "http://127.0.0.1:8091"),
            llm_server_url=os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8092"),
            document_id=os.getenv("DOCUMENT_ID") or None,
            vector_top_k=_env_int("VECTOR_TOP_K", 8),
            keyword_top_k=_env_int("KEYWORD_TOP_K", 8),
            graph_expansion_limit=_env_int("GRAPH_EXPANSION_LIMIT", 5),
            final_evidence_limit=_env_int("FINAL_EVIDENCE_LIMIT", 10),
            semantic_similarity_threshold=_env_float("SEMANTIC_SIMILARITY_THRESHOLD", 0.50),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
            embed_max_chars=_env_int("EMBED_MAX_CHARS", 6000),
            request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 120),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            keyword_exact_boost=_env_float("KEYWORD_EXACT_BOOST", 0.25),
            keyword_term_boost=_env_float("KEYWORD_TERM_BOOST", 0.05),
            create_fulltext_index=_env_bool("CREATE_FULLTEXT_INDEX", False),
            # llama.cpp server auto-management
            llama_server_exe=os.getenv("LLAMA_SERVER_EXE") or None,
            embed_model_path=os.getenv("EMBED_MODEL_PATH", r"C:\llama-cpp\models\bge-m3-Q8_0.gguf"),
            embed_server_port=_env_int("EMBED_SERVER_PORT", 8091),
            embed_n_ctx=_env_int("EMBED_N_CTX", 8192),
            llm_model_path=os.getenv("LLM_MODEL_PATH", r"C:\llama-cpp\models\Qwen3.5-4B-Q8_0.gguf"),
            llm_server_port=_env_int("LLM_SERVER_PORT", 8092),
            llm_n_ctx=_env_int("LLM_N_CTX", 4096),
            llama_health_timeout=_env_int("LLAMA_HEALTH_TIMEOUT", 120),
            auto_start_servers=_env_bool("AUTO_START_SERVERS", False),
        )


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
