from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import chainlit as cl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyst import TraceablePDFAnalyst
from config import Settings, configure_logging
from embeddings_client import EmbeddingsClient
from evidence.trace_formatter import (
    format_answer_markdown,
    format_debug_json,
    format_document_map,
    format_table_explorer,
)
from generation.answer_generator import AnswerGenerator
from llm_client import LLMClient
from neo4j_client import Neo4jClient
from retrievers.graph_expander import GraphExpander
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.keyword_retriever import KeywordRetriever
from retrievers.semantic_retriever import SemanticRetriever
from server_manager import ServerManager


def build_analyst() -> tuple[TraceablePDFAnalyst, Neo4jClient, ServerManager | None]:
    settings = Settings.from_env()
    configure_logging(settings)

    server_manager: ServerManager | None = None
    if settings.auto_start_servers:
        server_manager = ServerManager.from_settings(settings)
        server_manager.start_all()

    neo4j = Neo4jClient(settings)
    neo4j.verify_connectivity()
    neo4j.ensure_fulltext_index()
    embeddings = EmbeddingsClient(settings)
    llm = LLMClient(settings)
    semantic = SemanticRetriever(neo4j, embeddings, settings)
    keyword = KeywordRetriever(neo4j, settings)
    graph_expander = GraphExpander(neo4j, settings)
    retriever = HybridRetriever(semantic, keyword, graph_expander, settings)
    answer_generator = AnswerGenerator(llm)
    analyst = TraceablePDFAnalyst(retriever, answer_generator, neo4j, settings)
    return analyst, neo4j, server_manager


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        analyst, neo4j, server_manager = await asyncio.to_thread(build_analyst)
    except Exception as exc:
        await cl.Message(
            content=(
                "I could not initialize the Traceable PDF Analyst.\n\n"
                f"Error: `{exc}`\n\n"
                "Check Neo4j, `NEO4J_*`, `EMBED_SERVER_URL`, `LLM_SERVER_URL`, "
                "and `LLAMA_SERVER_EXE` / model paths if `AUTO_START_SERVERS=true`."
            )
        ).send()
        return

    cl.user_session.set("analyst", analyst)
    cl.user_session.set("neo4j_client", neo4j)
    cl.user_session.set("server_manager", server_manager)
    cl.user_session.set("last_question", None)
    cl.user_session.set("last_evidence_bundle", None)
    cl.user_session.set("last_trace", None)
    cl.user_session.set("last_answer_json", None)
    cl.user_session.set("selected_document_id", analyst.settings.document_id)

    await cl.Message(
        content=(
            "Ask questions about the parsed PDF knowledge graph. I will answer using "
            "the Neo4j graph and show page numbers, block IDs, source snippets, and "
            "the retrieval trace.\n\n"
            "Commands: `/table <block_id>`, `/table <block_id> <RELATION>`, `/map`, `/debug last`."
        )
    ).send()


@cl.on_chat_end
async def on_chat_end() -> None:
    neo4j = cl.user_session.get("neo4j_client")
    if neo4j is not None:
        await asyncio.to_thread(neo4j.close)
    server_manager = cl.user_session.get("server_manager")
    if server_manager is not None:
        await asyncio.to_thread(server_manager.stop_all)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    analyst: TraceablePDFAnalyst | None = cl.user_session.get("analyst")
    if analyst is None:
        await cl.Message(content="The analyst is not initialized. Restart the chat after checking configuration.").send()
        return

    content = message.content.strip()
    if not content:
        await cl.Message(content="Please enter a question or command.").send()
        return
    if content.startswith("/"):
        await handle_command(analyst, content)
        return
    await answer_question(analyst, content)


async def answer_question(analyst: TraceablePDFAnalyst, question: str) -> None:
    async with cl.Step(name="Searching vector index"):
        pass
    async with cl.Step(name="Running keyword search"):
        pass
    async with cl.Step(name="Expanding graph context"):
        pass
    async with cl.Step(name="Building evidence bundle and generating answer"):
        answer = await asyncio.to_thread(analyst.ask, question)

    cl.user_session.set("last_question", question)
    cl.user_session.set("last_evidence_bundle", answer.raw_evidence_bundle)
    cl.user_session.set("last_trace", answer.trace)
    cl.user_session.set("last_answer_json", answer.raw_answer_json)
    await cl.Message(content=format_answer_markdown(answer)).send()


async def handle_command(analyst: TraceablePDFAnalyst, content: str) -> None:
    parts = content.split()
    command = parts[0].lower()
    try:
        if command == "/table":
            await handle_table_command(analyst, parts)
        elif command == "/map":
            result = await asyncio.to_thread(analyst.document_map)
            await cl.Message(content=format_document_map(result)).send()
        elif command == "/debug" and len(parts) == 2 and parts[1].lower() == "last":
            await handle_debug_last()
        else:
            await cl.Message(content=command_help()).send()
    except Exception as exc:
        await cl.Message(content=f"Command failed: `{exc}`").send()


async def handle_table_command(analyst: TraceablePDFAnalyst, parts: list[str]) -> None:
    if len(parts) not in {2, 3}:
        await cl.Message(content="Usage: `/table <block_id>` or `/table <block_id> <RELATION>`.").send()
        return
    table_id = parts[1]
    relation = parts[2].upper() if len(parts) == 3 else None
    result = await asyncio.to_thread(analyst.table_explorer, table_id, relation)
    await cl.Message(content=format_table_explorer(result)).send()


async def handle_debug_last() -> None:
    bundle = cl.user_session.get("last_evidence_bundle")
    trace = cl.user_session.get("last_trace")
    answer_json = cl.user_session.get("last_answer_json")
    question = cl.user_session.get("last_question")
    if bundle is None:
        await cl.Message(content="No previous retrieval is available. Ask a question first.").send()
        return
    debug = {
        "last_question": question,
        "evidence_bundle": bundle.to_dict() if hasattr(bundle, "to_dict") else bundle,
        "trace": trace,
        "raw_answer_json": answer_json,
    }
    await cl.Message(content="### Debug: Last Retrieval\n\n" + format_debug_json(debug)).send()


def command_help() -> str:
    return (
        "Available commands:\n\n"
        "- `/table <block_id>`\n"
        "- `/table <block_id> <COMPARES|SUPPLEMENTS|CONTRASTS|ABLATES>`\n"
        "- `/map`\n"
        "- `/debug last`"
    )
