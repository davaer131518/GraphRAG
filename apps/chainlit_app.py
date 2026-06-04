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
    format_related_documents,
    format_table_explorer,
)
from generation.answer_generator import AnswerGenerator
from llm_client import LLMClient
from neo4j_client import Neo4jClient
from retrievers.entity_retriever import EntityRetriever
from retrievers.graph_expander import GraphExpander
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.keyword_retriever import KeywordRetriever
from retrievers.scope import RetrievalScope
from retrievers.scope_resolver import ScopeResolver
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
    entity_retriever = EntityRetriever(neo4j, settings)
    graph_expander = GraphExpander(neo4j, settings)
    retriever = HybridRetriever(semantic, keyword, entity_retriever, graph_expander, settings)
    answer_generator = AnswerGenerator(llm, settings)
    scope_resolver = ScopeResolver(settings, llm)
    analyst = TraceablePDFAnalyst(retriever, answer_generator, neo4j, settings, scope_resolver)
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

    # Load and cache the document list once per session
    try:
        documents = await asyncio.to_thread(neo4j.list_documents)
    except Exception as exc:
        documents = []
        import logging
        logging.getLogger(__name__).warning("Could not list documents: %s", exc)

    corpus_id = documents[0].get("corpus_id") if documents else None

    # Seed sticky scope from DOCUMENT_ID env (back-compat with single-doc usage)
    seed_doc_id = analyst.settings.document_id
    if seed_doc_id:
        session_scope = RetrievalScope(
            document_ids=(seed_doc_id,),
            corpus_id=corpus_id,
            rationale=f"Pinned at startup from DOCUMENT_ID={seed_doc_id}.",
            source="sticky",
        )
    else:
        session_scope = None  # no pin; query-driven / whole corpus

    cl.user_session.set("analyst", analyst)
    cl.user_session.set("neo4j_client", neo4j)
    cl.user_session.set("server_manager", server_manager)
    cl.user_session.set("documents", documents)
    cl.user_session.set("corpus_id", corpus_id)
    cl.user_session.set("session_scope", session_scope)
    cl.user_session.set("selected_document_id", seed_doc_id)  # brief's named var (back-compat)
    cl.user_session.set("last_question", None)
    cl.user_session.set("last_evidence_bundle", None)
    cl.user_session.set("last_trace", None)
    cl.user_session.set("last_answer_json", None)

    doc_count = len(documents)
    scope_note = (
        f" (pinned to `{seed_doc_id}`)" if seed_doc_id
        else f" ({doc_count} document{'s' if doc_count != 1 else ''} available)"
    )
    await cl.Message(
        content=(
            f"Ask questions about the knowledge graph{scope_note}. I will answer using "
            "the Neo4j graph and show page numbers, block IDs, source snippets, and "
            "the retrieval trace.\n\n"
            + command_help()
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
    scope: RetrievalScope | None = cl.user_session.get("session_scope")

    async with cl.Step(name="Searching vector index"):
        pass
    async with cl.Step(name="Running keyword search"):
        pass
    if analyst.settings.enable_entity_retriever:
        async with cl.Step(name="Searching entity mentions"):
            pass
    async with cl.Step(name="Expanding graph context"):
        pass
    async with cl.Step(name="Building evidence bundle and generating answer"):
        answer = await asyncio.to_thread(analyst.ask, question, scope)

    cl.user_session.set("last_question", question)
    cl.user_session.set("last_evidence_bundle", answer.raw_evidence_bundle)
    cl.user_session.set("last_trace", answer.trace)
    cl.user_session.set("last_answer_json", answer.raw_answer_json)
    async with cl.Step(name="Evidence Trace") as trace_step:
        trace_step.output = "```text\n" + "\n".join(answer.trace) + "\n```"
    await cl.Message(content=format_answer_markdown(answer)).send()


async def handle_command(analyst: TraceablePDFAnalyst, content: str) -> None:
    parts = content.split()
    command = parts[0].lower()
    try:
        if command == "/table":
            await handle_table_command(analyst, parts)
        elif command == "/map":
            scope: RetrievalScope | None = cl.user_session.get("session_scope")
            result = await asyncio.to_thread(analyst.document_map, scope)
            await cl.Message(content=format_document_map(result)).send()
        elif command == "/debug" and len(parts) == 2 and parts[1].lower() == "last":
            await handle_debug_last()
        elif command == "/docs":
            await handle_docs_command()
        elif command == "/use" and len(parts) == 2:
            await handle_use_command(parts[1])
        elif command == "/scope":
            await handle_scope_command(parts)
        elif command == "/related":
            await handle_related_command(analyst, parts)
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
    scope: RetrievalScope | None = cl.user_session.get("session_scope")
    result = await asyncio.to_thread(analyst.table_explorer, table_id, relation, scope=scope)
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


async def handle_docs_command() -> None:
    documents: list[dict] = cl.user_session.get("documents") or []
    if not documents:
        await cl.Message(content="No documents found in the graph.").send()
        return
    lines = ["### Available Documents\n"]
    for doc in documents:
        fname = doc.get("filename") or doc["doc_id"]
        pages = doc.get("num_pages")
        family = doc.get("doc_family") or ""
        version = doc.get("version_id") or ""
        pub = doc.get("published_at") or ""
        meta_parts = [p for p in [family, version, pub] if p]
        meta = f" — {', '.join(meta_parts)}" if meta_parts else ""
        page_str = f", {pages} pages" if pages else ""
        lines.append(f"- `{doc['doc_id']}` — **{fname}**{page_str}{meta}")
    scope: RetrievalScope | None = cl.user_session.get("session_scope")
    if scope and scope.is_scoped:
        lines.append(f"\n> Active scope: {scope.rationale}")
    await cl.Message(content="\n".join(lines)).send()


async def handle_use_command(doc_id: str) -> None:
    documents: list[dict] = cl.user_session.get("documents") or []
    corpus_id: str | None = cl.user_session.get("corpus_id")
    valid_ids = {d["doc_id"] for d in documents}
    if doc_id not in valid_ids:
        doc_list = "\n".join(f"  - `{d['doc_id']}` ({d.get('filename') or ''})" for d in documents[:10])
        await cl.Message(
            content=f"Unknown doc_id `{doc_id}`. Available:\n{doc_list}\n\nUse `/docs` for the full list."
        ).send()
        return
    doc = next(d for d in documents if d["doc_id"] == doc_id)
    fname = doc.get("filename") or doc_id
    scope = RetrievalScope(
        document_ids=(doc_id,),
        corpus_id=corpus_id,
        rationale=f"Pinned by /use to '{fname}' (doc_id={doc_id}).",
        source="sticky",
    )
    cl.user_session.set("session_scope", scope)
    cl.user_session.set("selected_document_id", doc_id)
    await cl.Message(content=f"> Active scope: **{fname}** (`{doc_id}`)").send()


async def handle_scope_command(parts: list[str]) -> None:
    documents: list[dict] = cl.user_session.get("documents") or []
    corpus_id: str | None = cl.user_session.get("corpus_id")
    if len(parts) < 2:
        await cl.Message(content="Usage: `/scope all` or `/scope <doc_id>[,<doc_id>…]`").send()
        return
    arg = parts[1]
    if arg.lower() == "all":
        scope = RetrievalScope.whole_corpus(corpus_id=corpus_id)
        cl.user_session.set("session_scope", None)  # None = no sticky pin; query-driven resumes
        cl.user_session.set("selected_document_id", None)
        await cl.Message(content="> Active scope: **whole corpus** (query-driven scope resumed).").send()
        return
    # Parse comma-separated list (may be in one part or split across whitespace)
    raw_ids = " ".join(parts[1:]).replace(",", " ").split()
    valid_ids = {d["doc_id"] for d in documents}
    good_ids = [id_.strip() for id_ in raw_ids if id_.strip() in valid_ids]
    bad_ids = [id_.strip() for id_ in raw_ids if id_.strip() and id_.strip() not in valid_ids]
    if bad_ids:
        await cl.Message(content=f"Unknown doc_id(s): {', '.join(f'`{b}`' for b in bad_ids)}. Use `/docs` to see valid IDs.").send()
        return
    if not good_ids:
        await cl.Message(content="No valid doc_ids provided. Use `/docs` to see available documents.").send()
        return
    fnames = ", ".join(
        f"'{d.get('filename') or d['doc_id']}'"
        for d in documents
        if d["doc_id"] in good_ids
    )
    scope = RetrievalScope(
        document_ids=tuple(good_ids),
        corpus_id=corpus_id,
        rationale=f"Pinned by /scope to {len(good_ids)} document(s): {fnames}.",
        source="sticky",
    )
    cl.user_session.set("session_scope", scope)
    cl.user_session.set("selected_document_id", good_ids[0] if len(good_ids) == 1 else None)
    await cl.Message(content=f"> Active scope: {scope.rationale}").send()


async def handle_related_command(analyst: TraceablePDFAnalyst, parts: list[str]) -> None:
    scope: RetrievalScope | None = cl.user_session.get("session_scope")
    documents: list[dict] = cl.user_session.get("documents") or []

    # Determine which doc_id to look up related documents for
    if len(parts) >= 2:
        doc_id = parts[1]
    elif scope and scope.is_scoped and len(scope.document_ids) == 1:
        doc_id = scope.document_ids[0]
    else:
        await cl.Message(
            content="Usage: `/related <doc_id>`. Pin a single document first with `/use <doc_id>`, or provide a doc_id directly."
        ).send()
        return

    valid_ids = {d["doc_id"] for d in documents}
    if doc_id not in valid_ids:
        await cl.Message(content=f"Unknown doc_id `{doc_id}`. Use `/docs` to see available documents.").send()
        return

    related = await asyncio.to_thread(analyst.related_documents, doc_id, scope=scope)
    if not related:
        await cl.Message(content=f"No related documents found for `{doc_id}`.").send()
        return

    src_doc = next((d for d in documents if d["doc_id"] == doc_id), {"filename": doc_id})
    src_fname = src_doc.get("filename") or doc_id
    await cl.Message(content=format_related_documents(related, source_label=src_fname)).send()


def command_help() -> str:
    return (
        "Available commands:\n\n"
        "- `/docs` — list all documents in the corpus\n"
        "- `/use <doc_id>` — pin to a single document\n"
        "- `/scope <doc_id>[,<doc_id>…]` — pin to a set of documents\n"
        "- `/scope all` — release pin (return to query-driven scope)\n"
        "- `/related [doc_id]` — show RELATED_DOCUMENT neighbors\n"
        "- `/table <block_id>` — explore table relationships\n"
        "- `/table <block_id> <COMPARES|SUPPLEMENTS|CONTRASTS|ABLATES>` — filtered\n"
        "- `/map` — document structure map\n"
        "- `/debug last` — show last retrieval trace"
    )
