# Traceable PDF Analyst

A local Graph-RAG chat system that answers questions about a parsed PDF document using a Neo4j knowledge graph. Every answer includes the source blocks, page numbers, section headings, relationship paths used during retrieval, and a step-by-step evidence trace.

The user interface is built with Chainlit. All retrieval, graph expansion, evidence bundling, prompting, and answer parsing happen in backend modules so the same analyst can be used outside the UI.

---

## How it works

The system does not do plain text search over a document. It reads from a pre-built Neo4j knowledge graph where every block of text from a parsed PDF is a node (`Block`) connected to other blocks through typed relationships that encode reading order, section structure, caption connections, semantic similarity, and LLM-labelled table relationships.

When you ask a question the pipeline runs in three layers:

### Layer 1 — Seed retrieval

- **Vector search**: the question is embedded with the same bge-m3 model that was used to embed the blocks, and Neo4j's native ANN vector index returns the top-K semantically similar blocks.
- **Keyword search**: key phrases and terms are extracted from the question and matched against `Block.text`, preferring a full-text index if available and falling back to a `CONTAINS` query.
- Both result sets are merged and deduplicated.

### Layer 2 — Graph expansion

The top seed blocks are expanded through the KG using relationship-specific rules:

- **Paragraph / caption / list item seeds**: follow `REFERS_TO` (blocks that discuss a table), `SEMANTICALLY_SIMILAR` (related tables above a score threshold), and a local `PRECEDES` window on the same page.
- **Table seeds**: follow `CONTEXT_BEFORE` and `CONTEXT_AFTER` (the blocks immediately surrounding the table in reading order), incoming `REFERS_TO` (paragraphs discussing it), incoming `DESCRIBES` (its caption), and `COMPARES` / `SUPPLEMENTS` / `CONTRASTS` / `ABLATES` (table-to-table relationships labelled by the LLM during graph construction).
- **Heading seeds**: follow `INTRODUCES` to the blocks that belong to that section.

All expansion is bounded and explicit. No variable-length traversal.

### Layer 3 — Answer generation

The expanded evidence is ranked by a combination of vector score, keyword match, relationship type weight, block type, and exact phrase overlap. The top-N blocks are packaged into a structured evidence bundle and sent to a local LLM (Qwen3.5-4B via llama.cpp) with a strict system prompt that requires a JSON answer citing every factual claim back to a specific page and block ID.

```
Question
  -> embed question        (bge-m3 on EMBED_SERVER_URL)
  -> vector search         (Neo4j block_embedding_index)
  -> keyword search        (Neo4j full-text or CONTAINS)
  -> merge seeds
  -> graph expansion       (typed KG relationship rules)
  -> rank + dedupe
  -> build evidence bundle
  -> LLM answer            (Qwen3.5-4B on LLM_SERVER_URL)
  -> render in Chainlit
```

---

## Prerequisites

| Component | Required |
|---|---|
| Neo4j 5.11+ with the KG already loaded | Required |
| `block_embedding_index` vector index in Neo4j | Required |
| llama.cpp embedding server running bge-m3 | Required for every question |
| llama.cpp LLM server running Qwen3.5-4B (or compatible) | Required for every question |

The KG is built separately by a graph-construction notebook (not included here) run against a `document.json` parser output. Once the graph is in Neo4j, that notebook and `document.json` are not needed to run the Chainlit app.

---

## Setup

### 1. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 2. Set environment variables

```powershell
$env:NEO4J_URI       = "neo4j://127.0.0.1:7687"
$env:NEO4J_USERNAME  = "neo4j"
$env:NEO4J_PASSWORD  = "<your password>"
$env:NEO4J_DATABASE  = "neo4j"
$env:EMBED_SERVER_URL = "http://127.0.0.1:8091"
$env:LLM_SERVER_URL   = "http://127.0.0.1:8092"
```

Optional:

```powershell
$env:DOCUMENT_ID                   = "<source_sha256 from document.json>"
$env:VECTOR_TOP_K                  = "8"
$env:KEYWORD_TOP_K                 = "8"
$env:GRAPH_EXPANSION_LIMIT         = "5"
$env:FINAL_EVIDENCE_LIMIT          = "10"
$env:SEMANTIC_SIMILARITY_THRESHOLD = "0.50"
$env:LOG_LEVEL                     = "INFO"
$env:CREATE_FULLTEXT_INDEX         = "false"
# LLM generation
$env:LLM_MAX_TOKENS                = "1024"
$env:LLM_TEMPERATURE               = "0.0"
# Embedding and request tuning
$env:EMBED_MAX_CHARS               = "6000"
$env:REQUEST_TIMEOUT_SECONDS       = "120"
# Keyword ranking boosts
$env:KEYWORD_EXACT_BOOST           = "0.25"
$env:KEYWORD_TERM_BOOST            = "0.05"
# llama.cpp server ports and context sizes (used with AUTO_START_SERVERS)
$env:EMBED_SERVER_PORT             = "8091"
$env:EMBED_N_CTX                   = "8192"
$env:LLM_SERVER_PORT               = "8092"
$env:LLM_N_CTX                     = "4096"
$env:LLAMA_HEALTH_TIMEOUT          = "120"
```

`DOCUMENT_ID` scopes all queries to a single document node. Leave it unset to query across all documents in the graph.

`CREATE_FULLTEXT_INDEX` creates the Neo4j full-text index on first run so keyword search uses Lucene scoring. Set to `true` once; after that it is safe to leave as `false`.

### 3. Configure server auto-start (recommended)

The app can launch both llama.cpp servers automatically when a chat session starts and shut them down when the session ends.

Set these in `.env`:

```
AUTO_START_SERVERS=true
LLAMA_SERVER_EXE=C:\llama-cpp\llama-server.exe
EMBED_MODEL_PATH=C:\llama-cpp\models\bge-m3-Q8_0.gguf
EMBED_SERVER_PORT=8091
EMBED_N_CTX=8192
LLM_MODEL_PATH=C:\llama-cpp\models\Qwen3.5-4B-Q8_0.gguf
LLM_SERVER_PORT=8092
LLM_N_CTX=4096
LLAMA_HEALTH_TIMEOUT=120
```

If a server is already running on the expected port when the app starts, it is reused and will not be stopped on chat end.

If you prefer to manage the servers manually, set `AUTO_START_SERVERS=false` and start them yourself before running Chainlit:

```powershell
# Embedding server
C:\llama-cpp\llama-server.exe -m C:\llama-cpp\models\bge-m3-Q8_0.gguf `
  --port 8091 --host 127.0.0.1 -ngl -1 --embedding --pooling mean -c 8192

# LLM server
C:\llama-cpp\llama-server.exe -m C:\llama-cpp\models\Qwen3.5-4B-Q8_0.gguf `
  --port 8092 --host 127.0.0.1 -ngl -1 -c 4096
```

### 4. Verify the knowledge graph exists (Neo4j Browser)

Run these queries in the Neo4j Browser:

```cypher
MATCH (d:Document) RETURN d.doc_id, d.filename, d.num_pages LIMIT 5;
```

```cypher
MATCH (b:Block) RETURN b.type, count(*) ORDER BY count(*) DESC;
```

```cypher
SHOW INDEXES YIELD name, type, state
WHERE name = 'block_embedding_index'
RETURN name, type, state;
```

### 5. Run Chainlit

```powershell
chainlit run apps/chainlit_app.py -w
```

Open `http://localhost:8000` in your browser.

---

## Using the app

### Asking questions

Type any natural-language question. The system retrieves evidence from the KG, generates a structured answer, and returns:

- A natural-language answer.
- Confidence level: `high`, `medium`, or `low`.
- Source cards showing page, block ID, block type, section heading, a snippet, and why that source was used.
- A step-by-step evidence trace showing which relationships were followed.

Example questions:

```
What does Apple say about risks related to the App Store?
What evidence discusses the EU Digital Markets Act?
Which blocks explain share repurchase activity?
What does the Risk Factors section say about cybersecurity?
```

### Commands

| Command | What it does |
|---|---|
| `/table <block_id>` | Show all related tables through `COMPARES`, `SUPPLEMENTS`, `CONTRASTS`, and `ABLATES`. |
| `/table <block_id> <RELATION>` | Filter to one relationship type, e.g. `/table p0033_b0002 SUPPLEMENTS`. |
| `/map` | Generate a structured Markdown document map from section headings, key blocks, tables, and relationship summaries. |
| `/debug last` | Show the full evidence bundle, per-block ranking scores, raw answer JSON, and trace from the last question. |

---

## Project structure

```
GraphRAG/
├── apps/
│   └── chainlit_app.py         Chainlit UI — session init, message routing, command handling
├── evidence/
│   ├── evidence_bundle.py      Typed data models: EvidenceItem, EvidenceBundle, AnalystAnswer, etc.
│   └── trace_formatter.py      Text and Markdown renderers for answers, sources, traces, debug output
├── generation/
│   ├── prompts.py              System and user prompt templates
│   └── answer_generator.py     JSON answer parsing, validation, and fallback handling
├── retrievers/
│   ├── semantic_retriever.py   Vector search seeds
│   ├── keyword_retriever.py    Keyword/full-text search seeds + term extraction
│   ├── graph_expander.py       Relationship-specific expansion policy
│   └── hybrid_retriever.py     Merge, dedupe, rank, and build evidence bundle
├── tests/
│   ├── test_evidence_bundle.py
│   ├── test_retrieval.py
│   ├── test_answer_generator.py
│   └── test_analyst_outputs.py
├── analyst.py                  TraceablePDFAnalyst: ask(), table_explorer(), document_map()
├── config.py                   Settings dataclass loaded from environment variables
├── embeddings_client.py        llama.cpp /v1/embeddings client with L2 normalization and retry
├── llm_client.py               llama.cpp /v1/chat/completions client with Qwen3 think-mode stripping
├── neo4j_client.py             All Cypher queries, Neo4j driver lifecycle
├── server_manager.py           LlamaServer / ServerManager: auto-start and stop llama.cpp processes
└── requirements.txt
```

---

## Running tests

```powershell
python -m pytest -q
```

All tests use mocked Neo4j and LLM clients so they run without any running services.

---

## Architecture notes

**Chainlit is a thin shell.** It initializes the analyst into session state on `@on_chat_start`, calls `analyst.ask(question)` on each message, and formats the result for the UI. No Cypher, no ranking logic, and no prompt construction happen in the Chainlit file.

**The evidence trace is explicit and inspectable.** Every evidence item carries its `relationship_path` (the sequence of KG edges followed to reach it) and its `retrieval_method`. The `/debug last` command exposes the full bundle including per-block ranking feature scores.

**The same analyst works without Chainlit.** The `TraceablePDFAnalyst` class in `analyst.py` can be imported directly in scripts or notebooks. The `HybridRetriever` can also be used independently.

**Server lifecycle is handled by `server_manager.py`.** The `LlamaServer` class wraps a single `llama-server.exe` process; `ServerManager` coordinates the embedding/LLM pair. If `AUTO_START_SERVERS=true`, the Chainlit `on_chat_start` hook starts both servers and `on_chat_end` stops only the ones it started (servers that were already running are left untouched).

**Graph expansion is bounded.** The expander only follows seeds up to `GRAPH_EXPANSION_LIMIT` and uses a fixed per-block limit on expansion rows. There is no variable-length traversal.

---

## Graph schema (built by the KG notebook)

```
(:Document {doc_id, filename, num_pages})
(:Page {page_id, page_number})
(:Block {block_id, type, text, page_number, reading_order, embedding})

(:Page)-[:PART_OF]->(:Document)
(:Block)-[:ON_PAGE]->(:Page)
(:Block)-[:PRECEDES]->(:Block)           reading order chain
(:Block)-[:DESCRIBES]->(:Block)          caption -> table/figure
(:Block)-[:INTRODUCES]->(:Block)         heading -> section content
(:Block)-[:IN_SECTION]->(:Block)         block -> deepest parent heading
(:Block)-[:CONTEXT_BEFORE]->(:Block)     N blocks before a table
(:Block)-[:CONTEXT_AFTER]->(:Block)      table -> N blocks after it
(:Block)-[:REFERS_TO {methods,mention}]->(:Block)
(:Block)-[:SEMANTICALLY_SIMILAR {score}]->(:Block)
(:Block)-[:COMPARES|SUPPLEMENTS|CONTRASTS|ABLATES {reason}]->(:Block)
```

---

## Future: multi-document support

The architecture is designed for this. All retrieval queries already accept an optional `DOCUMENT_ID` to scope queries to a single document. Adding cross-document comparison will require:

- Multiple `Document` nodes in Neo4j (each PDF gets its own node).
- Cross-document relationship edges such as `SIMILAR_SECTION`, `REPORTS_SAME_METRIC`, and `CHANGED_FROM`.
- A document selector in the Chainlit session.
- Comparative prompts that cite sources from each document separately.
