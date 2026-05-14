# Traceable PDF Analyst

A local Graph-RAG chat system that answers questions about a parsed PDF document using a Neo4j knowledge graph. Every answer includes the source blocks, page numbers, section headings, relationship paths used during retrieval, and a step-by-step evidence trace.

The user interface is built with Chainlit. All retrieval, graph expansion, evidence bundling, prompting, and answer parsing happen in backend modules so the same analyst can be used outside the UI.

---

## How it works

The system does not do plain text search over a document. It reads from a pre-built Neo4j knowledge graph where every block of text from a parsed PDF is a node (`Block`) connected to other blocks through typed relationships that encode reading order, section structure, caption connections, semantic similarity, and LLM-labelled table relationships.

When you ask a question the pipeline runs in three layers:

### Layer 1 — Seed retrieval

Three retrievers run in parallel and their results are merged and deduplicated on `block_id`:

- **Vector search**: the question is embedded with the same bge-m3 model used to embed the blocks, and Neo4j's native ANN vector index returns the top-K semantically similar blocks.
- **Keyword search**: key phrases and terms are extracted from the question and matched against `Block.text`, preferring a full-text index if available and falling back to a `CONTAINS` query.
- **Entity search**: extracted terms are matched against `:Entity` nodes using a 3-tier strategy — exact canonical name match (score 1.0), alias match (score 0.9), and partial substring match (score 0.6, min 4 chars). Each matched block carries the entity name, type, and match score for use in ranking. High-frequency generic terms (`TERM` entities above `TERM_DOC_FREQ_FILTER`) are excluded. Can be disabled with `ENABLE_ENTITY_RETRIEVER=false`.

### Layer 2 — Graph expansion

Each seed block is expanded through the KG using three arms:

**Arm 1 — Relationship-typed** (dispatched by `Block.type`):
- **Paragraph / caption / list item seeds**: follow `REFERS_TO` (blocks that discuss this block), `SEMANTICALLY_SIMILAR` with `scope='table'` (related tables above a configurable threshold), `SEMANTICALLY_SIMILAR` with `scope='global'` (any block, higher threshold — disable with `ENABLE_GLOBAL_SIMILARITY_EXPANSION=false`), and a local `PRECEDES` window on the same page.
- **Table seeds**: follow `CONTEXT_BEFORE` / `CONTEXT_AFTER` (blocks immediately surrounding the table), incoming `REFERS_TO` / `DESCRIBES` (paragraphs and captions), and table-pair edges `COMPARES` / `SUPPLEMENTS` / `CONTRASTS` / `ABLATES` / `TABLE_RELATES_TO` (the last is an APOC fallback; its logical label is normalised in code).
- **Heading seeds**: follow `INTRODUCES` to the blocks that belong to that section.

**Arm 2 — Entity-mediated** (`ENABLE_ENTITY_EXPANSION=true`): traverses `seed -[:MENTIONS]-> Entity <-[:MENTIONS]- related` to find blocks that mention the same entities as the seed. Bounded by `ENTITY_EXPANSION_ENTITIES_PER_SEED` and `ENTITY_EXPANSION_BLOCKS_PER_ENTITY`.

**Arm 3 — Section-aware** (`ENABLE_SECTION_EXPANSION=true`): returns sibling blocks in the same `:Section` ordered by reading-order proximity. Bounded by `SECTION_EXPANSION_LIMIT`. Structural blocks (tables, figures, captions) receive an additional `section_structural_bonus` in ranking.

All expansion is bounded and explicit. No variable-length Cypher traversal.

### Layer 3 — Answer generation

The expanded evidence is ranked by a multi-signal score:

- **Retrieval method bonuses**: vector (0.32), keyword (0.30), expansion (0.12)
- **Relationship type weight**: `REFERS_TO` (0.28), `DESCRIBES` (0.24), table-pair rels (0.18–0.22), `MENTIONS_SHARED` (0.14), `SAME_SECTION` (0.10), etc.
- **Block type weight**, **vector score**, **exact phrase overlap**
- **Entity bonuses**: `entity_match_bonus` scaled by match score (exact 1.0 / alias 0.9 / partial 0.6); `entity_confidence_bonus` scaled by `MENTIONS.confidence`
- **Section bonuses**: `same_section_bonus` (seed and candidate share a `:Section`), `section_path_match_bonus` (question token in section path/title), `section_structural_bonus` (table/figure/caption reached via section expansion)
- **Graph bonuses**: `global_similarity_bonus` (capped at 0.20), `relationship_confidence_bonus` (capped at 0.10)

The top-`FINAL_EVIDENCE_LIMIT` blocks are packaged into a structured evidence bundle and sent to a local LLM (Qwen3.5-4B via llama.cpp) with a strict system prompt that requires a JSON answer citing every factual claim back to a specific page and block ID.

```
Question
  -> embed question          (bge-m3 on EMBED_SERVER_URL)
  -> vector search           (Neo4j block_embedding_index)
  -> keyword search          (Neo4j full-text or CONTAINS)
  -> entity search           (Neo4j :Entity via MENTIONS, 3-tier matching)
  -> merge + dedupe seeds
  -> graph expansion         (relationship-typed + entity-mediated + section-aware)
  -> rank + dedupe
  -> build evidence bundle
  -> LLM answer              (Qwen3.5-4B on LLM_SERVER_URL)
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
# Entity retrieval bounds
$env:ENTITY_TOP_K                            = "8"
$env:ENTITY_EXPANSION_ENTITIES_PER_SEED      = "4"
$env:ENTITY_EXPANSION_BLOCKS_PER_ENTITY      = "5"
$env:SECTION_EXPANSION_LIMIT                 = "6"
$env:GLOBAL_SIMILARITY_THRESHOLD             = "0.65"
$env:TERM_DOC_FREQ_FILTER                    = "0.25"
$env:MENTIONED_ENTITIES_PER_BLOCK            = "5"
# Ranker bonuses (each individually capped)
$env:ENTITY_MATCH_BONUS                      = "0.18"
$env:ENTITY_CONFIDENCE_BONUS_WEIGHT          = "0.10"
$env:SAME_SECTION_BONUS                      = "0.08"
$env:SECTION_PATH_MATCH_BONUS                = "0.10"
$env:SECTION_STRUCTURAL_BONUS                = "0.05"
$env:GLOBAL_SIMILARITY_BONUS_WEIGHT          = "0.20"
$env:RELATIONSHIP_CONFIDENCE_BONUS_WEIGHT    = "0.10"
# Feature flags (set to "false" to disable individual retrieval arms)
$env:ENABLE_ENTITY_RETRIEVER         = "true"
$env:ENABLE_ENTITY_EXPANSION         = "true"
$env:ENABLE_SECTION_EXPANSION        = "true"
$env:ENABLE_GLOBAL_SIMILARITY_EXPANSION = "true"
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
│   ├── entity_retriever.py     Entity-based seeds via :Entity MENTIONS edges (3-tier matching)
│   ├── graph_expander.py       Three-arm expansion: relationship-typed, entity-mediated, section-aware
│   └── hybrid_retriever.py     Merge, dedupe, rank, and build evidence bundle
├── tests/
│   ├── fakes.py                Shared FakeNeo4j with new-schema row shapes
│   ├── test_neo4j_client_cypher.py  Cypher-shape guard (catches schema regressions)
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

**Graph expansion is bounded.** The expander walks at most `GRAPH_EXPANSION_LIMIT` seeds and applies three independently feature-flagged arms: relationship-typed (existing KG edges), entity-mediated (`MENTIONS`-bridged), and section-aware (sibling blocks in the same `:Section`). Per-arm row limits are hard-coded. There is no variable-length Cypher traversal.

---

## Graph schema (built by the KG notebook)

```
(:Document {doc_id, filename, num_pages})
(:Page {page_id, page_number})
(:Block {block_id, type, text, page_number, reading_order, embedding})
(:Section {section_id, title, path, level, page_start, page_end, block_count, heading_block_id})
(:Entity {entity_id, canonical_name, normalized_name, aliases, type, confidence, doc_frequency_ratio, doc_id})

(:Page)-[:PART_OF]->(:Document)
(:Block)-[:ON_PAGE]->(:Page)
(:Block)-[:PRECEDES]->(:Block)                         reading order chain
(:Block)-[:DESCRIBES]->(:Block)                        caption -> table/figure
(:Block)-[:INTRODUCES]->(:Block)                       heading -> section content
(:Block)-[:IN_SECTION]->(:Section)                     block -> deepest parent :Section node
(:Block)-[:CONTEXT_BEFORE]->(:Block)                   N blocks before a table
(:Block)-[:CONTEXT_AFTER]->(:Block)                    table -> N blocks after it
(:Block)-[:REFERS_TO {methods: list[str], confidence, scope, mention}]->(:Block)
(:Block)-[:SEMANTICALLY_SIMILAR {score, scope, methods}]-(:Block)  undirected; scope ∈ {"table","global"}
(:Block)-[:COMPARES|SUPPLEMENTS|CONTRASTS|ABLATES {reason}]->(:Block {type:'table'})
(:Block)-[:TABLE_RELATES_TO {label, reason}]->(:Block {type:'table'})  APOC fallback; label holds the logical type
(:Block)-[:MENTIONS {count, confidence, methods, spans_flat}]->(:Entity)
(:Document)-[:HAS_SECTION]->(:Section)
(:Section)-[:HAS_SUBSECTION]->(:Section)
(:Section)-[:STARTS_ON_PAGE]->(:Page)
```

**Notes:**
- `IN_SECTION` targets `:Section` nodes, not heading `:Block` nodes.
- `SEMANTICALLY_SIMILAR` is stored in canonical direction (src_id < tgt_id) so queries must use undirected `-[r:SEMANTICALLY_SIMILAR]-`.
- `TABLE_RELATES_TO` is used when APOC is not available; the logical relationship type is stored in `r.label` and normalized via `coalesce(r.label, type(r))` in queries.
- `REFERS_TO.methods` is always a list; never a single string.
- `:Entity` nodes with `type = 'TERM'` and high `doc_frequency_ratio` are filtered out by default (`TERM_DOC_FREQ_FILTER = 0.25`).
- `Block.type` values: `paragraph`, `table`, `caption`, `figure`, `formula`, `list_item`, `heading`.
- Vector index: `block_embedding_index` (1024-dim cosine on `Block.embedding`).
- Optional full-text index: `block_text_fulltext` (created lazily when `CREATE_FULLTEXT_INDEX=true`).

---

## Future: multi-document support

The architecture is designed for this. All retrieval queries already accept an optional `DOCUMENT_ID` to scope queries to a single document. Adding cross-document comparison will require:

- Multiple `Document` nodes in Neo4j (each PDF gets its own node).
- Cross-document relationship edges such as `SIMILAR_SECTION`, `REPORTS_SAME_METRIC`, and `CHANGED_FROM`.
- A document selector in the Chainlit session.
- Comparative prompts that cite sources from each document separately.
