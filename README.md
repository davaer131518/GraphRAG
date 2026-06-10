# Traceable PDF Analyst

A local Graph-RAG chat system that answers questions about a parsed PDF document using a Neo4j knowledge graph. Every answer includes the source blocks, page numbers, section headings, relationship paths used during retrieval, and a step-by-step evidence trace.

The user interface is built with Chainlit. All retrieval, graph expansion, evidence bundling, prompting, and answer parsing happen in backend modules so the same analyst can be used outside the UI.

---

## How it works

The system does not do plain text search over a document. It reads from a pre-built Neo4j knowledge graph where every block of text from a parsed PDF is a node (`Block`) connected to other blocks through typed relationships that encode reading order, section structure, caption connections, semantic similarity, and LLM-labelled table relationships.

When you ask a question the pipeline runs in three layers, optionally preceded by a document pre-retrieval step:

### Pre-step — Document pre-retrieval (optional)

When the scope is still whole-corpus and `ENABLE_DOCUMENT_PRE_RETRIEVAL=true`, the question is embedded with bge-m3 and compared against per-document centroid embeddings stored in `Document.embedding`. The top-N most relevant documents (default `DOCUMENT_PRE_RETRIEVAL_TOP_N=5`) are selected, and all subsequent block retrieval is restricted to those documents. This prevents cross-corpus dilution in large corpora — financial questions from one set of filings won't be drowned out by topically similar but wrong documents.

If the `document_embedding_index` is absent (graphs built before KnowledgeGraphBuilder added centroid storage), this step is silently skipped and the pipeline proceeds exactly as if the flag were off.

### Layer 1 — Seed retrieval

Four retrievers run and their results are merged and deduplicated on `block_id`:

- **Vector search**: the question is embedded with the same bge-m3 model used to embed the blocks, and Neo4j's native ANN vector index returns the top-K semantically similar blocks.
- **Keyword search**: key phrases and terms are extracted from the question and matched against `Block.text`, preferring a full-text index if available and falling back to a `CONTAINS` query.
- **Entity search**: extracted terms are matched against `:Entity` nodes using a 3-tier strategy — exact canonical name match (score 1.0), alias match (score 0.9), and partial substring match (score 0.6, min 4 chars). Each matched block carries the entity name, type, and match score for use in ranking. High-frequency generic terms (`TERM` entities above `TERM_DOC_FREQ_FILTER`) are excluded. Can be disabled with `ENABLE_ENTITY_RETRIEVER=false`.
- **Table keyword search**: searches only `type='table'` blocks for question terms, separate from the prose search. This ensures structured data tables (income statements, segment breakdowns, etc.) are seeded directly and not crowded out by risk-factor or discussion paragraphs that mention the same terms in passing. Uses the full-text index when available, falls back to `CONTAINS`. Gated by `ENABLE_TABLE_SEED_SEARCH=true`.

### Layer 2 — Graph expansion

Each seed block is expanded through the KG using up to six arms. The first three are same-document; the last three are cross-document and only fire when the scope spans more than one document.

**Arm 1 — Relationship-typed** (dispatched by `Block.type`):
- **Paragraph / caption / list item seeds**: follow `REFERS_TO`, `SEMANTICALLY_SIMILAR` (`scope='table'` and `scope='global'`), and a local `PRECEDES` window.
- **Table seeds**: follow `CONTEXT_BEFORE` / `CONTEXT_AFTER`, incoming `REFERS_TO` / `DESCRIBES`, and table-pair edges `COMPARES` / `SUPPLEMENTS` / `CONTRASTS` / `ABLATES` / `TABLE_RELATES_TO`.
- **Heading seeds**: follow `INTRODUCES` to the blocks introduced by that heading.

**Arm 2 — Entity-mediated** (`ENABLE_ENTITY_EXPANSION=true`): `seed -[:MENTIONS]-> Entity <-[:MENTIONS]- related`. Bounded by `ENTITY_EXPANSION_ENTITIES_PER_SEED` × `ENTITY_EXPANSION_BLOCKS_PER_ENTITY`.

**Arm 3 — Section-aware** (`ENABLE_SECTION_EXPANSION=true`): sibling blocks in the same `:Section`, ordered by reading-order proximity. Bounded by `SECTION_EXPANSION_LIMIT`.

**Arm 4 — Cross-doc canonical entity** (`ENABLE_CROSS_DOC_ENTITY_EXPANSION=true`): `seed -[:MENTIONS]-> Entity -[:RESOLVES_TO]-> CanonicalEntity <-[:RESOLVES_TO]- Entity2 <-[:MENTIONS]- relatedBlock` — bridges to blocks in other documents that discuss the same real-world entity. Bounded by `CROSS_DOC_ENTITY_ENTITIES_PER_SEED` × `CROSS_DOC_ENTITY_BLOCKS_PER_ENTITY`. Only fires when >1 document is in scope.

**Arm 5 — Cross-doc section** (`ENABLE_CROSS_DOC_SECTION_EXPANSION=true`): follows accepted `SIMILAR_SECTION` edges (undirected) to the analogous section in another document and returns its blocks. Bounded by `CROSS_DOC_SIMILAR_SECTIONS_PER_SEED` × `CROSS_DOC_BLOCKS_PER_SIMILAR_SECTION`.

**Arm 6 — Cross-doc table** (`ENABLE_CROSS_DOC_TABLE_EXPANSION=true`, table seeds only): follows accepted `SCHEMA_MATCH` / `REPORTS_SAME_METRIC` edges (undirected) to matching tables in other documents. Bounded by `CROSS_DOC_TABLE_LIMIT`.

All expansion is bounded and explicit. No variable-length Cypher traversal anywhere.

### Layer 3 — Answer generation

The expanded evidence is ranked by a multi-signal score:

- **Retrieval method bonuses**: vector (0.32), keyword / table_keyword (0.30), expansion (0.12), cross-doc arms (0.08)
- **Relationship type weights**: `REFERS_TO` (0.28), `DESCRIBES` (0.24), table-pair rels (0.18–0.22), `REPORTS_SAME_METRIC` (0.20), `MENTIONS_SHARED` (0.14), `SCHEMA_MATCH` (0.16), `SIMILAR_SECTION` (0.12), `MENTIONS_SHARED_CANONICAL` (0.12), `SAME_SECTION` (0.10), etc. Cross-doc weights are below their same-doc analogues so cross-doc evidence augments rather than dominates.
- **Per-document diversity cap**: a secondary document (not the highest-ranked item's doc) can occupy at most `CROSS_DOC_PER_DOC_CAP` (default 3) final slots, preventing one related document from flooding the answer. No-op at single-document scope.
- **Block type weight**, **vector score**, **exact phrase overlap**
- **Entity bonuses**: `entity_match_bonus` scaled by match score; `entity_confidence_bonus` scaled by `MENTIONS.confidence`
- **Section bonuses**: `same_section_bonus`, `section_path_match_bonus`, `section_structural_bonus`
- **Table-in-section bonus** (`TABLE_IN_SECTION_BONUS`, default 0.25): table blocks in the same section as a seed get a substantial boost, ensuring financial/data tables are included alongside the paragraphs that introduce them.
- **Graph bonuses**: `global_similarity_bonus` (capped at 0.20), `relationship_confidence_bonus` (capped at 0.10)

When the final evidence spans more than one document, the LLM is given a **comparative system prompt** that instructs it to attribute each claim to its source document and explicitly compare/contrast across documents. Single-document answers use the original system prompt unchanged.

```
Question
  -> embed question          (bge-m3 on EMBED_SERVER_URL)
  -> document pre-retrieval  (Neo4j document_embedding_index; narrows scope if whole-corpus)
  -> vector search           (Neo4j block_embedding_index, within narrowed scope)
  -> keyword search          (Neo4j full-text or CONTAINS)
  -> entity search           (Neo4j :Entity via MENTIONS, 3-tier matching)
  -> table keyword search    (Neo4j full-text or CONTAINS, type='table' only)
  -> merge + dedupe seeds
  -> graph expansion         (relationship-typed + entity-mediated + section-aware)
  -> rank + dedupe
  -> build evidence bundle
  -> LLM answer              (Qwen3.5-9B on LLM_SERVER_URL)
  -> render in Chainlit
```

---

## Prerequisites

| Component | Required |
|---|---|
| Neo4j 5.11+ with the KG already loaded | Required |
| `block_embedding_index` vector index in Neo4j | Required |
| llama.cpp embedding server running bge-m3 | Required for every question |
| llama.cpp LLM server running Qwen3.5-9B (or compatible) | Required for every question |

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
$env:TABLE_TOP_K                   = "5"
$env:GRAPH_EXPANSION_LIMIT         = "20"
$env:FINAL_EVIDENCE_LIMIT          = "20"
$env:SEMANTIC_SIMILARITY_THRESHOLD = "0.50"
$env:LOG_LEVEL                     = "INFO"
$env:CREATE_FULLTEXT_INDEX         = "false"
# LLM generation
$env:LLM_MAX_TOKENS                = "1024"
$env:LLM_TEMPERATURE               = "0.0"
$env:PROMPT_EVIDENCE_MAX_CHARS     = "1000"
# Embedding and request tuning
$env:EMBED_MAX_CHARS               = "6000"
$env:REQUEST_TIMEOUT_SECONDS       = "120"
# Keyword ranking boosts
$env:KEYWORD_TERM_BOOST            = "0.05"
# llama.cpp server ports and context sizes (used with AUTO_START_SERVERS)
$env:EMBED_SERVER_PORT             = "8091"
$env:EMBED_N_CTX                   = "8192"
$env:LLM_SERVER_PORT               = "8092"
$env:LLM_N_CTX                     = "32768"
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
$env:TABLE_IN_SECTION_BONUS                  = "0.25"
$env:GLOBAL_SIMILARITY_BONUS_WEIGHT          = "0.20"
$env:RELATIONSHIP_CONFIDENCE_BONUS_WEIGHT    = "0.10"
# Feature flags (set to "false" to disable individual retrieval arms)
$env:ENABLE_TABLE_SEED_SEARCH        = "true"
$env:ENABLE_ENTITY_RETRIEVER         = "true"
$env:ENABLE_ENTITY_EXPANSION         = "true"
$env:ENABLE_SECTION_EXPANSION        = "true"
$env:ENABLE_GLOBAL_SIMILARITY_EXPANSION = "true"
# Document pre-retrieval (narrows corpus to top-N most relevant docs before block search)
$env:ENABLE_DOCUMENT_PRE_RETRIEVAL  = "true"
$env:DOCUMENT_PRE_RETRIEVAL_TOP_N   = "5"
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
LLM_MODEL_PATH=C:\llama-cpp\models\Qwen3.5-9B-Q8_0.gguf
LLM_SERVER_PORT=8092
LLM_N_CTX=32768
LLAMA_HEALTH_TIMEOUT=120
```

If a server is already running on the expected port when the app starts, it is reused and will not be stopped on chat end.

If you prefer to manage the servers manually, set `AUTO_START_SERVERS=false` and start them yourself before running Chainlit:

```powershell
# Embedding server
C:\llama-cpp\llama-server.exe -m C:\llama-cpp\models\bge-m3-Q8_0.gguf `
  --port 8091 --host 127.0.0.1 -ngl -1 --embedding --pooling mean -c 8192

# LLM server
C:\llama-cpp\llama-server.exe -m C:\llama-cpp\models\Qwen3.5-9B-Q8_0.gguf `
  --port 8092 --host 127.0.0.1 -ngl -1 -c 32768
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

### Document scope

The analyst automatically scopes each question to the relevant document(s) — you don't need to do anything for the common case. The scope is visible above each answer as `> Scope: …`.

Explicit scope commands:

| Command | What it does |
|---|---|
| `/docs` | List all documents in the corpus (filename, doc_id, pages, family/version). |
| `/use <doc_id>` | Pin the session to a single document. |
| `/scope <doc_id>[,<doc_id>…]` | Pin the session to a set of documents (for comparisons). |
| `/scope all` | Release the pin; return to query-driven automatic scoping. |
| `/related [doc_id]` | Show RELATED_DOCUMENT neighbors with their evidence summary and a comparison tip. |

### Commands

| Command | What it does |
|---|---|
| `/table <block_id>` | Show same-doc table relationships (`COMPARES`, `SUPPLEMENTS`, `CONTRASTS`, `ABLATES`). When multi-doc scope is active, cross-document `SCHEMA_MATCH` / `REPORTS_SAME_METRIC` links are also shown. |
| `/table <block_id> <RELATION>` | Filter to one same-doc relationship type. |
| `/map` | Generate a structured Markdown document map. Shows one section per in-scope document with a per-doc header when multiple documents are in scope. |
| `/debug last` | Show the full evidence bundle (including `scope_rationale`, per-item `doc_id`, cross-doc hops in `relationship_path`), per-block ranking scores, raw answer JSON, and trace. |

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
│   ├── prompts.py              System and user prompt templates (single-doc + comparative)
│   └── answer_generator.py     JSON answer parsing, validation, and fallback handling
├── retrievers/
│   ├── scope.py                RetrievalScope frozen dataclass — per-turn document scoping
│   ├── scope_resolver.py       Deterministic scope resolver (filename, doc key, year, family cues)
│   ├── semantic_retriever.py   Vector search seeds
│   ├── keyword_retriever.py    Keyword/full-text search seeds + table keyword search + term extraction
│   ├── entity_retriever.py     Entity-based seeds via :Entity MENTIONS edges (3-tier matching)
│   ├── graph_expander.py       Six-arm expansion: 3 same-doc + 3 cross-doc
│   └── hybrid_retriever.py     Merge, dedupe, rank, and build evidence bundle
├── tests/
│   ├── fakes.py                Shared FakeNeo4j with full schema row shapes
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

**Server lifecycle is handled by `server_manager.py`.** The `LlamaServer` class wraps a single `llama-server.exe` process; `ServerManager` coordinates the embedding/LLM pair. If `AUTO_START_SERVERS=true`, the Chainlit `on_chat_start` hook starts both servers and `on_chat_end` stops only the ones it started (servers that were already running are left untouched). `LLM_N_CTX` must match what the server was launched with — a change requires a full server restart.

**Graph expansion is bounded.** The expander walks at most `GRAPH_EXPANSION_LIMIT` seeds and applies six independently feature-flagged arms. The first three are same-document (relationship-typed, entity-mediated, section-aware); the last three are cross-document and are inert at corpus size 1. Per-arm row limits are hard-coded. There is no variable-length Cypher traversal.

**Table retrieval uses two complementary mechanisms.** The table keyword search seeds table blocks directly from question terms (bypassing prose competition). The `table_in_section_bonus` promotes table blocks that land in the same section as any prose seed (catching intro-paragraph → table pairs). Together these ensure financial and structured data tables surface reliably for numeric/data questions.

**Table blocks follow two separate rendering paths.** The LLM and the Chainlit UI receive different representations of the same `Block.table_html` property:
- **LLM prompt** (`_html_table_to_llm_text` in `generation/prompts.py`): HTML is converted to a labeled-row text format (`Table columns: 2023 | 2022 | 2021` / `iPhone: 200583 | ...`). Thousands-separator commas are stripped from values (so `200,583` becomes `200583` — unambiguous for a model that might read commas as decimal points) and footnote markers are stripped from row labels. Table blocks are never merged into prose list groups regardless of their text length.
- **Source cards in Chainlit** (`_html_table_to_markdown` in `evidence/trace_formatter.py`): HTML is converted to a Markdown pipe table, which Chainlit renders visually. Numbers keep their commas for human readability. Falls back to the pipe-delimited `Block.text` snippet when `table_html` is absent.

---

## Graph schema (built by KnowledgeGraphBuilder)

### Per-document nodes and edges

```
(:Document {doc_id, filename, num_pages, corpus_id, logical_doc_key, doc_family, version_id, published_at, embedding, language, source_uri, ingested_at})
                                                     ↑ L2-normalised centroid of all block embeddings; indexed by document_embedding_index
(:Page {page_id, page_number})
(:Block {block_id, type, text, page_number, reading_order, embedding, table_html})
                                table_html is only present when type='table'; used for accurate LLM table reading and Chainlit rendering
(:Section {section_id, title, path, level, page_start, page_end, block_count, heading_block_id, doc_id})
(:Entity {entity_id, canonical_name, normalized_name, aliases, type, confidence, doc_frequency_ratio, doc_id})

(:Page)-[:PART_OF]->(:Document)
(:Block)-[:ON_PAGE]->(:Page)
(:Block)-[:PRECEDES]->(:Block)                          reading order chain
(:Block)-[:DESCRIBES]->(:Block)                         caption -> table/figure
(:Block)-[:INTRODUCES]->(:Block)                        heading -> section content
(:Block)-[:IN_SECTION]->(:Section)                      block -> deepest parent :Section node (NOT a heading Block)
(:Block)-[:CONTEXT_BEFORE|CONTEXT_AFTER]->(:Block)
(:Block)-[:REFERS_TO {methods: list[str], confidence, scope, mention}]->(:Block)
(:Block)-[:SEMANTICALLY_SIMILAR {score, scope, methods}]-(:Block)   undirected; scope ∈ {"table","global"}
(:Block)-[:COMPARES|SUPPLEMENTS|CONTRASTS|ABLATES {reason}]->(:Block {type:'table'})
(:Block)-[:TABLE_RELATES_TO {label, reason}]->(:Block {type:'table'})   APOC fallback; label = logical type
(:Block)-[:MENTIONS {count, confidence, methods, spans_flat}]->(:Entity)
(:Document)-[:HAS_SECTION]->(:Section)
(:Section)-[:HAS_SUBSECTION]->(:Section)
(:Section)-[:STARTS_ON_PAGE]->(:Page)
```

### Cross-document corpus layer

All cross-doc edges are **canonically ordered** (`min(id) → max(id)`) — always query them **undirected (no arrow)**. Only `decision='accepted'` edges are written. `methods` is always `list[str]`.

```
(:CanonicalEntity {canonical_id, corpus_id, type, display_name, normalized_name, aliases, cluster_size})

(:Entity)-[:RESOLVES_TO {score, methods, decision, run_id, source_doc_id}]->(:CanonicalEntity)
   (directed: Entity -> CanonicalEntity; constrain to one corpus_id)

(:Section)-[:SIMILAR_SECTION {score, methods, decision, run_id, source_doc_id, target_doc_id, ...}]-(:Section)
   (undirected; cross-document; filter decision='accepted')

(:Block{table})-[:SCHEMA_MATCH {score, schema_score, metric_score, methods, decision, ...}]-(:Block{table})
   (undirected; cross-document; filter decision='accepted')

(:Block{table})-[:REPORTS_SAME_METRIC {score, schema_score, metric_score, methods, decision, ...}]-(:Block{table})
   (undirected; only exists alongside a SCHEMA_MATCH for the same pair)

(:Document)-[:RELATED_DOCUMENT {score, methods, decision, similar_section_count, schema_match_count,
                                  reports_same_metric_count, shared_canonical_entity_count,
                                  high_value_shared_canonical_entity_count, evidence_summary, ...}]-(:Document)
   (undirected; conservative precomputed aggregate)
```

**Critical schema notes:**
- `CanonicalEntity` uses `display_name` (not `canonical_name`) and `cluster_size` (not `doc_count`).
- `IN_SECTION` targets `:Section` nodes, **not** heading `:Block` nodes.
- `SEMANTICALLY_SIMILAR` is stored canonically (src_id < tgt_id) — always query undirected.
- `REFERS_TO.methods` is always a list — never a single string.
- `TABLE_RELATES_TO` appears when APOC is absent; use `coalesce(r.label, type(r))` to normalise.
- `:Entity` nodes with `type='TERM'` and high `doc_frequency_ratio` are filtered by `TERM_DOC_FREQ_FILTER`.
- Vector index: `block_embedding_index` (1024-dim cosine on `Block.embedding`).
- Vector index: `document_embedding_index` (1024-dim cosine on `Document.embedding`). Present on graphs built by a recent KnowledgeGraphBuilder; absent on older graphs — the pipeline degrades gracefully when it is missing.
- Optional full-text index: `block_text_fulltext` (created lazily when `CREATE_FULLTEXT_INDEX=true`).

---

## Multi-document support

The analyst supports multi-document corpora natively. Scope is determined automatically on each turn:

1. **Sticky session scope** (highest priority): set by `/use <doc_id>` or `/scope <ids>`, seeded from `DOCUMENT_ID` env var at startup.
2. **Query-driven scope**: a deterministic resolver maps explicit document/version references in the question (filename, `logical_doc_key`, year, family) to `doc_id`s. Ambiguous references fall back to the whole corpus rather than guessing.
3. **Document pre-retrieval** (`ENABLE_DOCUMENT_PRE_RETRIEVAL=true`): when scope is still unresolved after step 2, the question embedding is compared against per-document centroid embeddings (`Document.embedding`) to select the top-N most relevant documents. Block retrieval then operates within that narrowed scope, preventing dilution in large corpora. Requires the `document_embedding_index` in Neo4j; degrades gracefully (no-op) on older graphs that don't have it.
4. **Whole-corpus default**: when no reference is found and pre-retrieval is off or unavailable, retrieval covers all documents in the corpus and relevance ranking surfaces the answer.

Cross-doc retrieval arms (Arms 4/5/6) activate automatically when more than one document is in scope and are inert at corpus size 1.
