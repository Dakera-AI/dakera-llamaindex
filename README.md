# llamaindex-dakera

**Persistent, decay-weighted agent memory and server-side vector indexing for LlamaIndex —
backed by a self-hosted [Dakera](https://dakera.ai) server.**

[![CI](https://github.com/Dakera-AI/dakera-llamaindex/actions/workflows/ci.yml/badge.svg)](https://github.com/Dakera-AI/dakera-llamaindex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![Downloads](https://img.shields.io/pypi/dm/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![Python](https://img.shields.io/pypi/pyversions/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/LICENSE)
[![Dakera server](https://img.shields.io/badge/Dakera%20server-v0.12.3-0ea5e9)](https://dakera.ai)
[![LoCoMo](https://img.shields.io/badge/LoCoMo-88.2%25-22c55e)](https://dakera.ai/benchmark)

Give LlamaIndex agents and chat engines a long-term memory that survives restarts: Dakera stores
what the user tells them and recalls the most relevant memories, ranked by importance, recency
and access, into every turn. The same server is a vector store that embeds your documents itself,
so neither memory nor RAG needs an embedding API key — and the whole backend is one self-hosted
binary.

## Why Dakera

- **Long-term memory for `Memory`** — `DakeraMemoryBlock` plugs into LlamaIndex `Memory`, so a
  `FunctionAgent` or chat engine remembers across processes, days and replicas.
- **Recall that ranks like memory should** — decay-weighted by importance, recency and access,
  with optional cross-encoder reranking, tag / importance filters and event time.
- **No embedding bill** — `DakeraIndexStore` sends text and the server embeds it (`bge-large`
  by default); set `Settings.embed_model = None` and no embedding API is called.
- **Semantic, hybrid and BM25 retrieval** with LlamaIndex metadata filters, in one store.
- **More than a vector store** — sessions, a memory knowledge graph, entity extraction and
  consolidation come with the same server.
- **Measured** — 88.2% Recall@20 on the [LoCoMo](https://dakera.ai/benchmark) long-conversation
  memory benchmark.

## Contents

- [Quick start](#quick-start)
- [Compatibility](#compatibility)
- [What's new in 0.4.0](#whats-new-in-040)
- [DakeraMemoryBlock](#dakeramemoryblock)
- [DakeraMemoryStore](#dakeramemorystore)
- [DakeraIndexStore](#dakeraindexstore)
- [DakeraSessionManager](#dakerasessionmanager)
- [DakeraKnowledgeGraph](#dakeraknowledgegraph)
- [DakeraEntityExtractor](#dakeraentityextractor)
- [DakeraNamespaceManager](#dakeranamespacemanager)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Related packages](#related-packages)
- [Links](#links)
- [License](#license)

## Quick start

**1. Run Dakera**

```bash
docker run -d --name dakera -p 3000:3000 \
  -v dakera-data:/data \
  -e DAKERA_ROOT_API_KEY=dk-mykey \
  -e DAKERA_STORAGE=filesystem \
  ghcr.io/dakera-ai/dakera:0.12.3

curl http://localhost:3000/health   # {"service":"dakera","status":"healthy","version":"0.12.3",...}
```

The image ships its default embedding model, so the server is ready in seconds; `dk-mykey` is
the root API key, and the `dakera-data` volume keeps the memories across container restarts.
Docker Compose, Kubernetes and Helm setups are in
[dakera-deploy](https://github.com/Dakera-AI/dakera-deploy).

**2. Install**

```bash
pip install llamaindex-dakera
```

**3. Remember across sessions**

```python
from llama_index_dakera import DakeraMemoryStore

store = DakeraMemoryStore(
    api_url="http://localhost:3000",
    api_key="dk-mykey",
    agent_id="my-agent",
)
store.put("The user's project is called NeuralBridge.")

# Later — another process, after a restart — the memory is still there:
for memory in store.get("What is the user's project?"):
    print(memory["score"], memory["content"])
# e.g. 0.77 The user's project is called NeuralBridge.
```

To give an agent this memory, add a [`DakeraMemoryBlock`](#dakeramemoryblock) to its `Memory`.

## Compatibility

| llamaindex-dakera | Dakera server | `dakera` SDK | LlamaIndex | Python |
|-------------------|---------------|--------------|------------|--------|
| **0.4.x** | v0.12.0 – **v0.12.3** | ≥ 0.14.1 | `llama-index-core` ≥ 0.12.35 (incl. 0.14) | ≥ 3.10 |
| 0.3.x | v0.12.0 | ≥ 0.13.1 | `llama-index-core` ≥ 0.10 | ≥ 3.10 |

Event time, the long-text report and `warnings` (marked *v0.12.3* below) need a v0.12.3 server;
older v0.12 servers ignore them.

## What's new in 0.4.0

- **Dakera server v0.12.3** support with `dakera>=0.14.1`.
- `DakeraMemoryBlock`: Dakera as long-term memory of LlamaIndex `Memory` and its agents.
- `DakeraIndexStore`: metadata filters, `hybrid` and `text_search` query modes, `doc_ids`,
  `delete(ref_doc_id)` that really deletes a document's nodes, `delete_nodes`, async writes,
  `embedding_model`, and no client-side embedding (`is_embedding_query = False`).
- Recalls return at most `top_k` ranked memories: the session-neighbourhood context a v0.12.3
  server can add is requested separately and left out, and a memory is never recalled twice
  next to the sentences the server derives from it.
- `put(event_time=...)` (*v0.12.3*), `get(rerank=...)` with `score` and `importance` per result,
  `consolidate(memory_ids)`.
- Sessions: `start(idle_timeout_secs=...)`, `touch()`, `end(summary=...)`.

Full details in the [CHANGELOG](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/CHANGELOG.md).

## DakeraMemoryBlock

A long-term memory block for LlamaIndex
[`Memory`](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/): on every
turn it recalls the agent's memories most relevant to the latest messages, and LlamaIndex puts
them into the system prompt.

```python
import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.llms.openai import OpenAI
from llama_index_dakera import DakeraMemoryBlock, DakeraMemoryStore

store = DakeraMemoryStore(api_url="http://localhost:3000", api_key="dk-mykey", agent_id="assistant")
memory = Memory.from_defaults(
    session_id="user-1",
    memory_blocks=[DakeraMemoryBlock(memory_store=store, similarity_top_k=5)],
)
agent = FunctionAgent(
    tools=[], llm=OpenAI(model="gpt-4o"), system_prompt="You are a helpful assistant."
)


async def main() -> None:
    store.put("The user's project is called NeuralBridge.", importance=0.9)
    response = await agent.run("What's the name of my project?", memory=memory)
    print(response)  # "Your project is called NeuralBridge."


asyncio.run(main())
```

Messages reach Dakera when `Memory` flushes them out of its short-term buffer (once the chat
history passes `token_limit × chat_history_token_ratio`); each user and assistant message is
then stored as one memory. Store a fact right away with `store.put()` or
`await block.aput([ChatMessage(...)])`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `memory_store` | `DakeraMemoryStore` | — | Store the block recalls from and writes to |
| `name` | `str` | `"DakeraMemory"` | Name of the block (the tag around its content in the prompt) |
| `similarity_top_k` | `int` | `5` | Memories recalled per turn |
| `retrieval_context_window` | `int` | `5` | Latest messages that form the recall query |
| `importance` | `float \| None` | store's `default_importance` | Importance of stored messages |
| `priority`, `accept_short_term_memory` | | LlamaIndex defaults | LlamaIndex `BaseMemoryBlock` options |

## DakeraMemoryStore

The agent's memories on the Dakera server: store, recall, search, update and delete.

```python
from datetime import datetime, timezone

stored = store.put(
    "The user hiked the Tour du Mont Blanc.",
    memory_type="semantic",
    importance=0.8,
    tags=["travel"],
    event_time=datetime(2025, 7, 14, tzinfo=timezone.utc),  # when it happened (v0.12.3)
)
memory_id = stored["id"]

store.get("Where has the user hiked?", top_k=3, tags=["travel"], rerank=True)
# [{"content", "id", "score", "importance", "metadata"}, ...] best first
store.hybrid_search("Mont Blanc", top_k=5)  # the server's vector + BM25 memory search
store.batch_get(["hiking", "work"], top_k=2)  # one result list per query
store.update_importance(memory_id, 0.95)
store.stats()  # {"total_memories", "memories_by_type", ...}
store.delete(memory_id)
```

- `put()` takes `memory_type` (`episodic`, `semantic`, `procedural`, `working`), `importance`,
  `metadata`, `tags`, `ttl_seconds`, `session_id` and `event_time` (Unix seconds, an RFC 3339
  string or a timezone-aware `datetime`: when the remembered event happened; recency and time
  filters use it instead of the time of storage). It returns the server's answer: the memory
  `id`, plus on a v0.12.3 server `truncated` / `embedded_tokens` when the text is longer than the
  embedding model reads, and `warnings` naming request fields the server ignored.
- `get()` results are ranked by the server's `rank_score` (`score`); `rerank=False` skips the
  cross-encoder reranking; `tags` keeps memories carrying at least one of them.
- `consolidate([id1, id2, ...], target_type="semantic")` merges memories into one new memory and
  deletes the sources. Memories with a TTL, an attachment or user metadata (including
  `event_time`) cannot be consolidated (`400`).
- `aput()` / `aget()` are the async variants; `batch_delete(ids)` deletes several memories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `agent_id` | `str` | — | Agent whose memories are stored and recalled |
| `api_key` | `str` | `""` | Dakera API key (empty when the server runs without auth) |
| `recall_k` | `int` | `5` | Memories `get()` returns by default |
| `min_importance` | `float` | `0.0` | Recall only memories at least this important |
| `default_importance` | `float` | `0.7` | Importance (0–1) of stored memories |

## DakeraIndexStore

A LlamaIndex vector store over a Dakera namespace. The server embeds the nodes (`bge-large` by
default) and indexes them for BM25 too, so the store ignores LlamaIndex embeddings — set
`Settings.embed_model = None` and no embedding API is called.

### Indexing and querying

```python
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index_dakera import DakeraIndexStore

Settings.embed_model = None  # Dakera embeds on the server

documents = SimpleDirectoryReader("./docs").load_data()
vector_store = DakeraIndexStore(
    api_url="http://localhost:3000",
    api_key="dk-mykey",
    namespace="product-docs",
)
index = VectorStoreIndex.from_documents(
    documents, storage_context=StorageContext.from_defaults(vector_store=vector_store)
)

# Needs an LLM: Settings.llm (OpenAI by default, OPENAI_API_KEY)
query_engine = index.as_query_engine(similarity_top_k=4)
print(query_engine.query("How does the billing work?"))
```

Reopen an existing namespace without re-indexing:

```python
index = VectorStoreIndex.from_vector_store(vector_store)
```

### Retrieval modes and filters

```python
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

index.as_retriever(similarity_top_k=4).retrieve("billing")  # semantic
index.as_retriever(similarity_top_k=4, vector_store_query_mode="hybrid", alpha=0.5).retrieve(
    "invoice PDF"
)
index.as_retriever(similarity_top_k=4, vector_store_query_mode="text_search").retrieve(
    "invoice PDF"
)

recent = MetadataFilters(
    filters=[MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE)]
)
index.as_retriever(similarity_top_k=4, filters=recent).retrieve("billing")

index.delete_ref_doc(documents[0].doc_id)  # every node of that document
```

- `hybrid` takes the `max(4·k, 20)` (or `hybrid_top_k`) semantically nearest nodes matching the
  filters and scores each `alpha · cosine + (1 − alpha) · bm25 / best_bm25` (`alpha` default
  0.5); `text_search` ranks those candidates by BM25.
- Filter operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `nin`, `text_match`,
  `text_match_insensitive`, `is_empty`, nested with `and` / `or` / `not`. `doc_ids` restricts a
  query to given documents.
- `delete_nodes(node_ids=..., filters=...)` deletes nodes; `async_add` / `adelete` are async.

### Chat with your documents and remember the user

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import Memory
from llama_index_dakera import DakeraMemoryBlock, DakeraMemoryStore

memory = Memory.from_defaults(
    session_id="user-1",
    memory_blocks=[
        DakeraMemoryBlock(
            memory_store=DakeraMemoryStore(
                api_url="http://localhost:3000", api_key="dk-mykey", agent_id="doc-chat"
            )
        )
    ],
)
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=index.as_retriever(similarity_top_k=4),
    memory=memory,
)
print(chat_engine.chat("What are the pricing tiers?"))
```

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `namespace` | `str` | — | Namespace to read/write (created on the first write) |
| `api_key` | `str` | `""` | Dakera API key |
| `embedding_model` | `str \| None` | server default | Server-side embedding model, e.g. `"bge-large"`, `"minilm"`; use one model per namespace |

## DakeraSessionManager

Groups the memories of one conversation. The server ends a session after 4 h without activity
by default (`ended_reason: "idle"`).

```python
from llama_index_dakera import DakeraSessionManager

sessions = DakeraSessionManager(
    api_url="http://localhost:3000", api_key="dk-mykey", agent_id="assistant"
)
session_id = sessions.start(metadata={"topic": "onboarding"}, idle_timeout_secs=1800)
store.put("The user asked how to reset their password.", session_id=session_id)

sessions.touch()  # record activity: "active" (or "ended")
sessions.memories(session_id)  # the session's memories, oldest first
sessions.end(summary="Password reset help")
sessions.get(session_id)  # {"id", "summary", "ended_reason", "memory_count", ...}
sessions.list(active_only=False)

with DakeraSessionManager(
    api_url="http://localhost:3000", api_key="dk-mykey", agent_id="assistant"
):
    ...  # started on entry, ended on exit
```

| Method | Description |
|--------|-------------|
| `start(metadata=None, *, idle_timeout_secs=None)` | Start a session and make it the active one; `idle_timeout_secs=0` never ends it for inactivity |
| `touch(session_id=None)` | Record activity; returns `"active"` or `"ended"` |
| `end(summary=None)` | End the active session, optionally with a summary |
| `get(session_id)` / `list(active_only=False)` | Session details: `summary`, `ended_reason`, `memory_count`, ... |
| `memories(session_id, *, include_derived=False)` | The session's memories, without the server's derived sentence sub-memories unless asked |

Constructor: `api_url`, `agent_id`, `api_key=""`.

## DakeraKnowledgeGraph

Links between an agent's memories: explicit ones you add and the ones the server infers.

```python
from llama_index_dakera import DakeraKnowledgeGraph

# memory_a, memory_b, memory_c: ids of the agent's memories (e.g. store.put(...)["id"])
kg = DakeraKnowledgeGraph(api_url="http://localhost:3000", api_key="dk-mykey", agent_id="assistant")
kg.link(memory_a, memory_b, label="same project")  # explicit link between two memories
kg.link(memory_b, memory_c)
kg.query(root_id=memory_a, max_depth=2)  # {"edges": [{"source", "target", "type"}], ...}
kg.find_path(memory_a, memory_c)  # {"path": [a, b, c], "hop_count": 2}
kg.build(memory_a, depth=2)  # the graph around one memory
kg.export()  # every edge of the agent
kg.summarize([memory_a, memory_b])  # stores a new summary memory
kg.deduplicate()  # merge the agent's near-duplicate memories
```

Constructor: `api_url`, `agent_id`, `api_key=""`.

## DakeraEntityExtractor

Named-entity extraction from text and from the agent's stored memories.

```python
from llama_index_dakera import DakeraEntityExtractor

entities = DakeraEntityExtractor(
    api_url="http://localhost:3000", api_key="dk-mykey", agent_id="assistant"
)
entities.extract(
    "Sarah Chen joined Acme in Paris.", entity_types=["person", "organization", "location"]
)  # [{"type", "value", "score"}, ...]
entities.configure(["person", "organization"])  # extract entities from the agent's new memories
entities.memory_entities(memory_a)
```

Named entity types load the server's GLiNER model on first use (a ~782 MB download from Hugging
Face); a server that cannot load it serves its rule-based extractor only. `configure()` applies to
the agent's memory namespace, `_dakera_agent_<agent_id>`.

Constructor: `api_url`, `agent_id`, `api_key=""`.

## DakeraNamespaceManager

Create and inspect namespaces (vector collections) for your own data.

```python
from llama_index_dakera import DakeraNamespaceManager

namespaces = DakeraNamespaceManager(api_url="http://localhost:3000", api_key="dk-mykey")
namespaces.create("embeddings-768", dimensions=768)  # dimensions is required by the server
namespaces.get("embeddings-768")
namespaces.list()
namespaces.stats("product-docs")
namespaces.delete("embeddings-768")  # deletes the namespace and all its data
```

`DakeraIndexStore` creates its namespace on the first write, so you only need this class for
namespaces with a fixed dimension or distance.

Constructor: `api_url`, `api_key=""`.

## Configuration

Every class takes the server URL and API key as arguments — the package reads no environment
variables itself. The examples, tests and the snippets in this README use:

| Variable | Used for | Example |
|----------|----------|---------|
| `DAKERA_API_URL` | Dakera server URL | `http://localhost:3000` |
| `DAKERA_API_KEY` | API key (leave unset for a server running without auth) | `dk-mykey` |

```python
import os

from llama_index_dakera import DakeraMemoryStore

store = DakeraMemoryStore(
    api_url=os.environ["DAKERA_API_URL"],
    api_key=os.environ.get("DAKERA_API_KEY", ""),
    agent_id="my-agent",
)
```

**Auth.** `DAKERA_ROOT_API_KEY` sets the server's root key. For applications, create scoped keys
(`read`, `write` or `admin`, optionally limited to namespaces) with the
[`dakera`](https://pypi.org/project/dakera/) SDK (`DakeraClient.create_key()`). An app that
stores and recalls memories needs a `write` key (`write` includes `read`).

## Examples

Every example runs against a live server and needs no LLM API key:

| Example | Shows | LLM key? |
|---------|-------|----------|
| [`basic_memory.py`](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/examples/basic_memory.py) | `put` / `get` | No |
| [`memory_block.py`](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/examples/memory_block.py) | `DakeraMemoryBlock` in LlamaIndex `Memory` | No |
| [`advanced_memory.py`](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/examples/advanced_memory.py) | importance, tags, types, TTL, `event_time`, filtered recall, hybrid search, consolidate | No |
| [`batch_operations.py`](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/examples/batch_operations.py) | `batch_get`, `batch_delete` | No |
| [`sessions.py`](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/examples/sessions.py) | sessions: idle timeout, `touch`, summary, context manager | No |
| [`knowledge_graph.py`](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/examples/knowledge_graph.py) | link, query, path, build, export, summarize | No |
| [`rag_pipeline.py`](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/examples/rag_pipeline.py) | `VectorStoreIndex` on Dakera, retrieval, RAG prompt, `delete_ref_doc` | No |
| [`hybrid_search.py`](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/examples/hybrid_search.py) | semantic, hybrid (`alpha`), BM25 and filtered retrieval | No |

```bash
git clone https://github.com/Dakera-AI/dakera-llamaindex && cd dakera-llamaindex
pip install -e .
export DAKERA_API_URL=http://localhost:3000
export DAKERA_API_KEY=dk-mykey   # leave unset if the server runs without auth
python examples/basic_memory.py
```

## Troubleshooting

**`ConnectionError` / connection refused.** The server listens on port 3000 (`curl
http://localhost:3000/health`). This repository's CI maps it to 3300 — use whichever port your
`docker run -p` published.

**`AuthenticationError` (401).** A server started with `DAKERA_ROOT_API_KEY` requires a key on
every request: pass `api_key=` (the root key or a scoped key). A `403` means the key lacks the
scope or namespace.

**`NotFoundError` (404 `NAMESPACE_NOT_FOUND`) on the first recall.** Servers before v0.12.3
answer a recall for an agent that has stored nothing with a 404; v0.12.3 returns no memories.
Upgrade the server, or store a memory (`store.put(...)`) before the agent's first turn.

**Fewer or more results than expected.** Recalls return at most `top_k` ranked memories. A
v0.12.3 server can add session-neighbourhood context; this package requests it separately and
leaves it out, so the prompt holds ranked memories only.

**The agent does not remember the last turns.** `Memory` hands messages to
`DakeraMemoryBlock` only when it flushes them out of its short-term buffer; until then they are
in the chat history, not in Dakera. Store important facts at once with `store.put()`.

**Indexing asks for `llama-index-embeddings-openai` or `OPENAI_API_KEY`.** LlamaIndex's default
embedding model is OpenAI's. Set `Settings.embed_model = None`: `DakeraIndexStore` embeds on the
server and ignores client embeddings.

**`event_time`, `warnings` or `truncated` missing.** These are v0.12.3 features; older servers
ignore the field or leave it out of the answer. Check the version with `curl
http://localhost:3000/health`.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy src/
pytest tests/                                          # unit tests, no server needed
DAKERA_API_URL=http://localhost:3000 pytest tests/     # + live tests against a server
```

See [CONTRIBUTING.md](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/CONTRIBUTING.md)
for the full workflow.

## Related packages

| Package | What it is |
|---|---|
| [`dakera`](https://pypi.org/project/dakera/) · [`@dakera-ai/dakera`](https://www.npmjs.com/package/@dakera-ai/dakera) · [`dakera-client`](https://crates.io/crates/dakera-client) · [`dakera-go`](https://github.com/Dakera-AI/dakera-go) | Python, TypeScript, Rust and Go SDKs |
| [`langchain-dakera`](https://pypi.org/project/langchain-dakera/) | LangChain (Python): memory, vector store, sessions, knowledge graph |
| [`@dakera-ai/langchain`](https://www.npmjs.com/package/@dakera-ai/langchain) | LangChain.js: memory, vector store, sessions, knowledge graph |
| **[`llamaindex-dakera`](https://pypi.org/project/llamaindex-dakera/)** | **LlamaIndex: memory block, vector store, memory store** |
| [`autogen-dakera`](https://pypi.org/project/autogen-dakera/) | AutoGen: `Memory` for `AssistantAgent`, sessions, knowledge graph |
| [`crewai-dakera`](https://pypi.org/project/crewai-dakera/) | CrewAI: storage for a crew's memories |
| [`strands-dakera`](https://pypi.org/project/strands-dakera/) | Strands Agents: memory tool and `MemoryStore` |
| [`@dakera-ai/ai-sdk`](https://www.npmjs.com/package/@dakera-ai/ai-sdk) | Vercel AI SDK: memory middleware and tools |

## Links

- [dakera.ai](https://dakera.ai) · [Documentation](https://dakera.ai/docs) ·
  [LlamaIndex integration guide](https://dakera.ai/integrations/llamaindex.html) ·
  [LoCoMo benchmark](https://dakera.ai/benchmark)
- Server image: `ghcr.io/dakera-ai/dakera` ·
  [dakera-deploy](https://github.com/Dakera-AI/dakera-deploy) (Docker Compose, Kubernetes) ·
  [dakera-helm](https://github.com/Dakera-AI/dakera-helm)
- [dakera-mcp](https://github.com/Dakera-AI/dakera-mcp) (MCP server) ·
  [dakera-cli](https://github.com/Dakera-AI/dakera-cli) (`dk` CLI)
- [Issues](https://github.com/Dakera-AI/dakera-llamaindex/issues)

## License

[MIT](https://github.com/Dakera-AI/dakera-llamaindex/blob/main/LICENSE) © [Dakera AI](https://dakera.ai)
