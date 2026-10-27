# llamaindex-dakera

[![CI](https://github.com/Dakera-AI/dakera-llamaindex/actions/workflows/ci.yml/badge.svg)](https://github.com/Dakera-AI/dakera-llamaindex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![Downloads](https://img.shields.io/pypi/dm/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![Python](https://img.shields.io/pypi/pyversions/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![dakera.ai](https://img.shields.io/badge/dakera.ai-website-22c55e?style=flat-square)](https://dakera.ai)
[![Docs](https://img.shields.io/badge/docs-dakera.ai%2Fdocs-3b82f6?style=flat-square)](https://dakera.ai/docs)

**LlamaIndex components backed by [Dakera](https://dakera.ai) — persistent agent memory and
server-side vector indexing with no local embedding model.**

- `DakeraMemoryStore` + `DakeraMemoryBlock` — long-term agent memory that survives restarts,
  plugged into LlamaIndex `Memory`: relevant memories are recalled into every turn
  (decay-weighted by importance, recency and access).
- `DakeraIndexStore` — a LlamaIndex vector store whose nodes are embedded by the Dakera server
  (no OpenAI / Hugging Face embedding calls), with semantic, hybrid and BM25 retrieval and
  metadata filters.
- `DakeraSessionManager`, `DakeraKnowledgeGraph`, `DakeraEntityExtractor`,
  `DakeraNamespaceManager` — sessions, the memory graph, entity extraction and namespaces.

## Compatibility

| llamaindex-dakera | Dakera server | `dakera` SDK | LlamaIndex |
|-------------------|---------------|--------------|------------|
| **0.4.x** | v0.12.0 – **v0.12.3** | ≥ 0.14.1 | `llama-index-core` ≥ 0.12.35 (incl. 0.14) |
| 0.3.x | v0.12.0 | ≥ 0.13.1 | `llama-index-core` ≥ 0.10 |

Features marked *v0.12.3* below (event time, the long-text report, `warnings`) need a v0.12.3
server; older v0.12 servers ignore them. Python ≥ 3.10.

---

## Quick start

### 1. Run a Dakera server

```bash
docker run -d --name dakera -p 3000:3000 \
  -v dakera-data:/data \
  -e DAKERA_ROOT_API_KEY=dk-mykey \
  -e DAKERA_STORAGE=filesystem \
  ghcr.io/dakera-ai/dakera:0.12.3

curl http://localhost:3000/health   # {"service":"dakera","status":"healthy",...}
```

The server listens on port 3000 and requires an API key (here the root key `dk-mykey`). The
image ships its default embedding model, so it is ready in seconds. Docker Compose, Kubernetes
and Helm setups are in [dakera-deploy](https://github.com/Dakera-AI/dakera-deploy).

### 2. Install

```bash
pip install llamaindex-dakera
```

### 3. Remember across sessions

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
```

---

## What's new in 0.4.0

- **Server v0.12.3** support with `dakera>=0.14.1`.
- `DakeraMemoryBlock`: Dakera as long-term memory of LlamaIndex `Memory` and its agents.
- `DakeraIndexStore`: metadata filters, `hybrid` and `text_search` query modes, `doc_ids`,
  `delete(ref_doc_id)` that really deletes a document's nodes, `delete_nodes`, async writes,
  `embedding_model`, and no client-side embedding (`is_embedding_query = False`).
- Recalls return at most `top_k` ranked memories (a v0.12.3 server can add a session's
  neighbouring memories as context; they are requested separately and left out), and a memory
  and the sentences the server derives from it are not recalled twice.
- `put(event_time=...)` (*v0.12.3*), `get(rerank=...)` with `score` and `importance` per result,
  `consolidate(memory_ids)`; sessions: `start(idle_timeout_secs=...)`, `touch()`,
  `end(summary=...)`.

See the [CHANGELOG](CHANGELOG.md) for details.

---

## Agent memory

### With a LlamaIndex agent

`DakeraMemoryBlock` is a long-term memory block for LlamaIndex
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

`DakeraMemoryBlock` options: `memory_store`, `name` (`"DakeraMemory"`, the tag in the prompt),
`similarity_top_k` (`5`), `retrieval_context_window` (`5` latest messages form the query),
`importance` (of stored messages; default the store's `default_importance`), and LlamaIndex's
`priority` / `accept_short_term_memory`.

### DakeraMemoryStore

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `agent_id` | `str` | — | Agent whose memories are stored and recalled |
| `api_key` | `str` | `""` | Dakera API key (empty when the server runs without auth) |
| `recall_k` | `int` | `5` | Memories `get()` returns by default |
| `min_importance` | `float` | `0.0` | Recall only memories at least this important |
| `default_importance` | `float` | `0.7` | Importance (0–1) of stored memories |

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
- `aput()` / `aget()` are the async variants.

---

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `namespace` | `str` | — | Namespace to read/write (created on the first write) |
| `api_key` | `str` | `""` | Dakera API key |
| `embedding_model` | `str \| None` | server default | Server-side embedding model, e.g. `"bge-large"`, `"minilm"`; use one model per namespace |

---

## Sessions

Group the memories of one conversation. The server ends a session after 4 h without activity by
default (`ended_reason: "idle"`).

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

## Knowledge graph, entities and namespaces

```python
from llama_index_dakera import DakeraEntityExtractor, DakeraKnowledgeGraph, DakeraNamespaceManager

# memory_a, memory_b, memory_c: ids of the agent's memories (e.g. store.put(...)["id"])
kg = DakeraKnowledgeGraph(api_url="http://localhost:3000", api_key="dk-mykey", agent_id="assistant")
kg.link(memory_a, memory_b, label="same project")  # explicit link between two memories
kg.link(memory_b, memory_c)
kg.query(root_id=memory_a, max_depth=2)  # {"edges": [{"source", "target", "type"}], ...}
kg.find_path(memory_a, memory_c)  # {"path": [a, b, c], "hop_count": 2}
kg.build(memory_a, depth=2)  # the graph around one memory
kg.export()  # every edge of the agent
kg.summarize([memory_a, memory_b])  # stores a new summary memory
kg.deduplicate()  # merge duplicate memories

entities = DakeraEntityExtractor(
    api_url="http://localhost:3000", api_key="dk-mykey", agent_id="assistant"
)
entities.extract(
    "Sarah Chen joined Acme in Paris.", entity_types=["person", "organization", "location"]
)
entities.configure(["person", "organization"])  # extract entities from the agent's new memories
entities.memory_entities(memory_a)

namespaces = DakeraNamespaceManager(api_url="http://localhost:3000", api_key="dk-mykey")
namespaces.create("embeddings-768", dimensions=768)
namespaces.list()
namespaces.stats("product-docs")
```

Named entity types load the server's GLiNER model on first use (a ~782 MB download from Hugging
Face); a server that cannot load it serves its rule-based extractor only.

---

## Examples

Every example runs against a live server and needs no LLM API key:

| Example | Shows |
|---------|-------|
| [`basic_memory.py`](examples/basic_memory.py) | `put` / `get` |
| [`memory_block.py`](examples/memory_block.py) | `DakeraMemoryBlock` in LlamaIndex `Memory` |
| [`advanced_memory.py`](examples/advanced_memory.py) | importance, tags, types, TTL, `event_time`, filtered recall, hybrid search, consolidate |
| [`batch_operations.py`](examples/batch_operations.py) | `batch_get`, `batch_delete` |
| [`sessions.py`](examples/sessions.py) | sessions: idle timeout, `touch`, summary, context manager |
| [`knowledge_graph.py`](examples/knowledge_graph.py) | link, query, path, build, export, summarize |
| [`rag_pipeline.py`](examples/rag_pipeline.py) | `VectorStoreIndex` on Dakera, retrieval, RAG prompt, `delete_ref_doc` |
| [`hybrid_search.py`](examples/hybrid_search.py) | semantic, hybrid (`alpha`), BM25 and filtered retrieval |

```bash
export DAKERA_API_URL=http://localhost:3000
export DAKERA_API_KEY=dk-mykey   # leave unset if the server runs without auth
python examples/basic_memory.py
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) — unit tests run without a server; the live tests in
`tests/test_integration.py` run when `DAKERA_API_URL` is set.

## Related packages

| Package | Framework | Language |
|---------|-----------|----------|
| [`langchain-dakera`](https://pypi.org/project/langchain-dakera/) | LangChain | Python |
| [`crewai-dakera`](https://pypi.org/project/crewai-dakera/) | CrewAI | Python |
| [`autogen-dakera`](https://pypi.org/project/autogen-dakera/) | AutoGen | Python |
| [`strands-dakera`](https://pypi.org/project/strands-dakera/) | Strands Agents | Python |
| [`@dakera-ai/langchain`](https://www.npmjs.com/package/@dakera-ai/langchain) | LangChain.js | TypeScript |

## Links

- [Dakera](https://dakera.ai) · [Documentation](https://dakera.ai/docs) ·
  [LlamaIndex integration guide](https://dakera.ai/integrations/llamaindex.html)
- [dakera-deploy](https://github.com/Dakera-AI/dakera-deploy) — Docker Compose, Kubernetes, Helm
- [Dakera Python SDK](https://github.com/Dakera-AI/dakera-py) — the low-level client this package uses

## License

MIT © [Dakera AI](https://dakera.ai)
