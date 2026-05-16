# llamaindex-dakera

[![CI](https://github.com/Dakera-AI/dakera-llamaindex/actions/workflows/ci.yml/badge.svg)](https://github.com/Dakera-AI/dakera-llamaindex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![Downloads](https://img.shields.io/pypi/dm/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![Python](https://img.shields.io/pypi/pyversions/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![dakera.ai](https://img.shields.io/badge/dakera.ai-website-22c55e?style=flat-square)](https://dakera.ai) [![Docs](https://img.shields.io/badge/docs-dakera.ai%2Fdocs-3b82f6?style=flat-square)](https://dakera.ai/docs)

**Drop-in LlamaIndex components backed by [Dakera](https://github.com/Dakera-AI/dakera-deploy) — persistent agent memory and server-side vector indexing with no local embedding model.**

`DakeraMemoryStore` gives your LlamaIndex agents conversation memory that survives restarts. `DakeraIndexStore` replaces local vector indices with Dakera's server-side embedding engine — no OpenAI embeddings API needed for RAG.

---

## Quick Start

### Step 1 — Run Dakera

Dakera is a self-hosted memory server. Spin it up with Docker:

```bash
docker run -d \
  --name dakera \
  -p 3300:3300 \
  -e DAKERA_ROOT_API_KEY=dk-mykey \
  ghcr.io/dakera-ai/dakera:latest
```

For a production setup with persistent storage, use Docker Compose:

```bash
# Download and start
curl -sSfL https://raw.githubusercontent.com/Dakera-AI/dakera-deploy/main/docker-compose.yml \
  -o docker-compose.yml
DAKERA_API_KEY=dk-mykey docker compose up -d

# Verify it's running
curl http://localhost:3300/health
```

> Full deployment guide: [github.com/Dakera-AI/dakera-deploy](https://github.com/Dakera-AI/dakera-deploy)

### Step 2 — Install the integration

```bash
pip install llamaindex-dakera
```

### Step 3 — Use it

```python
from llama_index_dakera import DakeraMemoryStore, DakeraIndexStore

# Agent memory
memory = DakeraMemoryStore(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    agent_id="my-agent",
)

# RAG index — no local embedding model needed
vector_store = DakeraIndexStore(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    namespace="my-docs",
)
```

---

## Installation

```bash
pip install llamaindex-dakera
```

**Requirements:** Python ≥ 3.10, a running Dakera server (see Step 1 above)

---

## DakeraMemoryStore

Persistent conversation memory for LlamaIndex agents. Drop-in replacement for the default in-memory store.

### Usage with a chat agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index_dakera import DakeraMemoryStore

store = DakeraMemoryStore(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    agent_id="react-agent",
)

memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=store,
    chat_store_key="user-1",
)

agent = ReActAgent.from_tools(
    tools=[...],
    llm=OpenAI(model="gpt-4o"),
    memory=memory,
    verbose=True,
)

# First session
response = agent.chat("My project is called NeuralBridge.")
print(response)

# Later session — memory persists
response = agent.chat("What's the name of my project?")
print(response)  # "Your project is called NeuralBridge."
```

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `api_key` | `str` | `""` | Dakera API key |
| `agent_id` | `str` | — | Namespace for this agent's memories |
| `top_k` | `int` | `5` | Memories to retrieve per query |
| `min_importance` | `float` | `0.0` | Minimum importance for recall |

---

## DakeraIndexStore

Server-side embedded vector store for RAG. Dakera embeds documents on the server — no local model, no OpenAI embeddings API needed.

### Indexing documents

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index_dakera import DakeraIndexStore

# Load documents
documents = SimpleDirectoryReader("./docs").load_data()

# Create index backed by Dakera
vector_store = DakeraIndexStore(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    namespace="product-docs",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
```

### Querying

```python
query_engine = index.as_query_engine(similarity_top_k=4)
response = query_engine.query("How does the billing work?")
print(response)
```

### Chat with your documents

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index_dakera import DakeraIndexStore, DakeraMemoryStore

vector_store = DakeraIndexStore(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    namespace="product-docs",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_defaults(storage_context=storage_context)

memory_store = DakeraMemoryStore(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    agent_id="doc-chat",
)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=index.as_retriever(similarity_top_k=4),
    memory=ChatMemoryBuffer.from_defaults(chat_store=memory_store),
)

response = chat_engine.chat("What are the pricing tiers?")
print(response)
```

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `api_key` | `str` | `""` | Dakera API key |
| `namespace` | `str` | — | Vector namespace to read/write |
| `embedding_model` | `str` | namespace default | Server-side embedding model override |

---

## Related packages

| Package | Framework | Language |
|---------|-----------|----------|
| `crewai-dakera` | CrewAI | Python |
| `langchain-dakera` | LangChain | Python |
| `autogen-dakera` | AutoGen | Python |
| `@dakera-ai/langchain` | LangChain.js | TypeScript |

---

## Links

- [Dakera Server](https://github.com/Dakera-AI/dakera-deploy) — self-hosted memory server
- [Dakera Python SDK](https://github.com/Dakera-AI/dakera-py) — low-level API client
- [Integration guide](https://dakera.ai/integrations/llamaindex.html) — full setup walkthrough
- [All integrations](https://dakera.ai/integrations/)

---

## License

MIT © [Dakera AI](https://dakera.ai)

---

<div align="center">

**[dakera.ai](https://dakera.ai)** · [Documentation](https://dakera.ai/docs) · [Request Early Access](https://dakera.ai#cta)

</div>
