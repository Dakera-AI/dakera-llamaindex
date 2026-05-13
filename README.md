# llamaindex-dakera

[![PyPI](https://img.shields.io/pypi/v/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![Python](https://img.shields.io/pypi/pyversions/llamaindex-dakera)](https://pypi.org/project/llamaindex-dakera/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LlamaIndex integration for the [Dakera AI](https://dakera.ai) memory platform.**

Drop-in LlamaIndex components using Dakera's server-side embedding — no local model required.

## Installation

```bash
pip install llamaindex-dakera
```

## Quick Start

```python
from llama_index_dakera import DakeraMemoryStore, DakeraIndexStore
from llama_index.core import VectorStoreIndex, StorageContext

# Agent memory
memory = DakeraMemoryStore(api_url="https://your-dakera-instance.com", api_key="dk-...", agent_id="my-agent")

# RAG index (no local embedding model needed)
vector_store = DakeraIndexStore(api_url="https://your-dakera-instance.com", api_key="dk-...", namespace="docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

## Links

- [Dakera Documentation](https://docs.dakera.ai/integrations/llamaindex)
- [Dakera AI](https://dakera.ai)
