"""Basic agent memory with LlamaIndex and Dakera.

Stores a few memories of an agent and recalls the most relevant ones for a
query. The memories live on the Dakera server, so they survive restarts.

Run a Dakera server first (see the README), then:

    pip install llamaindex-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/basic_memory.py
"""

import os
import uuid

from llama_index_dakera import DakeraMemoryStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

store = DakeraMemoryStore(
    api_url=api_url,
    api_key=api_key,
    agent_id=f"llamaindex-basic-{uuid.uuid4().hex[:8]}",  # a fresh agent per run
    recall_k=2,  # memories get() returns by default
    default_importance=0.8,
)

store.put("The user prefers Python over JavaScript.")
store.put("The project deadline is next Friday.", importance=0.9)
store.put("The codebase uses FastAPI with SQLAlchemy.", memory_type="semantic")

for query in ("programming language preferences", "project timeline"):
    print(f"Recalled for {query!r}:")
    for m in store.get(query):
        print(f"  [{m['score']:.3f}] {m['content']}")
