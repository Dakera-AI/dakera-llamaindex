"""Basic agent memory with LlamaIndex and Dakera.

Stores and retrieves agent memories using semantic search.

Usage:
    export DAKERA_API_URL="http://localhost:3300"
    export DAKERA_API_KEY="dk-..."          # optional
    pip install llamaindex-dakera
    python basic_memory.py
"""

import os

from llama_index_dakera import DakeraMemoryStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3300")
api_key = os.environ.get("DAKERA_API_KEY", "")

store = DakeraMemoryStore(
    api_url=api_url,
    agent_id="llamaindex-demo",
    api_key=api_key,
    recall_k=3,
    default_importance=0.8,
)

store.put("The user prefers Python over JavaScript.")
store.put("Project deadline is next Friday.", memory_type="episodic", importance=0.9)
store.put("The codebase uses FastAPI with SQLAlchemy.", memory_type="semantic")

print("Recalling memories about 'programming language preferences':")
memories = store.get("programming language preferences")
for m in memories:
    print(f"  [{m['score']:.3f}] {m['content']}")

print("\nRecalling memories about 'project timeline':")
memories = store.get("project timeline")
for m in memories:
    print(f"  [{m['score']:.3f}] {m['content']}")
