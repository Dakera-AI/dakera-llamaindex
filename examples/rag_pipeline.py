"""RAG pipeline using DakeraIndexStore with LlamaIndex.

Indexes documents into Dakera's vector store and queries them using
LlamaIndex's VectorStoreQuery interface. No local embeddings needed —
Dakera handles embedding server-side.

Usage:
    export DAKERA_API_URL="http://localhost:3300"
    export DAKERA_API_KEY="dk-..."          # optional
    pip install llamaindex-dakera llama-index-core
    python rag_pipeline.py
"""

import os

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from llama_index_dakera import DakeraIndexStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3300")
api_key = os.environ.get("DAKERA_API_KEY", "")

store = DakeraIndexStore(
    api_url=api_url,
    namespace="llamaindex-rag-demo",
    api_key=api_key,
)

nodes = [
    TextNode(
        text="Dakera provides persistent memory for AI agents.",
        metadata={"topic": "overview"},
    ),
    TextNode(
        text="Vector search uses cosine similarity over embeddings.",
        metadata={"topic": "search"},
    ),
    TextNode(
        text="LlamaIndex orchestrates data ingestion for LLMs.",
        metadata={"topic": "framework"},
    ),
    TextNode(
        text="Server-side embedding removes the need for local GPUs.",
        metadata={"topic": "architecture"},
    ),
]

print("Indexing nodes...")
ids = store.add(nodes)
print(f"Indexed {len(ids)} nodes.")

query = VectorStoreQuery(query_str="How does Dakera handle embeddings?", similarity_top_k=2)
result = store.query(query)

print(f"\nQuery: '{query.query_str}'")
for node, score in zip(result.nodes or [], result.similarities or []):
    print(f"  [{score:.3f}] {node.text}")
