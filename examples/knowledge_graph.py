"""Knowledge graph operations with LlamaIndex and Dakera.

Demonstrates graph querying, traversal, and export.

Usage:
    export DAKERA_API_URL="http://localhost:3000"
    python knowledge_graph.py
"""

import os

from llama_index_dakera.knowledge_graph import DakeraKnowledgeGraph
from llama_index_dakera.memory_store import DakeraMemoryStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

kg = DakeraKnowledgeGraph(
    api_url=api_url,
    api_key=api_key,
    agent_id="llamaindex-kg-demo",
)

print("--- Graph export ---")
graph = kg.export()
print(f"Nodes: {graph['node_count']}, Edges: {graph['edge_count']}")

print("\n--- Graph query ---")
results = kg.query(max_depth=3, limit=10)
print(f"Found {results['edge_count']} edges")
for edge in results["edges"][:5]:
    print(f"  {edge}")

print("\n--- Summarize ---")
store = DakeraMemoryStore(api_url=api_url, api_key=api_key, agent_id="llamaindex-kg-demo")
ids = [
    store.put("Project Alpha uses Python and is led by Sarah Chen.")["id"],
    store.put("Sarah Chen works with Bob Smith on the backend.")["id"],
]
summary = kg.summarize(ids, dry_run=True)
print(f"Summary of {summary['source_count']} memories: {summary['summary_memory']['content'][:80]}")
