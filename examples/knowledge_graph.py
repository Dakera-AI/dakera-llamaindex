"""Knowledge graph operations with LlamaIndex and Dakera.

Links an agent's memories, then queries, traverses, builds, exports and
summarizes the resulting graph.

Run a Dakera server first (see the README), then:

    pip install llamaindex-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/knowledge_graph.py
"""

import os
import uuid

from llama_index_dakera import DakeraKnowledgeGraph, DakeraMemoryStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")
agent_id = f"llamaindex-kg-{uuid.uuid4().hex[:8]}"

store = DakeraMemoryStore(api_url=api_url, api_key=api_key, agent_id=agent_id)
project = store.put("Project Alpha is a Python service led by Sarah Chen.")["id"]
team = store.put("Sarah Chen works with Bob Smith on the Project Alpha backend.")["id"]
stack = store.put("Project Alpha runs on FastAPI, PostgreSQL and Redis.")["id"]

kg = DakeraKnowledgeGraph(api_url=api_url, api_key=api_key, agent_id=agent_id)

print("--- Link memories: project -> team -> stack ---")
kg.link(project, team, label="same project")
kg.link(team, stack, label="same project")

print("\n--- Query edges from the project memory ---")
result = kg.query(root_id=project, max_depth=2)
print(f"  {result['node_count']} nodes, {result['edge_count']} edges")
for edge in result["edges"]:
    print(f"    {edge['source']} -[{edge['type']}]-> {edge['target']}")

print("\n--- Shortest path from project to stack ---")
path = kg.find_path(project, stack)
print(f"  {path['hop_count']} hops: {' -> '.join(path['path'])}")

print("\n--- Graph around the team memory ---")
graph = kg.build(team, depth=2)
print(f"  root: {graph['root']['memory']['content']}")

print("\n--- Export ---")
export = kg.export()
print(f"  {export['node_count']} nodes, {export['edge_count']} edges")

print("\n--- Summarize two memories into a new one ---")
summary = kg.summarize([project, stack])
print(f"  {summary['source_count']} sources -> {summary['summary_memory']['id']}")
