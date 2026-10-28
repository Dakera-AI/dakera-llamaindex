"""Advanced memory features with LlamaIndex and Dakera.

Importance, tags, memory types, TTL and event time on store; filtered and
reranked recall; the server's hybrid memory search; importance updates;
consolidating memories into one; deleting; and agent statistics.

Run a Dakera server first (see the README), then:

    pip install llamaindex-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/advanced_memory.py
"""

import os
import uuid
from datetime import datetime, timezone

from llama_index_dakera import DakeraMemoryStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

memory = DakeraMemoryStore(
    api_url=api_url,
    api_key=api_key,
    agent_id=f"llamaindex-advanced-{uuid.uuid4().hex[:8]}",
)

print("--- Store with importance, tags, type, TTL and event time ---")
memory.put("The user said hello.", importance=0.2, ttl_seconds=3600)  # expires in 1 h
allergy = memory.put(
    "The user is allergic to peanuts.", memory_type="semantic", importance=0.95, tags=["health"]
)
memory.put("The user's birthday is March 15.", importance=0.8, tags=["personal"])
# event_time (server v0.12.3+): when it happened, for history written after the fact.
# Recency and time filters use it instead of the time of storage.
memory.put(
    "The user hiked the Tour du Mont Blanc.",
    importance=0.6,
    tags=["travel"],
    event_time=datetime(2025, 7, 14, 9, 0, tzinfo=timezone.utc),
)
lisbon = memory.put("The user visited Lisbon for a conference.", tags=["travel"])
porto = memory.put("The user visited Porto after Lisbon.", tags=["travel"])

print("\n--- Recall: the important things ---")
for r in memory.get("What should I know about the user?", min_importance=0.8):
    print(f"  [{r['importance']:.2f}] score={r['score']:.3f} {r['content']}")

print("\n--- Recall: only memories tagged 'travel', without reranking ---")
for r in memory.get("Where has the user been?", tags=["travel"], rerank=False):
    print(f"  [{r['importance']:.2f}] {r['content']}")

print("\n--- Hybrid memory search (vector + BM25 on the server) ---")
for r in memory.hybrid_search("birthday", top_k=1):
    print(f"  score={r['score']:.3f} {r['content']}")

print("\n--- Update importance ---")
memory.update_importance(lisbon["id"], 0.9)
print(f"  {lisbon['id']} -> 0.9")

print("\n--- Consolidate the two Portugal memories into one ---")
# Memories with a TTL or user metadata (event_time included) are never consolidated.
merged = memory.consolidate([lisbon["id"], porto["id"]], target_type="semantic")
print(f"  {merged['memories_removed']} memories -> {merged['consolidated_memory']['content']!r}")

print("\n--- Delete ---")
memory.delete(allergy["id"])
print(f"  deleted {allergy['id']}")

stats = memory.stats()
print(f"\n{stats['total_memories']} memories, by type: {stats['memories_by_type']}")
