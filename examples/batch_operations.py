"""Batch operations with LlamaIndex and Dakera.

Stores a batch of memories, recalls for several queries at once, and deletes
several memories in one call.

Run a Dakera server first (see the README), then:

    pip install llamaindex-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/batch_operations.py
"""

import os
import uuid

from llama_index_dakera import DakeraMemoryStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

memory = DakeraMemoryStore(
    api_url=api_url,
    api_key=api_key,
    agent_id=f"llamaindex-batch-{uuid.uuid4().hex[:8]}",
)

items = [
    "30-day return policy for unopened items.",
    "International shipping to 40+ countries.",
    "Payment by Visa, Mastercard and PayPal.",
    "Loyalty program: 1 point per dollar spent.",
    "Gift cards in $25, $50 and $100 denominations.",
]
for item in items:
    memory.put(item, importance=0.7, tags=["faq"])
print(f"Stored {len(items)} memories.")

print("\n--- Batch recall ---")
queries = ["shipping abroad", "paying by card", "sending items back"]
results = memory.batch_get(queries, top_k=1)
for query, matches in zip(queries, results):
    for m in matches:
        print(f"  {query!r}: [{m['score']:.3f}] {m['content']}")

print("\n--- Batch delete ---")
to_delete = [m["id"] for matches in results for m in matches]
memory.batch_delete(to_delete)
print(f"  Deleted {len(to_delete)} memories.")
