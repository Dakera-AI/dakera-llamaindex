"""Advanced memory features with LlamaIndex and Dakera.

Demonstrates importance levels, tags, and agent tools.

Usage:
    export DAKERA_API_URL="http://localhost:3300"
    python advanced_memory.py
"""

import os

from llama_index_dakera import DakeraMemoryStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3300")
api_key = os.environ.get("DAKERA_API_KEY", "")

memory = DakeraMemoryStore(
    api_url=api_url,
    api_key=api_key,
    agent_id="llamaindex-advanced-demo",
)

print("--- Different importance levels ---")
memory.put("casual greeting", importance=0.3)
memory.put("user is allergic to peanuts", importance=0.95)
memory.put("birthday is March 15", importance=0.8, tags=["personal"])

print("--- High-importance recall ---")
results = memory.get("important facts", min_importance=0.7)
for r in results:
    print(f"  [{r['importance']:.1f}] {r['content']}")
