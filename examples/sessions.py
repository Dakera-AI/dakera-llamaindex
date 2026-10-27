"""Session-scoped memory with LlamaIndex and Dakera.

Starts a session, stores memories in it, keeps it alive, reads it back and
ends it with a summary. The server ends a session after 4 h without activity
by default.

Run a Dakera server first (see the README), then:

    pip install llamaindex-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/sessions.py
"""

import os
import uuid

from llama_index_dakera import DakeraMemoryStore, DakeraSessionManager

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")
agent_id = f"llamaindex-sessions-{uuid.uuid4().hex[:8]}"

sessions = DakeraSessionManager(api_url=api_url, api_key=api_key, agent_id=agent_id)
memory = DakeraMemoryStore(api_url=api_url, api_key=api_key, agent_id=agent_id)

# idle_timeout_secs (server v0.12.2+): end this session after 30 min without activity.
session_id = sessions.start(metadata={"topic": "onboarding"}, idle_timeout_secs=1800)
print(f"Started session {session_id}")

memory.put("The user asked how to reset their password.", session_id=session_id)
memory.put("The user asked where to find their API keys.", session_id=session_id)
print(f"Touched the session: {sessions.touch()}")  # keeps an idle session open

print("\n--- Memories in the session ---")
for m in sessions.memories(session_id):
    print(f"  - {m['content']}")

sessions.end(summary="Helped with password reset and API keys.")
info = sessions.get(session_id)
print(f"\nEnded: {info['summary']!r} ({info['memory_count']} memories)")

# As a context manager the session starts on entry and ends on exit.
with DakeraSessionManager(api_url=api_url, api_key=api_key, agent_id=agent_id) as follow_up:
    print(f"\nFollow-up session {follow_up.active_session_id} is active")

print(f"Sessions of the agent: {len(sessions.list())}")
