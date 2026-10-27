"""Live tests against a real Dakera server.

Skipped unless ``DAKERA_API_URL`` is set (CI runs them against the pinned server image):

    DAKERA_API_URL=http://localhost:3300 DAKERA_API_KEY=test-key pytest tests/test_integration.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from llama_index_dakera import (
    DakeraEntityExtractor,
    DakeraIndexStore,
    DakeraKnowledgeGraph,
    DakeraMemoryStore,
    DakeraSessionManager,
)

API_URL = os.environ.get("DAKERA_API_URL", "")
API_KEY = os.environ.get("DAKERA_API_KEY", "test-key")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not API_URL, reason="DAKERA_API_URL not set"),
]


def _agent() -> str:
    return f"li-it-{uuid.uuid4().hex[:10]}"


def _store(agent_id: str, **kwargs: int) -> DakeraMemoryStore:
    return DakeraMemoryStore(api_url=API_URL, api_key=API_KEY, agent_id=agent_id, **kwargs)


def test_recall_of_an_agent_without_memories_is_empty():
    store = _store(_agent())
    assert store.get("anything") == []
    assert asyncio.run(store.aget("anything")) == []


def test_session_recall_returns_at_most_top_k():
    agent = _agent()
    store = _store(agent, recall_k=1)
    sessions = DakeraSessionManager(api_url=API_URL, api_key=API_KEY, agent_id=agent)
    with sessions:
        sid = sessions.active_session_id
        assert sid
        for text in ("I have a dog named Rex", "Rex is three years old", "Rex likes the park"):
            store.put(text, session_id=sid)
        assert sessions.touch() == "active"
        # Neighbourhood context of the session stays out of the ranked results.
        results = store.get("Tell me about my dog")
        assert len(results) == 1
        assert isinstance(results[0]["score"], float)
        assert len(asyncio.run(store.aget("Tell me about my dog"))) == 1
        assert [len(r) for r in store.batch_get(["dog", "park"])] == [1, 1]
        assert len(sessions.memories(sid)) == 3
    info = sessions.get(sid)
    assert info["ended_at"] is not None
    assert info["ended_reason"] == "client"


def test_put_with_event_time_and_consolidate():
    store = _store(_agent())
    dated = store.put("Alice moved to Lisbon", event_time="2024-05-01T10:00:00Z")
    assert dated["metadata"]["_dakera_event_time"] == 1714557600
    first = store.put("Bob prefers tea")
    second = asyncio.run(store.aput("Bob drinks green tea every morning"))
    merged = store.consolidate([first["id"], second["id"]])
    assert sorted(merged["source_memory_ids"]) == sorted([first["id"], second["id"]])


def test_knowledge_graph_and_entity_config():
    agent = _agent()
    store = _store(agent)
    a = store.put("Carol leads Project Alpha")["id"]
    b = store.put("Project Alpha ships in May")["id"]
    kg = DakeraKnowledgeGraph(api_url=API_URL, api_key=API_KEY, agent_id=agent)
    assert kg.link(a, b, label="same project")["edge_type"] == "linked_by"
    assert kg.find_path(a, b)["path"] == [a, b]
    assert kg.summarize([a, b])["source_count"] == 2
    DakeraEntityExtractor(api_url=API_URL, api_key=API_KEY, agent_id=agent).configure(["PERSON"])


def test_index_store_add_and_query():
    index = DakeraIndexStore(
        api_url=API_URL, api_key=API_KEY, namespace=f"li-it-{uuid.uuid4().hex[:8]}"
    )
    index.add(
        [
            TextNode(id_="n1", text="Rust provides memory safety", metadata={"lang": "rust"}),
            TextNode(id_="n2", text="Python is readable", metadata={"lang": "python"}),
        ]
    )
    result = index.query(VectorStoreQuery(query_str="memory safe language", similarity_top_k=1))
    assert result.ids == ["n1"]
    assert result.nodes is not None and result.nodes[0].metadata["lang"] == "rust"
