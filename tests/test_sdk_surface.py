"""Calls into the dakera SDK checked against its real signatures.

``create_autospec`` mocks fail on a method the SDK does not have or on
arguments its signature does not accept, which plain ``MagicMock`` hides.
"""

from unittest.mock import create_autospec, patch

from dakera import DakeraClient
from dakera.models import IndexStats

from llama_index_dakera import (
    DakeraIndexStore,
    DakeraKnowledgeGraph,
    DakeraNamespaceManager,
    DakeraSessionManager,
)


def _spec_client() -> DakeraClient:
    return create_autospec(DakeraClient, instance=True)


def test_list_sessions_unwraps_server_envelope():
    client = _spec_client()
    client.list_sessions.return_value = {
        "sessions": [{"id": "sess_1", "started_at": 1, "memory_count": 2}],
        "total": 1,
    }
    with patch("llama_index_dakera.sessions.DakeraClient", return_value=client):
        mgr = DakeraSessionManager(api_url="http://localhost:3000", agent_id="a")
    assert mgr.list() == [
        {
            "id": "sess_1",
            "started_at": 1,
            "ended_at": None,
            "memory_count": 2,
            "ended_reason": None,
        }
    ]
    client.list_sessions.assert_called_once_with("a", active_only=False)


def test_summarize_sends_memory_ids():
    client = _spec_client()
    client.summarize.return_value = {"summary_memory": {"id": "s"}, "source_count": 2}
    with patch("llama_index_dakera.knowledge_graph.DakeraClient", return_value=client):
        kg = DakeraKnowledgeGraph(api_url="http://localhost:3000", agent_id="a")
    kg.summarize(["m1", "m2"], target_type="semantic")
    client.summarize.assert_called_once_with("a", memory_ids=["m1", "m2"], target_type="semantic")


def test_namespace_stats_maps_index_stats():
    client = _spec_client()
    client.get_index_stats.return_value = IndexStats.from_dict(
        {"vector_count": 3, "dimension": 384, "index_type": "hnsw"}
    )
    with patch("llama_index_dakera.namespaces.DakeraClient", return_value=client):
        mgr = DakeraNamespaceManager(api_url="http://localhost:3000")
    assert mgr.stats("docs") == {"total_vectors": 3, "dimensions": 384, "index_type": "hnsw"}


def test_index_store_delete_uses_filter_not_delete_all():
    client = _spec_client()
    client.delete.return_value = {"deleted": 1, "failed": 0, "errors": []}
    with (
        patch("llama_index_dakera.index_store.DakeraClient", return_value=client),
        patch("llama_index_dakera.index_store.AsyncDakeraClient"),
    ):
        store = DakeraIndexStore(api_url="http://localhost:3000", namespace="docs")
    store.delete("doc-1")
    client.delete.assert_called_once_with("docs", filter={"ref_doc_id": {"$eq": "doc-1"}})


def test_memory_store_get_forwards_tags():
    from dakera import AsyncDakeraClient
    from dakera.models import RecallResponse

    from llama_index_dakera import DakeraMemoryStore

    client = _spec_client()
    client.recall.return_value = RecallResponse.from_dict({"memories": []})
    with (
        patch("llama_index_dakera.memory_store.DakeraClient", return_value=client),
        patch(
            "llama_index_dakera.memory_store.AsyncDakeraClient",
            return_value=create_autospec(AsyncDakeraClient, instance=True),
        ),
    ):
        store = DakeraMemoryStore(api_url="http://localhost:3000", agent_id="a")
    assert store.get("q", tags=["work"]) == []
    _, kwargs = client.recall.call_args
    assert kwargs["tags"] == ["work"]


# --- Server v0.12.3 / dakera 0.14.1 ---------------------------------------------


def _memory_store(client: DakeraClient, async_client=None):
    from dakera import AsyncDakeraClient

    from llama_index_dakera import DakeraMemoryStore

    with (
        patch("llama_index_dakera.memory_store.DakeraClient", return_value=client),
        patch(
            "llama_index_dakera.memory_store.AsyncDakeraClient",
            return_value=async_client or create_autospec(AsyncDakeraClient, instance=True),
        ),
    ):
        return DakeraMemoryStore(api_url="http://localhost:3000", agent_id="a", recall_k=2)


def _recall_with_inline_context():
    from dakera import RecallResponse

    # A v0.12.3 server in neighborhood_mode="inline" appends session context after
    # the ranked results; a v0.12.2 server does the same without marking it.
    return RecallResponse.from_dict(
        {
            "memories": [
                {"id": "m1", "content": "first", "rank_score": 0.9, "importance": 0.8},
                {"id": "m2", "content": "second", "rank_score": 0.8},
                {"id": "n1", "content": "context", "score": 0.99, "retrieval": "neighborhood"},
                {"id": "m3", "content": "unmarked extra", "rank_score": 0.1},
            ]
        }
    )


def test_get_returns_at_most_top_k_ranked_results():
    client = _spec_client()
    client.recall.return_value = _recall_with_inline_context()
    store = _memory_store(client)
    results = store.get("q", rerank=False)
    assert [r["id"] for r in results] == ["m1", "m2"]
    assert results[0] == {
        "content": "first",
        "id": "m1",
        "score": 0.9,
        "importance": 0.8,
        "metadata": None,
    }
    client.recall.assert_called_once_with(
        "a", query="q", top_k=2, neighborhood_mode="separate", rerank=False
    )


def test_batch_get_caps_each_query():
    client = _spec_client()
    client.recall.return_value = _recall_with_inline_context()
    store = _memory_store(client)
    assert [[r["id"] for r in rs] for rs in store.batch_get(["x", "y"], top_k=1)] == [
        ["m1"],
        ["m1"],
    ]
    client.recall.assert_called_with("a", query="y", top_k=1, neighborhood_mode="separate")


def test_aget_returns_at_most_top_k_ranked_results():
    import asyncio

    from dakera import AsyncDakeraClient

    async_client = create_autospec(AsyncDakeraClient, instance=True)
    async_client.recall.return_value = _recall_with_inline_context()
    store = _memory_store(_spec_client(), async_client)
    results = asyncio.run(store.aget("q", top_k=1))
    assert [r["id"] for r in results] == ["m1"]
    async_client.recall.assert_called_once_with(
        "a", query="q", top_k=1, neighborhood_mode="separate"
    )


def test_put_and_aput_forward_event_time():
    import asyncio

    from dakera import AsyncDakeraClient

    client = _spec_client()
    client.store_memory.return_value = {"id": "m1"}
    async_client = create_autospec(AsyncDakeraClient, instance=True)
    async_client.store_memory.return_value = {"id": "m2"}
    store = _memory_store(client, async_client)
    store.put("met Bob", event_time=1714557600)
    client.store_memory.assert_called_once_with(
        "a", content="met Bob", memory_type="episodic", importance=0.7, event_time=1714557600
    )
    asyncio.run(store.aput("met Carol", event_time="2024-05-01T10:00:00Z"))
    async_client.store_memory.assert_called_once_with(
        "a",
        content="met Carol",
        memory_type="episodic",
        importance=0.7,
        event_time="2024-05-01T10:00:00Z",
    )


def test_consolidate_sends_memory_ids():
    import pytest

    client = _spec_client()
    client.consolidate.return_value = {"consolidated_memory": {"id": "c"}}
    store = _memory_store(client)
    store.consolidate(["m1", "m2"], target_type="semantic")
    client.consolidate.assert_called_once_with("a", memory_ids=["m1", "m2"], target_type="semantic")
    with pytest.raises(ValueError):
        store.consolidate(["m1"])


def test_entity_configure_targets_agent_memory_namespace():
    from llama_index_dakera import DakeraEntityExtractor

    client = _spec_client()
    client.configure_namespace_ner.return_value = {}
    with patch("llama_index_dakera.entities.DakeraClient", return_value=client):
        ex = DakeraEntityExtractor(api_url="http://localhost:3000", agent_id="a")
    ex.configure(["PERSON"], extract_entities=False)
    client.configure_namespace_ner.assert_called_once_with(
        "_dakera_agent_a", extract_entities=False, entity_types=["PERSON"]
    )


def test_session_idle_timeout_touch_and_summary():
    from dakera.models import SessionTouchResponse

    client = _spec_client()
    client.start_session.return_value = {"id": "urn:chat:42", "agent_id": "a"}
    client.touch_session.return_value = SessionTouchResponse.from_dict(
        {"session": {"id": "urn:chat:42"}, "session_state": "active"}
    )
    client.end_session.return_value = {}
    with patch("llama_index_dakera.sessions.DakeraClient", return_value=client):
        mgr = DakeraSessionManager(api_url="http://localhost:3000", agent_id="a")
    assert mgr.start(idle_timeout_secs=0) == "urn:chat:42"
    client.start_session.assert_called_once_with("a", metadata=None, idle_timeout_secs=0)
    assert mgr.touch() == "active"
    client.touch_session.assert_called_once_with("urn:chat:42")
    mgr.end("talked about Rex")
    client.end_session.assert_called_once_with("urn:chat:42", "talked about Rex")


def test_session_memories_leave_out_derived_sub_memories():
    client = _spec_client()
    client.session_memories.return_value = [
        {"id": "m1", "content": "I have a dog", "importance": 0.7, "tags": []},
        {"id": "mem_s" + "0" * 24, "content": "a dog", "tags": ["dakera-curated"]},
    ]
    with patch("llama_index_dakera.sessions.DakeraClient", return_value=client):
        mgr = DakeraSessionManager(api_url="http://localhost:3000", agent_id="a")
    assert mgr.memories("s") == [{"id": "m1", "content": "I have a dog", "importance": 0.7}]
    assert len(mgr.memories("s", include_derived=True)) == 2
