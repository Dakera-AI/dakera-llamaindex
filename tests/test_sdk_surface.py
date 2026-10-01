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
        {"id": "sess_1", "started_at": 1, "ended_at": None, "memory_count": 2}
    ]
    client.list_sessions.assert_called_once_with("a", active_only=False)


def test_summarize_sends_memory_ids():
    client = _spec_client()
    client.summarize.return_value = {"summary_memory": {"id": "s"}, "source_count": 2}
    with patch("llama_index_dakera.knowledge_graph.DakeraClient", return_value=client):
        kg = DakeraKnowledgeGraph(api_url="http://localhost:3000", agent_id="a")
    kg.summarize(["m1", "m2"], target_type="semantic")
    client.summarize.assert_called_once_with(
        "a", memory_ids=["m1", "m2"], target_type="semantic"
    )


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
    with patch("llama_index_dakera.index_store.DakeraClient", return_value=client), patch(
        "llama_index_dakera.index_store.AsyncDakeraClient"
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
    with patch("llama_index_dakera.memory_store.DakeraClient", return_value=client), patch(
        "llama_index_dakera.memory_store.AsyncDakeraClient",
        return_value=create_autospec(AsyncDakeraClient, instance=True),
    ):
        store = DakeraMemoryStore(api_url="http://localhost:3000", agent_id="a")
    assert store.get("q", tags=["work"]) == []
    _, kwargs = client.recall.call_args
    assert kwargs["tags"] == ["work"]
