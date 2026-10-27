"""Tests for DakeraMemoryStore (LlamaIndex integration)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dakera import RecallResponse

from llama_index_dakera import DakeraMemoryStore


@pytest.fixture
def mem_store():
    with (
        patch("llama_index_dakera.memory_store.DakeraClient") as MC,
        patch("llama_index_dakera.memory_store.AsyncDakeraClient") as AMC,
    ):
        mock_client = MagicMock()
        mock_async = AsyncMock()
        MC.return_value = mock_client
        AMC.return_value = mock_async
        store = DakeraMemoryStore(
            api_url="http://localhost:3000", api_key="test", agent_id="agent-1", recall_k=3
        )
        store._client = mock_client
        store._async_client = mock_async
        yield store, mock_client, mock_async


def test_put_stores_memory(mem_store):
    store, mock_client, _ = mem_store
    store.put("The user prefers Python")
    mock_client.store_memory.assert_called_once_with(
        "agent-1", content="The user prefers Python", memory_type="episodic", importance=0.7
    )


def test_get_recalls_memories(mem_store):
    store, mock_client, _ = mem_store
    mock_client.recall.return_value = RecallResponse.from_dict(
        {"memories": [{"id": "m-1", "content": "The user prefers Python", "rank_score": 0.9}]}
    )
    results = store.get("What does the user prefer?")
    assert len(results) == 1
    assert results[0]["content"] == "The user prefers Python"
    assert results[0]["score"] == 0.9
    mock_client.recall.assert_called_once_with(
        "agent-1", query="What does the user prefer?", top_k=3, neighborhood_mode="separate"
    )


def test_delete_calls_forget(mem_store):
    store, mock_client, _ = mem_store
    store.delete("mem-123")
    mock_client.forget.assert_called_once_with("agent-1", "mem-123")


def test_get_collapses_derived_sentences_into_their_source(mem_store):
    store, mock_client, _ = mem_store
    mock_client.recall.return_value = RecallResponse.from_dict(
        {
            "memories": [
                {"memory": {"id": "m-1", "content": "I work at Acme. I like tea."}},
                {
                    "memory": {
                        "id": "mem_s1",
                        "content": "I like tea.",
                        "metadata": {"source_memory_id": "m-1"},
                    }
                },
            ]
        }
    )
    assert [r["id"] for r in store.get("tea")] == ["m-1"]


async def test_aget_takes_min_importance(mem_store):
    store, _, mock_async = mem_store
    mock_async.recall.return_value = RecallResponse.from_dict({"memories": []})
    await store.aget("q", min_importance=0.6)
    assert mock_async.recall.call_args.kwargs["min_importance"] == 0.6
