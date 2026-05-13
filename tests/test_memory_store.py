"""Tests for DakeraMemoryStore (LlamaIndex integration)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index_dakera import DakeraMemoryStore


@pytest.fixture
def mem_store():
    with patch("llama_index_dakera.memory_store.DakeraClient") as MC, \
         patch("llama_index_dakera.memory_store.AsyncDakeraClient") as AMC:
        mock_client = MagicMock()
        mock_async = AsyncMock()
        MC.return_value = mock_client
        AMC.return_value = mock_async
        store = DakeraMemoryStore(api_url="http://localhost:3000", api_key="test",
                                  agent_id="agent-1", recall_k=3)
        store._client = mock_client
        store._async_client = mock_async
        yield store, mock_client, mock_async


def test_put_stores_memory(mem_store):
    store, mock_client, _ = mem_store
    store.put("The user prefers Python")
    mock_client.store_memory.assert_called_once_with(
        "agent-1", content="The user prefers Python", memory_type="episodic",
        importance=0.7, metadata=None)


def test_get_recalls_memories(mem_store):
    store, mock_client, _ = mem_store
    mem = MagicMock(content="The user prefers Python", id="m-1", score=0.9)
    mock_recall = MagicMock()
    mock_recall.memories = [mem]
    mock_client.recall.return_value = mock_recall
    results = store.get("What does the user prefer?")
    assert len(results) == 1
    assert results[0]["content"] == "The user prefers Python"
    mock_client.recall.assert_called_once_with(
        "agent-1", query="What does the user prefer?", top_k=3, min_importance=None)


def test_delete_calls_forget(mem_store):
    store, mock_client, _ = mem_store
    store.delete("mem-123")
    mock_client.forget.assert_called_once_with("agent-1", "mem-123")
