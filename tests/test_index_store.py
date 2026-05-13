"""Tests for DakeraIndexStore (LlamaIndex integration)."""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from llama_index_dakera import DakeraIndexStore


@pytest.fixture
def index_store():
    with patch("llama_index_dakera.index_store.DakeraClient") as MC, \
         patch("llama_index_dakera.index_store.AsyncDakeraClient"):
        mock_client = MagicMock()
        MC.return_value = mock_client
        store = DakeraIndexStore(
            api_url="http://localhost:3000", api_key="test", namespace="test-ns"
        )
        store._client = mock_client
        yield store, mock_client


def test_delete_calls_client(index_store):
    store, mock_client = index_store
    store.delete("doc-abc")
    mock_client.delete.assert_called_once_with("test-ns", filter={"ref_doc_id": {"$eq": "doc-abc"}})


def test_add_upserts_nodes(index_store):
    store, mock_client = index_store
    node = TextNode(node_id="node-1", text="Hello world", metadata={"source": "test"})
    ids = store.add([node])
    assert ids == ["node-1"]
    mock_client.upsert_text.assert_called_once()
    call_args = mock_client.upsert_text.call_args
    assert call_args[0][0] == "test-ns"
    docs = call_args[0][1]
    assert len(docs) == 1
    assert docs[0]["id"] == "node-1"
    assert docs[0]["text"] == "Hello world"


def test_add_generates_id_when_missing(index_store):
    store, mock_client = index_store
    node = TextNode(text="No explicit ID")
    node.node_id = ""
    ids = store.add([node])
    assert len(ids) == 1
    assert ids[0]  # non-empty UUID generated


def test_add_empty_nodes_skips_upsert(index_store):
    store, mock_client = index_store
    ids = store.add([])
    assert ids == []
    mock_client.upsert_text.assert_not_called()


def test_query_returns_nodes_and_scores(index_store):
    store, mock_client = index_store
    mock_result = MagicMock()
    mock_result.id = "node-1"
    mock_result.text = "Hello world"
    mock_result.metadata = {"source": "test"}
    mock_result.score = 0.95
    mock_client.query_text.return_value = MagicMock(results=[mock_result])

    q = VectorStoreQuery(query_str="hello", similarity_top_k=4)
    result = store.query(q)

    assert len(result.nodes) == 1
    assert result.ids == ["node-1"]
    assert result.similarities == [0.95]
    assert result.nodes[0].node_id == "node-1"
    assert result.nodes[0].text == "Hello world"
    mock_client.query_text.assert_called_once_with(
        "test-ns", text="hello", top_k=4, include_text=True
    )


def test_query_uses_default_top_k(index_store):
    store, mock_client = index_store
    mock_client.query_text.return_value = MagicMock(results=[])
    q = VectorStoreQuery(query_str="hello")
    store.query(q)
    call_kwargs = mock_client.query_text.call_args[1]
    assert call_kwargs["top_k"] == 10  # default when similarity_top_k is None
