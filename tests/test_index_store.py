"""Tests for DakeraIndexStore (LlamaIndex integration)."""

from unittest.mock import MagicMock, patch

import pytest

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
