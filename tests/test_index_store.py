"""Tests for DakeraIndexStore (LlamaIndex integration)."""

from unittest.mock import MagicMock, patch

import pytest
from dakera.models import FullTextSearchResult, TextSearchResult
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index_dakera import DakeraIndexStore


@pytest.fixture
def index_store():
    with (
        patch("llama_index_dakera.index_store.DakeraClient") as MC,
        patch("llama_index_dakera.index_store.AsyncDakeraClient"),
    ):
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
    node = TextNode(id_="node-1", text="Hello world", metadata={"source": "test"})
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
        "test-ns", text="hello", top_k=4, filter=None, include_text=True, model=None
    )


def test_query_uses_default_top_k(index_store):
    store, mock_client = index_store
    mock_client.query_text.return_value = MagicMock(results=[])
    q = VectorStoreQuery(query_str="hello", similarity_top_k=None)
    store.query(q)
    call_kwargs = mock_client.query_text.call_args[1]
    assert call_kwargs["top_k"] == 10  # default when similarity_top_k is None


def _results(*rows):
    return MagicMock(
        results=[TextSearchResult(id=i, score=sc, text=t, metadata=md) for i, sc, t, md in rows]
    )


def test_store_needs_no_query_embedding(index_store):
    store, _ = index_store
    assert store.is_embedding_query is False
    assert store.stores_text is True


def test_add_records_the_source_document_for_delete(index_store):
    store, mock_client = index_store
    node = TextNode(id_="n1", text="chunk", metadata={"page": 1})
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="doc-1")
    store.add([node])
    doc = mock_client.upsert_text.call_args[0][1][0]
    assert doc["metadata"] == {"page": 1, "ref_doc_id": "doc-1"}


def test_query_restores_the_source_document(index_store):
    store, mock_client = index_store
    mock_client.query_text.return_value = _results(("n1", 0.9, "chunk", {"ref_doc_id": "doc-1"}))
    node = store.query(VectorStoreQuery(query_str="q", similarity_top_k=1)).nodes[0]
    assert node.ref_doc_id == "doc-1"
    assert node.metadata == {}


def test_embedding_model_is_sent():
    with (
        patch("llama_index_dakera.index_store.DakeraClient") as MC,
        patch("llama_index_dakera.index_store.AsyncDakeraClient"),
    ):
        client = MagicMock()
        MC.return_value = client
        store = DakeraIndexStore(
            api_url="http://localhost:3000", namespace="ns", embedding_model="minilm"
        )
    client.query_text.return_value = _results()
    store.add([TextNode(id_="n1", text="x")])
    store.query(VectorStoreQuery(query_str="x"))
    assert client.upsert_text.call_args.kwargs["model"] == "minilm"
    assert client.query_text.call_args.kwargs["model"] == "minilm"


def test_metadata_filters_are_translated(index_store):
    store, mock_client = index_store
    mock_client.query_text.return_value = _results()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="lang", value="py"),
                    MetadataFilter(key="lang", value=["go", "rs"], operator=FilterOperator.IN),
                ],
                condition=FilterCondition.OR,
            ),
        ]
    )
    store.query(VectorStoreQuery(query_str="q", filters=filters, doc_ids=["d1"]))
    assert mock_client.query_text.call_args.kwargs["filter"] == {
        "$and": [
            {
                "$and": [
                    {"year": {"$gte": 2020}},
                    {"$or": [{"lang": {"$eq": "py"}}, {"lang": {"$in": ["go", "rs"]}}]},
                ]
            },
            {"ref_doc_id": {"$in": ["d1"]}},
        ]
    }


def test_unsupported_filter_operator_raises(index_store):
    store, _ = index_store
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tags", value=["a"], operator=FilterOperator.ALL)]
    )
    with pytest.raises(ValueError, match="all"):
        store.query(VectorStoreQuery(query_str="q", filters=filters))


def test_hybrid_mode_fuses_semantic_and_bm25(index_store):
    store, mock_client = index_store
    mock_client.query_text.return_value = _results(
        ("n1", 0.8, "semantic", None), ("n2", 0.6, "keyword", None)
    )
    mock_client.fulltext_search.return_value = [FullTextSearchResult(id="n2", score=3.0)]
    result = store.query(
        VectorStoreQuery(
            query_str="q", similarity_top_k=2, mode=VectorStoreQueryMode.HYBRID, alpha=0.25
        )
    )
    assert result.ids == ["n2", "n1"]  # 0.25*0.6 + 0.75*1.0 = 0.9 > 0.25*0.8 = 0.2
    assert result.similarities == pytest.approx([0.9, 0.2])
    assert mock_client.query_text.call_args.kwargs["top_k"] == 20
    mock_client.fulltext_search.assert_called_once_with("test-ns", query="q", top_k=20)


def test_text_search_mode_ranks_by_bm25(index_store):
    store, mock_client = index_store
    mock_client.query_text.return_value = _results(("n1", 0.9, "a", None), ("n2", 0.1, "b", None))
    mock_client.fulltext_search.return_value = [FullTextSearchResult(id="n2", score=1.0)]
    result = store.query(
        VectorStoreQuery(query_str="q", similarity_top_k=1, mode=VectorStoreQueryMode.TEXT_SEARCH)
    )
    assert result.ids == ["n2"]


def test_delete_nodes_by_id_and_filter(index_store):
    store, mock_client = index_store
    store.delete_nodes(
        node_ids=["n1"], filters=MetadataFilters(filters=[MetadataFilter(key="k", value="v")])
    )
    assert mock_client.delete.call_args_list[0].args == ("test-ns",)
    assert mock_client.delete.call_args_list[0].kwargs == {"ids": ["n1"]}
    assert mock_client.delete.call_args_list[1].kwargs == {"filter": {"k": {"$eq": "v"}}}
