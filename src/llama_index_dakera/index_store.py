"""DakeraIndexStore — LlamaIndex vector store using Dakera server-side embedding."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from dakera import AsyncDakeraClient, DakeraClient
from dakera.models import TextSearchResult
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from pydantic import ConfigDict

# Metadata key holding the id of the source document of a node, for delete(ref_doc_id).
REF_DOC_ID_KEY = "ref_doc_id"

_OPERATORS = {
    FilterOperator.EQ: "$eq",
    FilterOperator.NE: "$ne",
    FilterOperator.GT: "$gt",
    FilterOperator.GTE: "$gte",
    FilterOperator.LT: "$lt",
    FilterOperator.LTE: "$lte",
    FilterOperator.IN: "$in",
    FilterOperator.NIN: "$nin",
    FilterOperator.TEXT_MATCH: "$contains",
    FilterOperator.TEXT_MATCH_INSENSITIVE: "$icontains",
}


def _to_dakera_filter(filters: MetadataFilters) -> dict[str, Any]:
    """Translate LlamaIndex ``MetadataFilters`` into a Dakera filter expression."""
    clauses: list[dict[str, Any]] = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            clauses.append(_to_dakera_filter(f))
        elif f.operator == FilterOperator.IS_EMPTY:
            clauses.append({f.key: {"$exists": False}})
        elif f.operator in _OPERATORS:
            clauses.append({f.key: {_OPERATORS[f.operator]: f.value}})
        else:
            raise ValueError(f"DakeraIndexStore does not support the {f.operator.value!r} filter")
    if filters.condition == FilterCondition.NOT:
        inner = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        return {"$not": inner}
    if len(clauses) == 1:
        return clauses[0]
    return {"$or" if filters.condition == FilterCondition.OR else "$and": clauses}


class DakeraIndexStore(BasePydanticVectorStore):
    """LlamaIndex vector store backed by a Dakera namespace — server-side embedding.

    The Dakera server embeds the nodes' text on write and the query text on search
    (``upsert-text`` / ``query-text``) and indexes the text for BM25 as well, so no
    LlamaIndex embedding model is needed: the store ignores node and query
    embeddings (``is_embedding_query = False``). With ``VectorStoreIndex``, set
    ``Settings.embed_model = None`` so LlamaIndex does not call an embedding API.

    Queries support ``similarity_top_k``, metadata ``filters`` (``==``, ``!=``, ``>``,
    ``>=``, ``<``, ``<=``, ``in``, ``nin``, ``text_match``, ``text_match_insensitive``,
    ``is_empty``; ``and`` / ``or`` / ``not``), ``doc_ids`` and the query modes
    ``default`` (semantic), ``hybrid`` (semantic + BM25 weighted by ``alpha``) and
    ``text_search`` (BM25).

    Args:
        api_url: Dakera server URL, e.g. ``http://localhost:3000``.
        namespace: Namespace holding the nodes (created on the first write).
        api_key: Dakera API key (empty when the server runs without auth).
        embedding_model: Server-side embedding model for writes and queries (e.g.
            ``"bge-large"``, ``"minilm"``); default: the server's / namespace's model.
            Use the same model for every write and query of a namespace.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_url: str
    namespace: str
    api_key: str = ""
    embedding_model: str | None = None
    _client: DakeraClient | None = None
    _async_client: AsyncDakeraClient | None = None
    stores_text: bool = True
    flat_metadata: bool = False
    is_embedding_query: bool = False

    def model_post_init(self, __context: Any) -> None:
        self._client = DakeraClient(self.api_url, api_key=self.api_key)
        self._async_client = AsyncDakeraClient(self.api_url, api_key=self.api_key)

    @property
    def client(self) -> DakeraClient:
        """The underlying synchronous ``dakera.DakeraClient``."""
        if self._client is None:
            raise RuntimeError(
                "DakeraIndexStore: client was not initialized; model_post_init may not have run"
            )
        return self._client

    @property
    def async_client(self) -> AsyncDakeraClient:
        """The underlying ``dakera.AsyncDakeraClient``."""
        if self._async_client is None:
            raise RuntimeError(
                "DakeraIndexStore: client was not initialized; model_post_init may not have run"
            )
        return self._async_client

    @staticmethod
    def _documents(nodes: Sequence[BaseNode]) -> tuple[list[str], list[Any]]:
        ids: list[str] = []
        docs: list[Any] = []
        for node in nodes:
            node_id = node.node_id or str(uuid.uuid4())
            metadata = dict(node.metadata or {})
            if node.ref_doc_id is not None:
                metadata[REF_DOC_ID_KEY] = node.ref_doc_id
            ids.append(node_id)
            docs.append(
                {
                    "id": node_id,
                    "text": node.get_content(metadata_mode=MetadataMode.NONE),
                    "metadata": metadata,
                }
            )
        return ids, docs

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        """Upsert nodes (text and metadata; the server computes the embeddings)."""
        ids, docs = self._documents(nodes)
        if docs:
            self.client.upsert_text(self.namespace, docs, model=self.embedding_model)
        return ids

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        """Async :meth:`add`."""
        ids, docs = self._documents(nodes)
        if docs:
            await self.async_client.upsert_text(self.namespace, docs, model=self.embedding_model)
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete every node of the source document ``ref_doc_id``."""
        self.client.delete(self.namespace, filter={REF_DOC_ID_KEY: {"$eq": ref_doc_id}})

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Async :meth:`delete`."""
        await self.async_client.delete(self.namespace, filter={REF_DOC_ID_KEY: {"$eq": ref_doc_id}})

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete nodes by id and / or metadata filter."""
        if node_ids:
            self.client.delete(self.namespace, ids=list(node_ids))
        if filters is not None:
            self.client.delete(self.namespace, filter=_to_dakera_filter(filters))

    def _filter(self, query: VectorStoreQuery) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if query.filters is not None and query.filters.filters:
            clauses.append(_to_dakera_filter(query.filters))
        if query.doc_ids:
            clauses.append({REF_DOC_ID_KEY: {"$in": list(query.doc_ids)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _semantic(
        self, text: str, top_k: int, flt: dict[str, Any] | None
    ) -> list[TextSearchResult]:
        response = self.client.query_text(
            self.namespace,
            text=text,
            top_k=top_k,
            filter=flt,
            include_text=True,
            model=self.embedding_model,
        )
        return list(response.results)

    def _bm25(self, text: str, top_k: int) -> dict[str, float]:
        # Not filtered: the server's BM25 index of upsert-text documents keeps no
        # metadata, so a filtered full-text query matches nothing; filters are applied
        # through the semantic candidates instead.
        hits = self.client.fulltext_search(self.namespace, query=text, top_k=top_k)
        return {h.id: float(h.score) for h in hits}

    @staticmethod
    def _node(r: TextSearchResult) -> TextNode:
        metadata = dict(r.metadata or {})
        # Back to the node's SOURCE relationship (kept out of the metadata the LLM sees).
        ref_doc_id = metadata.pop(REF_DOC_ID_KEY, None)
        node = TextNode(id_=r.id, text=r.text or "", metadata=metadata)
        if isinstance(ref_doc_id, str):
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref_doc_id)
        return node

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Search the namespace with ``query.query_str``.

        ``mode="default"`` ranks by cosine similarity. ``mode="hybrid"`` takes the
        ``max(4 * k, 20)`` (or ``hybrid_top_k``) semantically nearest nodes and scores
        each ``alpha * cosine + (1 - alpha) * bm25 / best_bm25`` (``alpha`` default
        0.5). ``mode="text_search"`` ranks by BM25 among those candidates.
        """
        if not query.query_str:
            raise ValueError("DakeraIndexStore.query requires a non-empty query_str")
        if query.node_ids:
            raise ValueError("DakeraIndexStore.query does not support node_ids")
        k = query.similarity_top_k or 10
        flt = self._filter(query)
        mode = query.mode
        if mode not in (VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.TEXT_SEARCH):
            if mode != VectorStoreQueryMode.DEFAULT:
                raise ValueError(f"DakeraIndexStore does not support the {mode.value!r} mode")
            results = self._semantic(query.query_str, k, flt)
            return VectorStoreQueryResult(
                nodes=[self._node(r) for r in results],
                ids=[r.id for r in results],
                similarities=[float(r.score) for r in results],
            )

        alpha = 0.0 if mode == VectorStoreQueryMode.TEXT_SEARCH else query.alpha
        alpha = 0.5 if alpha is None else alpha
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        fetch_k = query.hybrid_top_k or max(4 * k, 20)
        pool = self._semantic(query.query_str, fetch_k, flt)
        bm25 = self._bm25(query.query_str, fetch_k if flt is None else 5 * fetch_k)
        best = max((bm25[r.id] for r in pool if r.id in bm25), default=0.0)
        scored = sorted(
            (
                (
                    alpha * max(float(r.score), 0.0)
                    + (1.0 - alpha) * (bm25.get(r.id, 0.0) / best if best > 0 else 0.0),
                    r,
                )
                for r in pool
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:k]
        return VectorStoreQueryResult(
            nodes=[self._node(r) for _, r in scored],
            ids=[r.id for _, r in scored],
            similarities=[score for score, _ in scored],
        )
