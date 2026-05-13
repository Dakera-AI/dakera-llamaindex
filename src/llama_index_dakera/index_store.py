"""DakeraIndexStore — LlamaIndex vector store using Dakera server-side embedding."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from dakera import AsyncDakeraClient, DakeraClient
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from pydantic import ConfigDict


class DakeraIndexStore(BasePydanticVectorStore):
    """LlamaIndex vector store backed by Dakera AI — server-side embedding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_url: str
    namespace: str
    api_key: str = ""
    _client: DakeraClient | None = None
    _async_client: AsyncDakeraClient | None = None
    stores_text: bool = True
    flat_metadata: bool = False

    def model_post_init(self, __context: Any) -> None:
        self._client = DakeraClient(self.api_url, api_key=self.api_key)
        self._async_client = AsyncDakeraClient(self.api_url, api_key=self.api_key)

    @property
    def client(self) -> DakeraClient:
        if self._client is None:
            raise RuntimeError(
                "DakeraIndexStore: client was not initialized; model_post_init may not have run"
            )
        return self._client

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        docs: list[dict[str, Any]] = []
        ids: list[str] = []
        for node in nodes:
            node_id = node.node_id or str(uuid.uuid4())
            ids.append(node_id)
            docs.append({"id": node_id, "text": node.get_content(metadata_mode=MetadataMode.NONE),
                         "metadata": node.metadata or {}})
        if docs:
            self.client.upsert_text(self.namespace, docs)
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        self.client.delete(self.namespace, filter={"ref_doc_id": {"$eq": ref_doc_id}})

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if not query.query_str:
            raise ValueError("DakeraIndexStore.query requires a non-empty query_str")
        response = self.client.query_text(self.namespace, text=query.query_str,
                                          top_k=query.similarity_top_k or 10, include_text=True)
        nodes, ids, similarities = [], [], []
        for r in response.results:
            nodes.append(TextNode(node_id=r.id, text=r.text or "", metadata=r.metadata or {}))
            ids.append(r.id)
            similarities.append(r.score)
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=similarities)
