"""DakeraMemoryStore — LlamaIndex agent memory backed by Dakera AI."""

from __future__ import annotations

from typing import Any

from dakera import AsyncDakeraClient, DakeraClient


class DakeraMemoryStore:
    """Persistent agent memory store for LlamaIndex agents.

    Supports memory types, tags, TTL, batch operations, hybrid search,
    sessions, and importance scoring.
    """

    def __init__(
        self,
        api_url: str,
        agent_id: str,
        api_key: str = "",
        recall_k: int = 5,
        min_importance: float = 0.0,
        default_importance: float = 0.7,
    ) -> None:
        self._client = DakeraClient(api_url, api_key=api_key)
        self._async_client = AsyncDakeraClient(api_url, api_key=api_key)
        self.agent_id = agent_id
        self.recall_k = recall_k
        self.min_importance = min_importance
        self.default_importance = default_importance

    def put(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        tags: list[str] | None = None,
        ttl_seconds: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Store a memory with full parameter control."""
        kwargs: dict[str, Any] = {
            "memory_type": memory_type,
            "importance": importance if importance is not None else self.default_importance,
        }
        if metadata:
            kwargs["metadata"] = metadata
        if tags:
            kwargs["tags"] = tags
        if ttl_seconds is not None:
            kwargs["ttl_seconds"] = ttl_seconds
        if session_id:
            kwargs["session_id"] = session_id
        return self._client.store_memory(self.agent_id, content=content, **kwargs)

    def get(
        self,
        query: str,
        top_k: int | None = None,
        *,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_importance: float | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic recall with optional filtering."""
        k = top_k if top_k is not None else self.recall_k
        min_imp = (
            min_importance
            if min_importance is not None
            else (self.min_importance if self.min_importance > 0 else None)
        )
        kwargs: dict[str, Any] = {"top_k": k}
        if min_imp:
            kwargs["min_importance"] = min_imp
        if tags:
            kwargs["tags"] = tags
        if memory_type:
            kwargs["memory_type"] = memory_type
        response = self._client.recall(self.agent_id, query=query, **kwargs)
        return [
            {"content": m.content, "id": m.id, "score": m.score, "tags": m.tags}
            for m in response.memories
        ]

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Combined vector + BM25 search."""
        k = top_k if top_k is not None else self.recall_k
        result = self._client.search_memories(self.agent_id, query=query, top_k=k, alpha=alpha)
        return [{"content": m.content, "id": m.id, "score": m.score} for m in result.memories]

    def batch_get(self, queries: list[str], top_k: int | None = None) -> list[list[dict[str, Any]]]:
        """Run multiple recall queries in batch."""
        k = top_k if top_k is not None else self.recall_k
        results = []
        for q in queries:
            response = self._client.recall(self.agent_id, query=q, top_k=k)
            results.append(
                [{"content": m.content, "id": m.id, "score": m.score} for m in response.memories]
            )
        return results

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._client.forget(self.agent_id, memory_id)

    def batch_delete(self, memory_ids: list[str]) -> None:
        """Delete multiple memories."""
        self._client.batch_forget(self.agent_id, memory_ids=memory_ids)

    def update_importance(self, memory_id: str, importance: float) -> None:
        """Update a memory's importance score."""
        self._client.update_importance(self.agent_id, memory_id=memory_id, importance=importance)

    def consolidate(self) -> Any:
        """Deduplicate and consolidate memories."""
        return self._client.consolidate(self.agent_id)

    def stats(self) -> dict[str, Any]:
        """Get agent memory statistics."""
        return self._client.agent_stats(self.agent_id)

    async def aput(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        tags: list[str] | None = None,
        ttl_seconds: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Async store a memory."""
        kwargs: dict[str, Any] = {
            "memory_type": memory_type,
            "importance": importance if importance is not None else self.default_importance,
        }
        if metadata:
            kwargs["metadata"] = metadata
        if tags:
            kwargs["tags"] = tags
        if ttl_seconds is not None:
            kwargs["ttl_seconds"] = ttl_seconds
        if session_id:
            kwargs["session_id"] = session_id
        return await self._async_client.store_memory(self.agent_id, content=content, **kwargs)

    async def aget(
        self,
        query: str,
        top_k: int | None = None,
        *,
        tags: list[str] | None = None,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Async semantic recall with filtering."""
        k = top_k if top_k is not None else self.recall_k
        kwargs: dict[str, Any] = {"top_k": k}
        if self.min_importance > 0:
            kwargs["min_importance"] = self.min_importance
        if tags:
            kwargs["tags"] = tags
        if memory_type:
            kwargs["memory_type"] = memory_type
        response = await self._async_client.recall(self.agent_id, query=query, **kwargs)
        return [
            {"content": m.content, "id": m.id, "score": m.score, "tags": m.tags}
            for m in response.memories
        ]

    def __repr__(self) -> str:
        return f"DakeraMemoryStore(agent_id={self.agent_id!r}, recall_k={self.recall_k})"
