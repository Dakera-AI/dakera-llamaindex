"""DakeraMemoryStore — LlamaIndex agent memory backed by Dakera AI."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dakera import AsyncDakeraClient, DakeraClient, RecalledMemory, RecallResponse


def _ranked(response: RecallResponse, top_k: int) -> list[RecalledMemory]:
    """The ranked results of a recall, at most ``top_k`` of them.

    Recalls ask for ``neighborhood_mode="separate"``, so a v0.12.3+ server keeps the
    session neighbourhood context out of ``memories``; a server up to v0.12.2
    appends it inline (unmarked), so the list is cut at ``top_k`` too.
    """
    return response.ranked[:top_k]


def _result(m: RecalledMemory) -> dict[str, Any]:
    return {
        "content": m.content,
        "id": m.id,
        "score": m.score,
        "importance": m.importance,
        "metadata": m.metadata,
    }


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
        event_time: int | float | str | datetime | None = None,
    ) -> dict[str, Any]:
        """Store a memory with full parameter control.

        ``event_time`` is when the memory's event happened (Unix seconds, an RFC 3339
        string or a timezone-aware ``datetime``); the server (v0.12.3+) uses it for
        recency and ``since`` / ``until`` instead of the ingest time.
        """
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
        if event_time is not None:
            kwargs["event_time"] = event_time
        return self._client.store_memory(self.agent_id, content=content, **kwargs)

    def get(
        self,
        query: str,
        top_k: int | None = None,
        *,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_importance: float | None = None,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic recall with optional filtering.

        Returns at most ``top_k`` (default ``recall_k``) ranked memories, best first;
        ``score`` is the server's ranking score (``rank_score`` on v0.12.3+).
        ``rerank=False`` skips the cross-encoder reranking (server default: on).
        """
        k = top_k if top_k is not None else self.recall_k
        min_imp = (
            min_importance
            if min_importance is not None
            else (self.min_importance if self.min_importance > 0 else None)
        )
        kwargs: dict[str, Any] = {"top_k": k, "neighborhood_mode": "separate"}
        if min_imp:
            kwargs["min_importance"] = min_imp
        if tags:
            kwargs["tags"] = tags
        if memory_type:
            kwargs["memory_type"] = memory_type
        if rerank is not None:
            kwargs["rerank"] = rerank
        response = self._client.recall(self.agent_id, query=query, **kwargs)
        return [_result(m) for m in _ranked(response, k)]

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Combined vector + BM25 search via routing."""
        k = top_k if top_k is not None else self.recall_k
        results = self._client.search_memories(self.agent_id, query=query, top_k=k)
        return [
            {"content": m.get("content", ""), "id": m.get("id", ""), "score": m.get("score", 0.0)}
            for m in results
        ]

    def batch_get(self, queries: list[str], top_k: int | None = None) -> list[list[dict[str, Any]]]:
        """Run multiple recall queries in batch."""
        k = top_k if top_k is not None else self.recall_k
        results = []
        for q in queries:
            response = self._client.recall(
                self.agent_id, query=q, top_k=k, neighborhood_mode="separate"
            )
            results.append(
                [{"content": m.content, "id": m.id, "score": m.score} for m in _ranked(response, k)]
            )
        return results

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._client.forget(self.agent_id, memory_id)

    def batch_delete(self, memory_ids: list[str]) -> None:
        """Delete multiple memories."""
        for mid in memory_ids:
            self._client.forget(self.agent_id, mid)

    def update_importance(self, memory_id: str, importance: float) -> None:
        """Update a memory's importance score."""
        self._client.update_importance(self.agent_id, [memory_id], importance)

    def consolidate(
        self, memory_ids: list[str] | None = None, *, target_type: str | None = None
    ) -> Any:
        """Merge memories into one new memory; the merged sources are forgotten.

        ``memory_ids`` picks the memories (at least two); without it the server
        merges up to 5 memories of the agent's most common type. ``target_type`` is
        the memory type of the merged memory.
        """
        kwargs: dict[str, Any] = {}
        if memory_ids is not None:
            if len(memory_ids) < 2:
                raise ValueError("consolidate() needs at least two memory ids")
            kwargs["memory_ids"] = memory_ids
        if target_type is not None:
            kwargs["target_type"] = target_type
        return self._client.consolidate(self.agent_id, **kwargs)

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
        event_time: int | float | str | datetime | None = None,
    ) -> dict[str, Any]:
        """Async store a memory (see :meth:`put`)."""
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
        if event_time is not None:
            kwargs["event_time"] = event_time
        return await self._async_client.store_memory(self.agent_id, content=content, **kwargs)

    async def aget(
        self,
        query: str,
        top_k: int | None = None,
        *,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Async semantic recall with filtering (see :meth:`get`)."""
        k = top_k if top_k is not None else self.recall_k
        kwargs: dict[str, Any] = {"top_k": k, "neighborhood_mode": "separate"}
        if self.min_importance > 0:
            kwargs["min_importance"] = self.min_importance
        if tags:
            kwargs["tags"] = tags
        if memory_type:
            kwargs["memory_type"] = memory_type
        if rerank is not None:
            kwargs["rerank"] = rerank
        response = await self._async_client.recall(self.agent_id, query=query, **kwargs)
        return [_result(m) for m in _ranked(response, k)]

    def __repr__(self) -> str:
        return f"DakeraMemoryStore(agent_id={self.agent_id!r}, recall_k={self.recall_k})"
