"""DakeraMemoryStore — LlamaIndex agent memory backed by Dakera AI."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dakera import (
    AsyncDakeraClient,
    DakeraClient,
    ErrorCode,
    NotFoundError,
    RecalledMemory,
    RecallResponse,
)


def _agent_has_no_memories(err: NotFoundError) -> bool:
    """Whether a recall / search failed only because the agent has stored nothing yet.

    A server before v0.12.3 answers a recall or search of an agent that has no memories
    with ``404 NAMESPACE_NOT_FOUND`` for some queries (the agent's namespace is created
    by its first stored memory); v0.12.3 returns no memories.
    """
    return err.code == ErrorCode.NAMESPACE_NOT_FOUND


def _ranked(response: RecallResponse | None, top_k: int) -> list[RecalledMemory]:
    """The ranked results of a recall, at most ``top_k`` of them.

    Recalls ask for ``neighborhood_mode="separate"``, so a v0.12.3+ server keeps the
    session neighbourhood context out of ``memories``; a server up to v0.12.2
    appends it inline (unmarked), so the list is cut at ``top_k`` too.

    The server also derives sentence sub-memories from each stored memory and may
    recall them next to it: of a memory and the sentences derived from it, only the
    best ranked is kept, so the results do not repeat a text.
    """
    kept: list[RecalledMemory] = []
    seen: set[str] = set()
    if response is None:
        return kept
    for m in response.ranked[:top_k]:
        # A memory and the sentences derived from it share one key: its id.
        key = (m.metadata or {}).get("source_memory_id") or m.id
        if key not in seen:
            seen.add(key)
            kept.append(m)
    return kept


def _result(m: RecalledMemory) -> dict[str, Any]:
    return {
        "content": m.content,
        "id": m.id,
        "score": m.score,
        "importance": m.importance,
        "metadata": m.metadata,
    }


class DakeraMemoryStore:
    """Persistent memory of one agent, stored on a Dakera server.

    :meth:`put` stores a memory (with type, importance, tags, TTL, session and event
    time); :meth:`get` recalls the memories most relevant to a query (decay-weighted
    by importance, recency and access on the server); :meth:`hybrid_search`,
    :meth:`batch_get`, :meth:`delete`, :meth:`update_importance`, :meth:`consolidate`
    and :meth:`stats` complete it, with async :meth:`aput` / :meth:`aget`. Use it
    with LlamaIndex ``Memory`` through :class:`DakeraMemoryBlock`.

    Args:
        api_url: Dakera server URL, e.g. ``http://localhost:3000``.
        agent_id: The agent whose memories are stored and recalled.
        api_key: Dakera API key (empty when the server runs without auth).
        recall_k: Memories :meth:`get` returns by default.
        min_importance: Recall only memories at least this important (``0`` = all).
        default_importance: Importance (0-1) of stored memories.
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
        """Store a memory; returns the server's answer (with the new memory's ``id``).

        ``memory_type`` is ``episodic``, ``semantic``, ``procedural`` or ``working``.
        On a v0.12.3+ server the answer also carries ``truncated`` / ``embedded_tokens``
        when the content is longer than the embedding model reads, and ``warnings``
        naming ignored request fields. ``event_time`` is when the memory's event
        happened (Unix seconds, an RFC 3339 string or a timezone-aware ``datetime``);
        the server (v0.12.3+) uses it for recency and ``since`` / ``until`` instead of
        the ingest time.
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
        """Recall the memories most relevant to ``query``, best first.

        Returns at most ``top_k`` (default ``recall_k``) memories as
        ``[{"content", "id", "score", "importance", "metadata"}, ...]``; ``score`` is
        the server's ranking score (``rank_score`` on v0.12.3+). ``tags`` keeps
        memories carrying at least one of them; ``rerank=False`` skips the
        cross-encoder reranking (server default: on).
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
        return [_result(m) for m in _ranked(self._recall(query, **kwargs), k)]

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search the agent's memories with the server's memory search, best first.

        ``POST /v1/memory/search`` combines vector similarity and BM25 on the server;
        returns ``[{"content", "id", "score"}, ...]``.
        """
        k = top_k if top_k is not None else self.recall_k
        try:
            results = self._client.search_memories(self.agent_id, query=query, top_k=k)
        except NotFoundError as err:
            if not _agent_has_no_memories(err):
                raise
            return []
        return [
            {"content": m.get("content", ""), "id": m.get("id", ""), "score": m.get("score", 0.0)}
            for m in results
        ]

    def batch_get(self, queries: list[str], top_k: int | None = None) -> list[list[dict[str, Any]]]:
        """Recall for several queries; one list per query (as :meth:`get` returns)."""
        k = top_k if top_k is not None else self.recall_k
        results = []
        for q in queries:
            response = self._recall(q, top_k=k, neighborhood_mode="separate")
            results.append([_result(m) for m in _ranked(response, k)])
        return results

    def _recall(self, query: str, **kwargs: Any) -> RecallResponse | None:
        """Recall for this agent; ``None`` when it has stored nothing yet (older servers)."""
        try:
            return self._client.recall(self.agent_id, query=query, **kwargs)
        except NotFoundError as err:
            if not _agent_has_no_memories(err):
                raise
            return None

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
        the memory type of the merged memory. The server refuses (``400``) memories
        with a TTL, an attachment or user metadata (``metadata`` or ``event_time``).
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
        """Agent statistics: ``total_memories``, ``memories_by_type``, sessions, ..."""
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
        min_importance: float | None = None,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Async :meth:`get`."""
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
        response: RecallResponse | None
        try:
            response = await self._async_client.recall(self.agent_id, query=query, **kwargs)
        except NotFoundError as err:
            if not _agent_has_no_memories(err):
                raise
            response = None
        return [_result(m) for m in _ranked(response, k)]

    def __repr__(self) -> str:
        return f"DakeraMemoryStore(agent_id={self.agent_id!r}, recall_k={self.recall_k})"
