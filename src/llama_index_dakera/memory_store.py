"""DakeraMemoryStore — LlamaIndex agent memory backed by Dakera AI."""

from __future__ import annotations

from typing import Any

from dakera import AsyncDakeraClient, DakeraClient


class DakeraMemoryStore:
    """Persistent agent memory store for LlamaIndex agents."""

    def __init__(self, api_url: str, agent_id: str, api_key: str = "",
                 recall_k: int = 5, min_importance: float = 0.0,
                 default_importance: float = 0.7) -> None:
        self._client = DakeraClient(api_url, api_key=api_key)
        self._async_client = AsyncDakeraClient(api_url, api_key=api_key)
        self.agent_id = agent_id
        self.recall_k = recall_k
        self.min_importance = min_importance
        self.default_importance = default_importance

    def put(self, content: str, memory_type: str = "episodic",
            importance: float | None = None,
            metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._client.store_memory(
            self.agent_id, content=content, memory_type=memory_type,
            importance=importance if importance is not None else self.default_importance,
            metadata=metadata)

    def get(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self._client.recall(
            self.agent_id, query=query,
            top_k=top_k if top_k is not None else self.recall_k,
            min_importance=self.min_importance if self.min_importance > 0 else None)

    def delete(self, memory_id: str) -> None:
        self._client.forget(self.agent_id, memory_id)

    async def aput(self, content: str, memory_type: str = "episodic",
                   importance: float | None = None,
                   metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._async_client.store_memory(
            self.agent_id, content=content, memory_type=memory_type,
            importance=importance if importance is not None else self.default_importance,
            metadata=metadata)

    async def aget(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return await self._async_client.recall(
            self.agent_id, query=query,
            top_k=top_k if top_k is not None else self.recall_k,
            min_importance=self.min_importance if self.min_importance > 0 else None)

    def __repr__(self) -> str:
        return f"DakeraMemoryStore(agent_id={self.agent_id!r}, recall_k={self.recall_k})"
