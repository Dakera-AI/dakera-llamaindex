"""DakeraMemoryBlock — Dakera as a long-term memory block of LlamaIndex ``Memory``."""

from __future__ import annotations

from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.memory import BaseMemoryBlock
from pydantic import ConfigDict, Field

from llama_index_dakera.memory_store import DakeraMemoryStore

_STORED_ROLES = {MessageRole.USER: "User", MessageRole.ASSISTANT: "Assistant"}


class DakeraMemoryBlock(BaseMemoryBlock[str]):
    """Long-term memory block for LlamaIndex ``Memory``, backed by Dakera.

    On every turn the block recalls the agent's memories most relevant to the
    latest messages (decay-weighted recall on the server) and LlamaIndex inserts
    them into the prompt. Messages that ``Memory`` flushes out of its short-term
    buffer are stored as Dakera memories (user and assistant messages, one memory
    each), so they persist across restarts and processes.

    Example::

        from llama_index.core.memory import Memory
        from llama_index_dakera import DakeraMemoryBlock, DakeraMemoryStore

        store = DakeraMemoryStore(api_url="http://localhost:3000", agent_id="assistant")
        memory = Memory.from_defaults(
            session_id="user-1",
            memory_blocks=[DakeraMemoryBlock(memory_store=store)],
        )

    Args:
        memory_store: The :class:`DakeraMemoryStore` holding the agent's memories.
        name: Name of the block in the prompt (``<DakeraMemory>`` by default).
        similarity_top_k: Memories recalled per turn.
        retrieval_context_window: How many of the latest messages form the query.
        importance: Importance of stored messages (default: the store's
            ``default_importance``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="DakeraMemory", description="The name of the memory block.")
    memory_store: DakeraMemoryStore = Field(description="The Dakera memory store.")
    similarity_top_k: int = Field(default=5, description="Memories recalled per turn.")
    retrieval_context_window: int = Field(
        default=5, description="Latest messages used as the recall query."
    )
    importance: float | None = Field(default=None, description="Importance of stored messages.")

    async def _aget(self, messages: list[ChatMessage] | None = None, **block_kwargs: Any) -> str:
        """Recall the memories relevant to the latest messages, one per line."""
        if not messages:
            return ""
        window = messages[-self.retrieval_context_window :]
        query = " ".join(m.content for m in window if m.content)
        if not query.strip():
            return ""
        results = await self.memory_store.aget(query, top_k=self.similarity_top_k)
        return "\n".join(r["content"] for r in results)

    async def _aput(self, messages: list[ChatMessage]) -> None:
        """Store the user and assistant messages as memories, one each."""
        for message in messages:
            role = _STORED_ROLES.get(message.role)
            if role is None or not message.content:
                continue
            await self.memory_store.aput(f"{role}: {message.content}", importance=self.importance)
