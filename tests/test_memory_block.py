"""Tests for DakeraMemoryBlock (LlamaIndex ``Memory`` long-term block)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dakera import RecallResponse
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import Memory

from llama_index_dakera import DakeraMemoryBlock, DakeraMemoryStore


@pytest.fixture
def block():
    with (
        patch("llama_index_dakera.memory_store.DakeraClient") as MC,
        patch("llama_index_dakera.memory_store.AsyncDakeraClient") as AMC,
    ):
        MC.return_value = MagicMock()
        mock_async = AsyncMock()
        AMC.return_value = mock_async
        store = DakeraMemoryStore(api_url="http://localhost:3000", agent_id="agent-1")
    yield DakeraMemoryBlock(memory_store=store, similarity_top_k=2), mock_async


async def test_aget_recalls_for_the_latest_messages(block):
    blk, client = block
    client.recall.return_value = RecallResponse.from_dict(
        {"memories": [{"id": "m1", "content": "User: I build chatbots"}]}
    )
    messages = [ChatMessage(role=MessageRole.USER, content=f"msg {i}") for i in range(7)]
    assert await blk.aget(messages) == "User: I build chatbots"
    call = client.recall.call_args
    assert call.kwargs["query"] == "msg 2 msg 3 msg 4 msg 5 msg 6"  # the last 5 messages
    assert call.kwargs["top_k"] == 2


async def test_aget_without_messages_recalls_nothing(block):
    blk, client = block
    assert await blk.aget([]) == ""
    client.recall.assert_not_called()


async def test_aput_stores_user_and_assistant_messages(block):
    blk, client = block
    await blk.aput(
        [
            ChatMessage(role=MessageRole.SYSTEM, content="be nice"),
            ChatMessage(role=MessageRole.USER, content="I build chatbots"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Cool!"),
        ]
    )
    contents = [c.kwargs["content"] for c in client.store_memory.call_args_list]
    assert contents == ["User: I build chatbots", "Assistant: Cool!"]


async def test_plugs_into_llamaindex_memory(block):
    blk, client = block
    client.recall.return_value = RecallResponse.from_dict(
        {"memories": [{"id": "m1", "content": "User: my project is NeuralBridge"}]}
    )
    memory = Memory.from_defaults(session_id="s1", memory_blocks=[blk])
    await memory.aput(ChatMessage(role=MessageRole.USER, content="What's my project?"))
    messages = await memory.aget()
    assert "NeuralBridge" in messages[0].content
