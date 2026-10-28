"""Dakera as long-term memory of a LlamaIndex agent.

``DakeraMemoryBlock`` plugs Dakera into LlamaIndex ``Memory``: on every turn
it recalls the memories relevant to the latest messages and LlamaIndex puts
them into the prompt; messages flushed out of the short-term buffer are stored
in Dakera. Pass the memory to an agent: ``await agent.run(msg, memory=memory)``.
This example needs no LLM: it shows what the agent would receive.

Run a Dakera server first (see the README), then:

    pip install llamaindex-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/memory_block.py
"""

import asyncio
import os
import uuid

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import Memory

from llama_index_dakera import DakeraMemoryBlock, DakeraMemoryStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")


async def main() -> None:
    store = DakeraMemoryStore(
        api_url=api_url,
        api_key=api_key,
        agent_id=f"llamaindex-block-{uuid.uuid4().hex[:8]}",
    )
    block = DakeraMemoryBlock(memory_store=store, similarity_top_k=3)

    # Earlier conversation, already in Dakera (here stored directly).
    await block.aput(
        [
            ChatMessage(role=MessageRole.USER, content="My project is called NeuralBridge."),
            ChatMessage(role=MessageRole.ASSISTANT, content="Nice name! Noted."),
        ]
    )

    # A new session: short-term memory starts empty, Dakera remembers.
    memory = Memory.from_defaults(session_id="user-1", memory_blocks=[block])
    await memory.aput(ChatMessage(role=MessageRole.USER, content="What's my project called?"))

    print("Messages the agent receives:")
    for message in await memory.aget():
        print(f"--- {message.role.value} ---\n{message.content.strip()}")


asyncio.run(main())
