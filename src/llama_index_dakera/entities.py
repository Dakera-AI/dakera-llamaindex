"""DakeraEntityExtractor — named entity extraction for LlamaIndex."""

from __future__ import annotations

from typing import Any

from dakera import DakeraClient


class DakeraEntityExtractor:
    """Entity extraction with Dakera's NER pipeline, for one agent.

    The server combines a rule-based extractor with GLiNER zero-shot NER.
    :meth:`configure` turns extraction on for the agent's memories; the entities
    of each stored memory are then read with :meth:`memory_entities`.
    """

    def __init__(self, api_url: str, agent_id: str, api_key: str = "") -> None:
        self._client = DakeraClient(api_url, api_key=api_key)
        self._agent_id = agent_id

    def extract(self, text: str, entity_types: list[str] | None = None) -> list[dict[str, Any]]:
        """Extract entities from text (``POST /v1/memories/extract``, needs a Write key).

        Without ``entity_types`` the server uses its rule-based extractor (and GLiNER
        only if its model is already cached); naming types, e.g. ``["person",
        "organization", "location"]``, loads GLiNER on demand (a ~782 MB download on
        first use). Returns ``[{"type", "value", "score"}, ...]``.
        """
        result = self._client.extract_entities(text, entity_types=entity_types)
        return [
            {"type": e.entity_type, "value": e.value, "score": e.score} for e in result.entities
        ]

    def memory_entities(self, memory_id: str) -> list[dict[str, Any]]:
        """Entities extracted from a stored memory: ``[{"type", "value", "score"}, ...]``."""
        result = self._client.memory_entities(memory_id)
        return [
            {"type": e.entity_type, "value": e.value, "score": e.score} for e in result.entities
        ]

    def configure(
        self, entity_types: list[str] | None = None, *, extract_entities: bool = True
    ) -> None:
        """Configure entity extraction on this agent's memory namespace.

        The agent's memories live in ``_dakera_agent_{agent_id}``; that is the
        namespace whose extraction settings apply when memories are stored.
        ``entity_types=None`` keeps the configured types.
        """
        self._client.configure_namespace_ner(
            f"_dakera_agent_{self._agent_id}",
            extract_entities=extract_entities,
            entity_types=entity_types,
        )
