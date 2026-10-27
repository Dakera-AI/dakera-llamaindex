"""DakeraKnowledgeGraph — knowledge graph operations for LlamaIndex."""

from __future__ import annotations

from typing import Any

from dakera import DakeraClient


def _edge_type(value: Any) -> str:
    """The edge type as a plain string (the SDK returns an ``EdgeType`` enum)."""
    return str(getattr(value, "value", value))


class DakeraKnowledgeGraph:
    """The knowledge graph of one agent's memories.

    Memories are linked automatically (similarity, time, shared entities) and
    explicitly with :meth:`link`; :meth:`query`, :meth:`find_path`, :meth:`build`
    and :meth:`export` read the graph, :meth:`summarize` and :meth:`deduplicate`
    act on the memories.
    """

    def __init__(self, api_url: str, agent_id: str, api_key: str = "") -> None:
        self._client = DakeraClient(api_url, api_key=api_key)
        self._agent_id = agent_id

    def query(
        self,
        root_id: str | None = None,
        edge_type: str | None = None,
        max_depth: int = 3,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Edges reachable from ``root_id`` (or all), up to ``max_depth`` hops.

        Returns ``{"edges": [{"source", "target", "type"}, ...], "node_count", "edge_count"}``.
        """
        result = self._client.knowledge_query(
            self._agent_id, root_id=root_id, edge_type=edge_type, max_depth=max_depth, limit=limit
        )
        return {
            "edges": [
                {"source": e.source_id, "target": e.target_id, "type": _edge_type(e.edge_type)}
                for e in result.edges
            ],
            "node_count": result.node_count,
            "edge_count": result.edge_count,
        }

    def find_path(self, from_id: str, to_id: str) -> dict[str, Any]:
        """Find shortest path between two memory nodes."""
        result = self._client.knowledge_path(self._agent_id, from_id=from_id, to_id=to_id)
        return {"path": result.path, "hop_count": result.hop_count}

    def link(self, source_id: str, target_id: str, *, label: str | None = None) -> dict[str, Any]:
        """Link two of this agent's memories (``POST /v1/memories/{id}/links``).

        The server records every explicit link as ``linked_by``; ``label`` is
        an optional human-readable note stored with it.
        """
        result = self._client.memory_link(
            source_id, target_id, agent_id=self._agent_id, label=label
        )
        return {
            "from_id": result.from_id,
            "to_id": result.to_id,
            "edge_type": result.edge_type,
        }

    def export(self) -> dict[str, Any]:
        """Every edge of the agent's graph, in the shape :meth:`query` returns."""
        result = self._client.knowledge_export(self._agent_id)
        return {
            "edges": [
                {"source": e.source_id, "target": e.target_id, "type": _edge_type(e.edge_type)}
                for e in result.edges
            ],
            "node_count": result.node_count,
            "edge_count": result.edge_count,
        }

    def build(self, memory_id: str, depth: int | None = None) -> dict[str, Any]:
        """Build the knowledge graph around one seed memory.

        ``POST /v1/knowledge/graph`` needs ``memory_id``.
        """
        return self._client.knowledge_graph(self._agent_id, memory_id=memory_id, depth=depth)

    def summarize(self, memory_ids: list[str], *, target_type: str | None = None) -> dict[str, Any]:
        """Summarize the given memories into one new summary memory.

        ``POST /v1/knowledge/summarize`` needs at least two ``memory_ids`` and
        always stores the summary (the server has no dry run); it answers
        ``{"summary_memory", "source_count"}``.
        """
        if len(memory_ids) < 2:
            raise ValueError("summarize() needs at least two memory ids")
        return self._client.summarize(
            self._agent_id, memory_ids=memory_ids, target_type=target_type
        )

    def deduplicate(self) -> dict[str, Any]:
        """Find the agent's duplicate memories and merge them (``POST /v1/knowledge/deduplicate``).

        Returns ``{"groups", "duplicates_found", "duplicates_merged", ...}``.
        """
        return self._client.deduplicate(self._agent_id)
