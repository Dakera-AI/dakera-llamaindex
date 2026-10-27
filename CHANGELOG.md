# Changelog

## [0.4.0] - 2026-10-27

Dakera server **v0.12.3** support. Requires the `dakera` Python SDK **0.14.1** or later and
`llama-index-core` **0.12.35** or later; works with Dakera server v0.12.0 – v0.12.3 (the v0.12.3
additions are ignored by older servers).

### Added
- `DakeraMemoryBlock`: Dakera as a long-term memory block of LlamaIndex `Memory` (and so of its
  agents and chat engines). Each turn it recalls the memories relevant to the latest messages
  into the prompt; messages flushed from the short-term buffer are stored in Dakera.
- `DakeraIndexStore`: metadata `filters` (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `nin`,
  `text_match`, `text_match_insensitive`, `is_empty`; `and` / `or` / `not`) and `doc_ids` on
  queries; the `hybrid` (semantic + BM25 weighted by `alpha`) and `text_search` (BM25) query
  modes; `delete_nodes(node_ids=, filters=)`; `async_add` / `adelete` on the async client;
  `embedding_model`; `async_client` property.
- `DakeraMemoryStore.put(..., event_time=)` / `aput(..., event_time=)`: when the memory's event
  happened (Unix seconds, an RFC 3339 string or a timezone-aware `datetime`); the server keeps it
  as `_dakera_event_time` and uses it for recency and `since` / `until`.
- `DakeraMemoryStore.get(..., rerank=)` / `aget(..., rerank=, min_importance=)`; `get()`,
  `aget()` and `batch_get()` results carry `importance` and `metadata`, and `score` is the
  server's `rank_score` when it sends one.
- `DakeraMemoryStore.consolidate(memory_ids=None, *, target_type=None)` merges the given memories.
- `DakeraSessionManager.start(..., idle_timeout_secs=)`, `touch()` and `end(summary=None)`: the
  server ends a session after 4 h without activity by default. `get()` and `list()` return
  `ended_reason`; `get()` also returns the session `summary`.
- `tests/test_integration.py`: live tests against a server, skipped unless `DAKERA_API_URL` is set.
- `examples/memory_block.py`.

### Changed
- Dependencies: `dakera>=0.14.1` (was `>=0.13.1`), `llama-index-core>=0.12.35` (was `>=0.10.0`;
  the release that added `Memory` blocks).
- `DakeraMemoryStore.get()`, `aget()` and `batch_get()` return at most `top_k` ranked memories. A
  v0.12.3 server can append a session's neighbouring memories to the results, so a recall could
  return more than `top_k` and mix unranked context in; recalls now ask for
  `neighborhood_mode="separate"` and keep the ranked results only (a v0.12.2 server's inline
  context is cut at `top_k`).
- Recalls no longer repeat a text: the server may recall the sentence sub-memories it derives
  from a stored memory next to the memory itself, and of a memory and its derived sentences only
  the best ranked is kept.
- `DakeraIndexStore` sets `is_embedding_query = False`: retrievers send the query text and the
  server embeds it, so no LlamaIndex embedding model is called (with `Settings.embed_model =
  None`, `VectorStoreIndex` needs no embedding API at all).
- `DakeraIndexStore` stores each node's source document id as `ref_doc_id` metadata and returns
  it as the node's `SOURCE` relationship.
- `DakeraKnowledgeGraph.query()` / `export()` return each edge `type` as a plain string
  (`"linked_by"`), not the SDK's enum.
- `DakeraSessionManager.memories()` leaves out the sentence sub-memories the server derives from
  each stored memory (tagged `dakera-curated`), which `GET /v1/sessions/{id}/memories` lists next
  to them; `include_derived=True` keeps them.
- Examples use a fresh agent / namespace per run, show the 0.4.0 features (`DakeraMemoryBlock`,
  `VectorStoreIndex` on Dakera with filters and the hybrid / BM25 modes, `event_time`, filtered
  and reranked recall, consolidate, idle sessions and `touch`) and need no LLM API key. README and
  CONTRIBUTING rewritten for LlamaIndex 0.12 / 0.14, with every component documented and every
  snippet run against a v0.12.3 server.

### Fixed
- The README's agent example passed `DakeraMemoryStore` as a `ChatMemoryBuffer` chat store, which
  LlamaIndex rejects (it is not a `BaseChatStore`), and used `ReActAgent.from_tools`, which
  LlamaIndex 0.14 no longer has; `DakeraMemoryBlock` with `Memory` replaces it.
- `DakeraIndexStore.delete(ref_doc_id)` (and `VectorStoreIndex.delete_ref_doc`) deleted nothing:
  it filtered on a `ref_doc_id` metadata key that `add()` never stored.
- `DakeraIndexStore` ignored the query's metadata `filters` and `doc_ids`, and the
  `embedding_model` the README documented did not exist.
- `DakeraEntityExtractor.configure()` raised `TypeError` (it did not send the `extract_entities`
  the SDK requires) and targeted a namespace named after the agent id. It now configures the
  agent's memory namespace `_dakera_agent_{agent_id}` and takes `extract_entities` (default `True`).
- Examples `advanced_memory.py` and `batch_operations.py` used queries that recall nothing on a
  v0.12 server (`advanced_memory.py` would also have read an `importance` the results lacked).
- README: the `DakeraMemoryStore` option is `recall_k`, not `top_k`.
- Docstrings: `DakeraKnowledgeGraph.deduplicate()` merges duplicate memories (not entities);
  `consolidate()` names the memories the server refuses to consolidate.
- Added the MIT `LICENSE` file the README links to.

### CI
- Integration tests and examples run against `ghcr.io/dakera-ai/dakera:0.12.3`. That image ships
  no `curl`, so the service container has no in-container health check; the job polls `/health`
  from the runner instead.
- `ruff format --check .` runs in CI; the whole repository is formatted.
- The examples job is no longer `continue-on-error`: a broken example fails CI.

## [0.3.0] - 2026-10-01

Dakera server **v0.12.0** support. Requires the `dakera` Python SDK **0.13.1** or later;
works with Dakera server v0.12.0 and v0.11.108.

### Changed
- Dependency: `dakera>=0.13.1` (was `>=0.8.6`).
- `DakeraKnowledgeGraph.link(source_id, target_id, *, label=None)` sends the agent id the server
  requires (every link was a 422 before) and returns the server's `{from_id, to_id, edge_type}`.
  The `edge_type` argument is gone: the server records every explicit link as `linked_by`.
- `DakeraKnowledgeGraph.summarize(memory_ids, *, target_type=None)` needs at least two ids and has
  no `dry_run`: the server has no dry run and always stores the summary (dakera 0.13.1 raises on
  `dry_run=True` rather than silently writing).
- `DakeraKnowledgeGraph.build(memory_id, depth=None)`: the seed `memory_id` is required by the server.
- `DakeraMemoryStore.get(tags=...)` / `aget(tags=...)` work: dakera 0.13.1 `recall()` accepts
  `tags` (it raised `TypeError` before).

### Fixed
- `DakeraSessionManager.list()` unwraps the `{"sessions": [...], "total": n}` answer of
  `GET /v1/sessions`; it iterated over the envelope's keys and raised `AttributeError`.
- The `knowledge_graph.py` example summarizes memories it stores (it called `summarize()`
  without memory ids and failed with a 422).
- CI: integration and example jobs run against `ghcr.io/dakera-ai/dakera:0.12.0`; mypy runs
  against each matrix interpreter (numpy 2.5 stubs use 3.12-only syntax that a pinned 3.10
  target cannot parse); the security audit upgrades `setuptools` and documents the ignored
  nltk advisory PYSEC-2026-3740, which has no fixed release yet.

### Tests
- `tests/test_sdk_surface.py` checks the SDK calls with `create_autospec(DakeraClient)`, so
  a method or argument the SDK does not have fails the test.

## [0.1.1] - 2026-05-13

### Fixed
- **Pydantic v2 compatibility**: migrated from `class Config` to `model_config = ConfigDict(...)` in `DakeraIndexStore`
- **API alignment**: replaced deprecated `node_id=` with `id_=` in `TextNode` constructor
- **Improved error handling**: `assert` statements converted to `RuntimeError` for clearer debugging in production
- **Removed unused import** in `index_store.py`
- **Test fix**: corrected `VectorStoreQuery` default `top_k` assertion

### Changed
- Bumped GitHub Actions: `actions/checkout` v4 → v6, `actions/setup-python` v5 → v6

### Added
- **5 new unit tests** for `DakeraIndexStore` covering store/load/delete/list operations
- Community health files: `CONTRIBUTING.md`, `SECURITY.md`, issue templates, PR template

## [0.1.0] - 2026-05-13

### Added
- Initial release — LlamaIndex integration for Dakera AI memory platform
- `DakeraIndexStore` class for document index persistence over Dakera memory
- PyPI publish via OIDC Trusted Publisher
