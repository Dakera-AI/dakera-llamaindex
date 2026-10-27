# Changelog

## [0.4.0] - 2026-10-27

Dakera server **v0.12.3** support. Requires the `dakera` Python SDK **0.14.1** or later; works
with Dakera server v0.12.0 – v0.12.3 (the v0.12.3 additions are ignored by older servers).

### Added
- `DakeraMemoryStore.put(..., event_time=)` / `aput(..., event_time=)`: when the memory's event
  happened (Unix seconds, an RFC 3339 string or a timezone-aware `datetime`); the server keeps it
  as `_dakera_event_time` and uses it for recency and `since` / `until`.
- `DakeraMemoryStore.get(..., rerank=)` / `aget(..., rerank=)`; `get()` / `aget()` results carry
  `importance`, and `score` is the server's `rank_score` when it sends one.
- `DakeraMemoryStore.consolidate(memory_ids=None, *, target_type=None)` merges the given memories.
- `DakeraSessionManager.start(..., idle_timeout_secs=)`, `touch()` and `end(summary=None)`: the
  server ends a session after 4 h without activity by default. `get()` and `list()` return
  `ended_reason`; `get()` also returns the session `summary`.
- `tests/test_integration.py`: live tests against a server, skipped unless `DAKERA_API_URL` is set.

### Changed
- Dependency: `dakera>=0.14.1` (was `>=0.13.1`).
- `DakeraMemoryStore.get()`, `aget()` and `batch_get()` return at most `top_k` ranked memories. A
  v0.12.3 server appends a session's neighbouring memories to the results by default, so a recall
  could return more than `top_k` and mix unranked context in; recalls now ask for
  `neighborhood_mode="separate"` and keep the ranked results only (a v0.12.2 server's inline
  context is cut at `top_k`).
- `DakeraSessionManager.memories()` leaves out the sentence sub-memories the server derives from
  each stored memory (tagged `dakera-curated`), which `GET /v1/sessions/{id}/memories` lists next
  to them; `include_derived=True` keeps them.

### Fixed
- `DakeraEntityExtractor.configure()` raised `TypeError` (it did not send the `extract_entities`
  the SDK requires) and targeted a namespace named after the agent id. It now configures the
  agent's memory namespace `_dakera_agent_{agent_id}` and takes `extract_entities` (default `True`).
- Examples `advanced_memory.py` and `batch_operations.py` used queries that recall nothing on a
  v0.12 server (`advanced_memory.py` would also have read an `importance` the results lacked).
- README: the `DakeraMemoryStore` option is `recall_k`, not `top_k`.

### CI
- Integration tests and examples run against `ghcr.io/dakera-ai/dakera:0.12.3`. That image ships
  no `curl`, so the service container has no in-container health check; the job polls `/health`
  from the runner instead.

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
