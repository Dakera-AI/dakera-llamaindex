# Changelog

## [0.3.0] - 2026-10-01

Dakera server **v0.12.0** support. Requires the `dakera` Python SDK **0.13.0** or later;
works with Dakera server v0.12.0 and v0.11.108.

### Changed
- Dependency: `dakera>=0.13.0` (was `>=0.8.6`).
- `DakeraKnowledgeGraph.summarize()` now takes the `memory_ids` to summarize (plus
  `target_type` and `dry_run`). `POST /v1/knowledge/summarize` requires them, so the old
  no-argument call always failed with a 422. An empty list raises `ValueError`.

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
