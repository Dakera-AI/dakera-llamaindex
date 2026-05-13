# Changelog

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
