# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.2] - 2026-06-20

### Features

- Config-driven multimodal detection: `is_multimodal_model` reads from `vision_model.json` instead of hardcoded patterns; new `add_vision_model()` with idempotent upsert (default/fallback roles)

### Refactored

- Per-model `context_length` in `metadata` (default 1M) replaces hardcoded `CONTEXT_WINDOW_SIZES` dict and `get_context_window_size()`
- `model_config_form` simplified: Base URL via `text()` instead of select; streamlined API key flow (keep/re-enter)
- `configure_new_model` prompts for context length before multimodal question; new `skip_method_select` param
- `edit_current_model` preserves existing `metadata` on form resubmit
- Subagent architecture: default Explore mode, parallel display rework, read-only safety, cross-session conflict fix
- LangSmith cross-platform config: Linux shell config file support; stderr guard suppresses errors instead of auto-disabling tracing
- Replace readline with prompt-toolkit `FileHistory` (50-entry cap); use native `prompt_async`
- Extract shared utilities, unify console/API key masking, add IPC event middleware
- `chat.py`: rewrite if/elif chains to match/case, remove dead code

### Fixed

- `configure_langsmith` project name `None` handling (Ctrl-C fallback to "chcode")
- Remove unused `error` marker from `additional_kwargs`

### Documentation

- Update README for subagent mode-aware features and project directory structure
