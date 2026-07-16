# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.4] - 2026-07-16

### Features
- **Shadow Repo Isolation**: Moved git checkpoint system from real `.git` to independent `.chat/cp-repo`, completely avoiding user repo pollution
- **Legacy Checkpoint Migration**: Added `migrate_legacy_git` method to auto-detect and migrate `.git/checkpoints.json` to new location (idempotent execution)
- **Fork Question Backfill**: When forking from any message, that message's human prompt is automatically filled into the input box for regeneration
- **No-Changes Detection Optimization**: Uses `diff --cached --quiet` instead of post-commit change detection for better accuracy

### Fixes
- **Rollback KeyError**: Changed to use `.get()` for `init` checkpoint to avoid KeyError when init is missing
- **Idempotent Context Update**: `_init_git` checks `cp_repo_dir` existence before calling `migrate_legacy_git` for idempotent behavior
- **Skip Commit When No Human**: `_post_process` now checks if human messages exist before calling git commit to prevent false positives
- **Shadow Init Warning**: Added explicit warning (`chat.git.shadow_init_exception`) when `init_shadow` fails
- **Init Commit with Initial Files**: `init_shadow` now `git add .` to include initial project files in init commit, preserving original content on rollback

### Refactored
- **Simplified State Tracking**: Removed `_git_cp_count` tracking mechanism; status bar now shows static Git indicator
- **Enhanced Error Feedback**: All git operation failure paths now include detailed stderr alerts
- **Idempotent Exclude File Management**: Added `_ensure_exclude()` method to automatically detect and fix info/exclude file content
- **Hook Cleanup**: Added `_clean_hooks` to delete user's active git hooks (except .sample) during migration to prevent accidental triggers

### Documentation
- Updated README.md and README_en.md, rewriting Git Integration section to 'Message Management & Shadow Repo (Git-aware)'
- Adjusted feature descriptions to highlight shadow repo isolation advantages

## [0.1.3] - 2026-07-03

### Features

- Internationalization (i18n) with zh/en language support: `--lang` flag, first-run wizard, `/lang` command
- Localized tool descriptions in `/tools` command output (14 `tool_desc.*` keys)
- Welcome title localized via i18n (`display.welcome_title`), avoids duplicate "ChCode — " prefix

### Refactored

- Consolidate `strings/` package into single `strings.py` with inline {zh,en} per key
- `manage_skills` / `_list_skills` / `_install_skill` / `_delete_skill` accept `workplace_path: Path` instead of `SessionManager`
- Remove `TYPE_CHECKING` and `SessionManager` dependency from skill_manager module
- Remove commented-out ZAI/DASHSCOPE dead code from `ENV_TO_CONFIG`

### Fixed

- Tool description fallback: detect missing `t()` translation (returns key itself) instead of showing raw key string
- `_app_version` wraps `_pkg_version` with try/except fallback for missing package metadata
- Remove unused imports (`TYPE_CHECKING`, `re`, `yaml`, `os`)

### Cleanup

- Provider labels in prompts.py use hardcoded English; removed obsolete i18n keys (`provider.bigmodel`, `provider.qwen`, `chat.skill.init_first`)

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