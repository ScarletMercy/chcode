from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


DEFAULT_MAX_RESULT_CHARS = 50_000
MAX_RESULTS_PER_TURN_CHARS = 200_000
PREVIEW_MAX_CHARS = 2_000
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"


def clean_tool_output(text: str) -> str:
    if not text:
        return text
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    elif not isinstance(text, str):
        text = str(text)
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    text = re.sub(r"\x1b\][^\x07]*\x07?", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def _content_size(content: str) -> int:
    return len(content.encode("utf-8"))


def _generate_preview(
    content: str, max_chars: int = PREVIEW_MAX_CHARS
) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_newline = truncated.rfind("\n")
    cut_point = last_newline if last_newline > max_chars * 0.5 else max_chars
    return content[:cut_point], True


def _persist_to_disk(
    content: str, tool_use_id: str, workplace: Path | None
) -> str | None:
    if workplace is None:
        return None
    try:
        result_dir = workplace / ".chat" / "tool-results"
        result_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_use_id)
        filepath = result_dir / f"{safe_id}.txt"
        filepath.write_text(content, encoding="utf-8")
        return str(filepath)
    except Exception:
        return None


def truncate_large_result(
    content: str,
    tool_name: str = "",
    tool_use_id: str = "",
    workplace: Path | None = None,
    threshold: int = DEFAULT_MAX_RESULT_CHARS,
) -> str:
    if not content or not content.strip():
        if content is not None and content != "" and content.strip() == "":
            return f"({tool_name} completed with no output)"
        return content

    size = _content_size(content)
    if size <= threshold:
        return content

    filepath = _persist_to_disk(content, tool_use_id, workplace)
    preview, has_more = _generate_preview(content)
    size_str = f"{size / 1024:.1f}KB" if size >= 1024 else f"{size}B"

    if filepath:
        message = (
            f"{PERSISTED_OUTPUT_TAG}\n"
            f"Output too large ({size_str}). Full output saved to: {filepath}\n\n"
            f"Preview (first {PREVIEW_MAX_CHARS} chars):\n"
            f"{preview}"
        )
        if has_more:
            message += "\n..."
        message += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
        return message

    fallback_preview = content[:threshold]
    message = (
        f"{PERSISTED_OUTPUT_TAG}\n"
        f"Output too large ({size_str}), truncated to {threshold} chars.\n\n"
        f"{fallback_preview}\n"
        f"...{PERSISTED_OUTPUT_CLOSING_TAG}"
    )
    return message


class BudgetState:
    def __init__(self) -> None:
        self.seen_ids: set[str] = set()
        self.replacements: dict[str, str] = {}

    def reset(self) -> None:
        self.seen_ids.clear()
        self.replacements.clear()


_budget_state: BudgetState | None = None


def get_budget_state() -> BudgetState:
    global _budget_state
    if _budget_state is None:
        _budget_state = BudgetState()
    return _budget_state


def reset_budget_state() -> None:
    global _budget_state
    _budget_state = BudgetState()


def _collect_tool_messages_by_turn(
    messages: list[BaseMessage],
) -> list[list[tuple[int, BaseMessage]]]:
    from langchain_core.messages import AIMessage, ToolMessage

    turns: list[list[tuple[int, BaseMessage]]] = []
    current_turn: list[tuple[int, BaseMessage]] = []

    for idx, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if current_turn:
                turns.append(current_turn)
            current_turn = [(idx, msg)]
        elif isinstance(msg, ToolMessage):
            current_turn.append((idx, msg))
        else:
            if current_turn:
                turns.append(current_turn)
                current_turn = []

    if current_turn:
        turns.append(current_turn)

    return turns


def _select_to_replace(
    fresh: list[tuple[int, BaseMessage]],
    frozen_size: int,
    limit: int,
) -> list[tuple[int, BaseMessage]]:
    fresh_total = sum(_content_size(m.content or "") for _, m in fresh)
    if frozen_size + fresh_total <= limit:
        return []
    deficit = frozen_size + fresh_total - limit
    sorted_fresh = sorted(
        fresh, key=lambda x: _content_size(x[1].content or ""), reverse=True
    )
    selected: list[tuple[int, BaseMessage]] = []
    reclaimed = 0
    for item in sorted_fresh:
        selected.append(item)
        reclaimed += _content_size(item[1].content or "")
        if reclaimed >= deficit:
            break
    return selected


def enforce_per_turn_budget(
    messages: list[BaseMessage],
    budget: int = MAX_RESULTS_PER_TURN_CHARS,
    workplace: Path | None = None,
    state: BudgetState | None = None,
) -> list[BaseMessage]:
    from langchain_core.messages import ToolMessage

    if not any(isinstance(m, ToolMessage) for m in messages):
        return messages

    if state is None:
        state = get_budget_state()

    turns = _collect_tool_messages_by_turn(messages)
    replacement_map: dict[int, str] = {}

    for turn in turns:
        fresh: list[tuple[int, BaseMessage]] = []
        frozen_size = 0

        for idx, msg in turn:
            if not isinstance(msg, ToolMessage):
                continue
            tool_use_id = msg.tool_call_id or ""
            if tool_use_id in state.seen_ids:
                if tool_use_id in state.replacements:
                    replacement_map[idx] = state.replacements[tool_use_id]
                else:
                    frozen_size += _content_size(msg.content or "")
            else:
                fresh.append((idx, msg))

        if not fresh:
            continue

        selected = _select_to_replace(fresh, frozen_size, budget)

        non_selected = [item for item in fresh if item not in selected]
        for idx, msg in non_selected:
            state.seen_ids.add(msg.tool_call_id or "")

        for idx, msg in selected:
            tool_use_id = msg.tool_call_id or ""
            state.seen_ids.add(tool_use_id)
            content = msg.content or ""
            result = truncate_large_result(
                content,
                msg.name or "",
                tool_use_id,
                workplace=workplace,
            )
            replacement_map[idx] = result
            state.replacements[tool_use_id] = result

    if not replacement_map:
        return messages

    result = []
    for idx, msg in enumerate(messages):
        if idx in replacement_map:
            result.append(msg.model_copy(update={"content": replacement_map[idx]}))
        else:
            result.append(msg)
    return result
