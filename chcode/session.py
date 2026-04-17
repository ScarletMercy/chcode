"""
会话管理 — thread_id, checkpointer DB, 历史会话列表/加载/删除/重命名
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain_core.messages import HumanMessage

console = Console()

_SUMMARY_MAX_LEN = 40


class SessionManager:
    def __init__(self, workplace_path: Path):
        self.workplace_path = workplace_path
        self.sessions_dir = workplace_path / ".chat" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.thread_id = self._new_thread_id()
        self._names_path = self.sessions_dir / "names.json"

    # ─── names.json 读写 ─────────────────────────────────

    def _load_names(self) -> dict[str, str]:
        if self._names_path.exists():
            try:
                return json.loads(self._names_path.read_text("utf-8"))
            except Exception:
                return {}
        return {}

    def _save_names(self, names: dict[str, str]) -> None:
        self._names_path.write_text(
            json.dumps(names, ensure_ascii=False, indent=2), "utf-8"
        )

    # ─── 基础 ────────────────────────────────────────────

    def _new_thread_id(self) -> str:
        return f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def new_session(self) -> None:
        self.thread_id = self._new_thread_id()

    def set_thread(self, thread_id: str) -> None:
        self.thread_id = thread_id

    # ─── 重命名 ──────────────────────────────────────────

    def rename_session(self, thread_id: str, new_name: str) -> None:
        names = self._load_names()
        if new_name:
            names[thread_id] = new_name
        else:
            names.pop(thread_id, None)
        self._save_names(names)

    # ─── 显示名 ──────────────────────────────────────────

    async def _get_summary(
        self, agent: CompiledStateGraph, thread_id: str
    ) -> str | None:
        cfg = {"configurable": {"thread_id": thread_id}}
        try:
            state = await agent.aget_state(cfg)
            messages = state.values.get("messages", [])
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    content = msg.content
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, str):
                                parts.append(part)
                            elif isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                        text = "".join(parts)
                    else:
                        continue
                    text = text.strip().replace("\n", " ")
                    if text:
                        return text[:_SUMMARY_MAX_LEN] + (
                            "…" if len(text) > _SUMMARY_MAX_LEN else ""
                        )
        except Exception:
            pass
        return None

    async def get_display_names(
        self,
        thread_ids: list[str],
        agent: CompiledStateGraph,
    ) -> dict[str, str]:
        names = self._load_names()
        result: dict[str, str] = {}
        need_summary: list[str] = []

        for tid in thread_ids:
            if tid in names:
                result[tid] = names[tid]
            else:
                need_summary.append(tid)

        if need_summary:
            summaries = await asyncio.gather(
                *[self._get_summary(agent, tid) for tid in need_summary]
            )
            for tid, summary in zip(need_summary, summaries):
                result[tid] = summary or tid

        return result

    # ─── 列表 / 删除 ─────────────────────────────────────

    async def list_sessions(self, checkpointer: AsyncSqliteSaver) -> list[str]:
        """从 checkpointer 获取所有历史 thread_id"""
        try:
            await checkpointer.setup()
            async with checkpointer.lock:
                rows = await checkpointer.conn.execute_fetchall(
                    "SELECT DISTINCT thread_id FROM checkpoints"
                )
            return [row[0] for row in rows if row[0]]
        except Exception:
            return []

    async def delete_session(
        self, thread_id: str, checkpointer: AsyncSqliteSaver
    ) -> bool:
        """删除指定会话的所有数据"""
        try:
            await checkpointer.adelete_thread(thread_id)
            names = self._load_names()
            if thread_id in names:
                del names[thread_id]
                self._save_names(names)
            return True
        except Exception as e:
            console.print(f"[red]删除会话失败: {e}[/red]")
            return False
