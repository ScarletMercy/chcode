"""
会话管理 — thread_id, checkpointer DB, 历史会话列表/加载/删除
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


class SessionManager:
    def __init__(self, workplace_path: Path):
        self.workplace_path = workplace_path
        self.sessions_dir = workplace_path / ".chat" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.sessions_dir / "checkpointer.db"
        self.thread_id = self._new_thread_id()

    def _new_thread_id(self) -> str:
        return f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def new_session(self) -> None:
        self.thread_id = self._new_thread_id()

    def set_thread(self, thread_id: str) -> None:
        self.thread_id = thread_id

    def list_sessions(self) -> list[str]:
        """从 checkpointer.db 获取所有历史 thread_id"""
        if not self.db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = cursor.fetchall()
            conn.close()
            return [row[0] for row in rows]
        except Exception:
            return []

    def delete_session(self, thread_id: str) -> bool:
        """删除指定会话的所有数据"""
        if not self.db_path.exists():
            return False
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoint_writes_v2"):
                try:
                    cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
                except sqlite3.OperationalError:
                    pass
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            console.print(f"[red]删除会话失败: {e}[/red]")
            return False
