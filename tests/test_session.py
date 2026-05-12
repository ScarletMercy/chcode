from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from chcode.utils.session import SessionManager


@pytest.fixture
def sm(tmp_path: Path):
    return SessionManager(tmp_path)


class TestSessionManager:
    def test_init_creates_dir(self, tmp_path: Path):
        sm = SessionManager(tmp_path)
        assert (tmp_path / ".chat" / "sessions").exists()

    def test_thread_id_format(self, sm: SessionManager):
        assert sm.thread_id.startswith("thread_")

    def test_config(self, sm: SessionManager):
        cfg = sm.config
        assert "thread_id" in cfg["configurable"]

    def test_new_session(self, sm: SessionManager):
        old = sm.thread_id
        sm.new_session()
        assert sm.thread_id != old

    def test_set_thread(self, sm: SessionManager):
        sm.set_thread("custom_id")
        assert sm.thread_id == "custom_id"


class TestNames:
    def test_load_names_empty(self, sm: SessionManager):
        assert sm._load_names() == {}

    def test_save_and_load_session(self, sm: SessionManager):
        sm._save_names({"t1": "My Session"})
        assert sm._load_names() == {"t1": "My Session"}

    def test_rename(self, sm: SessionManager):
        sm.rename_session("t1", "New Name")
        assert sm._load_names()["t1"] == "New Name"

    def test_rename_clear(self, sm: SessionManager):
        sm.rename_session("t1", "Name")
        sm.rename_session("t1", "")
        assert "t1" not in sm._load_names()

    def test_load_names_invalid_json(self, sm: SessionManager):
        sm._names_path.write_text("not json{{{")
        assert sm._load_names() == {}


class TestGetDisplayNames:
    async def test_with_custom_name(self, sm: SessionManager):
        sm.rename_session("t1", "My Session")
        agent = MagicMock()
        result = await sm.get_display_names(["t1"], agent)
        assert result["t1"] == "My Session"

    async def test_fallback_to_summary(self, sm: SessionManager):
        agent = AsyncMock()
        state = MagicMock()
        state.values = {"messages": [MagicMock(content="Hello world", __class__=type("H", (), {}))]}
        agent.aget_state = AsyncMock(return_value=state)

        from langchain_core.messages import HumanMessage

        state.values = {"messages": [HumanMessage(content="Hello world")]}
        agent.aget_state = AsyncMock(return_value=state)

        result = await sm.get_display_names(["t_unknown"], agent)
        assert "t_unknown" in result

    async def test_fallback_to_thread_id(self, sm: SessionManager):
        agent = AsyncMock()
        state = MagicMock()
        state.values = {"messages": []}
        agent.aget_state = AsyncMock(return_value=state)

        result = await sm.get_display_names(["t_empty"], agent)
        assert result["t_empty"] == "t_empty"


class TestListSessions:
    async def test_list_sessions(self, sm: SessionManager):
        checkpointer = AsyncMock()
        checkpointer.setup = AsyncMock()
        checkpointer.lock = MagicMock()
        checkpointer.lock.__aenter__ = AsyncMock(return_value=None)
        checkpointer.lock.__aexit__ = AsyncMock(return_value=None)
        checkpointer.conn.execute_fetchall = AsyncMock(
            return_value=[("thread_1",), ("thread_2",)]
        )
        result = await sm.list_sessions(checkpointer)
        assert len(result) == 2

    async def test_list_sessions_error(self, sm: SessionManager):
        checkpointer = AsyncMock()
        checkpointer.setup = AsyncMock(side_effect=Exception("db error"))
        result = await sm.list_sessions(checkpointer)
        assert result == []


class TestDeleteSession:
    async def test_delete(self, sm: SessionManager):
        sm.rename_session("t1", "name")
        checkpointer = AsyncMock()
        checkpointer.adelete_thread = AsyncMock()
        result = await sm.delete_session("t1", checkpointer)
        assert result is True
        assert "t1" not in sm._load_names()

    async def test_delete_error(self, sm: SessionManager):
        checkpointer = AsyncMock()
        checkpointer.adelete_thread = AsyncMock(side_effect=Exception("fail"))
        result = await sm.delete_session("t1", checkpointer)
        assert result is False


class TestLoadNamesIOError:
    def test_io_error_propagates(self, sm: SessionManager, monkeypatch):
        sm._names_path.write_text('{"t1": "name"}', encoding="utf-8")
        from unittest.mock import patch
        with patch.object(Path, "read_text", side_effect=PermissionError("locked")):
            with pytest.raises(PermissionError):
                sm._load_names()


class TestRenameSessionIOError:
    def test_rename_does_not_overwrite_on_io_error(self, sm: SessionManager):
        sm.rename_session("t1", "existing")
        sm.rename_session("t2", "another")
        original = sm._load_names

        def _load_fail():
            raise PermissionError("locked")

        sm._load_names = _load_fail
        sm.rename_session("t3", "should_not_save")
        sm._load_names = original
        assert sm._load_names() == {"t1": "existing", "t2": "another"}
