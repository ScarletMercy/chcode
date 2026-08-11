import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from chcode.agent_setup import (
    State,
    build_agent,
    create_checkpointer,
    emit_tool_events,
    handle_tool_errors,
    tool_call_storm_block,
)


def _dump(value) -> str:
    """与中间件内部 json.dumps 规范化保持一致的辅助函数。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


# ─── helpers ──────────────────────────────────────────────


def _snapshot(name: str, args: str, result: str) -> dict:
    return {
        "name": name,
        "args_key": args,
        "result_key": result,
    }


def _tracker(
    previous: dict,
    *,
    exact: int = 1,
    varying_result: int = 1,
    varying_args: int = 1,
) -> dict:
    return {
        "previous": previous,
        "exact_streak": exact,
        "varying_result_streak": varying_result,
        "varying_args_streak": varying_args,
    }


async def _call_guard(
    state: dict,
    *,
    name: str,
    args: dict,
    content: str,
) -> tuple[Command, AsyncMock]:
    """通过中间件执行一次工具调用，并把返回的 tracker 写回 state。"""
    handler = AsyncMock(
        return_value=ToolMessage(content=content, tool_call_id="call")
    )
    request = SimpleNamespace(
        state=state,
        tool_call={"name": name, "args": args, "id": "call"},
    )

    result = await tool_call_storm_block.awrap_tool_call(request, handler)

    assert isinstance(result, Command)
    state["tool_storm"] = result.update["tool_storm"]
    return result, handler


def _blocked_message(result: Command) -> ToolMessage:
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    return message


def _repeated_call(index: int, *, mode: str) -> tuple[str, dict, str]:
    """构造第 index 次工具调用的 (name, args, content)，复现三种风暴模式。"""
    if mode == "exact":
        return "read_file", {"file_path": "same.py"}, "same result"
    if mode == "varying_result":
        return "web_fetch", {"url": "https://example.test/data"}, f"response-{index}"
    return "grep", {"pattern": f"query-{index}"}, f"result-{index}"


def _make_request(
    state: dict,
    *,
    name: str,
    args: dict,
    tool_call_id: str = "call",
) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        tool_call={"name": name, "args": args, "id": tool_call_id},
    )


# ─── State schema ─────────────────────────────────────────


def test_tool_storm_state_is_optional():
    """tool_storm 字段是 NotRequired，不传也能正常工作。"""
    agent = create_agent(
        FakeListChatModel(responses=["ok"]),
        [],
        state_schema=State,
    )
    schema = agent.get_input_schema()

    value = schema.model_validate(
        {"messages": [HumanMessage(content="hello")]}
    )

    assert value is not None


# ─── tracker 累加逻辑（放行后 next_tracker 各字段）─────────


async def test_first_call_tracker_is_all_ones():
    """首次调用后 tracker 的三个 streak 都是 1。"""
    state: dict = {}
    await _call_guard(
        state,
        name="read_file",
        args={"file_path": "x.py"},
        content="result",
    )

    tracker = state["tool_storm"]
    assert tracker["exact_streak"] == 1
    assert tracker["varying_result_streak"] == 1
    assert tracker["varying_args_streak"] == 1


async def test_same_args_same_result_advances_exact_streak():
    """同工具+同参+同果：只有 exact_streak +1，其余重置为 1。"""
    state: dict = {}
    # 第一次
    await _call_guard(state, name="read_file", args={"file_path": "x.py"}, content="r")
    # 第二次：完全相同
    await _call_guard(state, name="read_file", args={"file_path": "x.py"}, content="r")

    tracker = state["tool_storm"]
    assert tracker["exact_streak"] == 2
    assert tracker["varying_result_streak"] == 1
    assert tracker["varying_args_streak"] == 1


async def test_same_args_different_result_advances_varying_result_streak():
    """同工具+同参+异果：只有 varying_result_streak +1，其余重置为 1。"""
    state: dict = {}
    await _call_guard(state, name="web_fetch", args={"url": "u"}, content="r1")
    await _call_guard(state, name="web_fetch", args={"url": "u"}, content="r2")

    tracker = state["tool_storm"]
    assert tracker["exact_streak"] == 1
    assert tracker["varying_result_streak"] == 2
    assert tracker["varying_args_streak"] == 1


async def test_different_args_advances_varying_args_streak():
    """同工具+异参：只有 varying_args_streak +1，其余重置为 1。"""
    state: dict = {}
    await _call_guard(state, name="grep", args={"pattern": "a"}, content="r1")
    await _call_guard(state, name="grep", args={"pattern": "b"}, content="r2")

    tracker = state["tool_storm"]
    assert tracker["exact_streak"] == 1
    assert tracker["varying_result_streak"] == 1
    assert tracker["varying_args_streak"] == 2


async def test_tool_name_change_resets_to_baseline():
    """换工具名：next_tracker 回到 baseline（全 1）。"""
    state: dict = {}
    # 先累积两次同参同果
    await _call_guard(state, name="read_file", args={"f": "x"}, content="r")
    await _call_guard(state, name="read_file", args={"f": "x"}, content="r")
    assert state["tool_storm"]["exact_streak"] == 2
    # 换工具
    await _call_guard(state, name="grep", args={"pattern": "x"}, content="r")

    tracker = state["tool_storm"]
    assert tracker["exact_streak"] == 1
    assert tracker["varying_result_streak"] == 1
    assert tracker["varying_args_streak"] == 1


# ─── 阈值边界 ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mode", "warmups"),
    [("exact", 3), ("varying_result", 5), ("varying_args", 8)],
    ids=["exact", "varying_result", "varying_args"],
)
async def test_call_is_blocked_once_streak_hits_the_limit(mode, warmups):
    """累积到阈值后，下一次同模式调用被阻断：handler 不执行、返回 error、tracker 置 None。"""
    state: dict = {}
    for index in range(warmups):
        name, args, content = _repeated_call(index, mode=mode)
        _, handler = await _call_guard(
            state, name=name, args=args, content=content
        )
        handler.assert_awaited_once()

    name, args, content = _repeated_call(warmups, mode=mode)
    blocked, handler = await _call_guard(
        state, name=name, args=args, content=content
    )

    handler.assert_not_awaited()
    assert _blocked_message(blocked).status == "error"
    assert blocked.update["tool_storm"] is None


@pytest.mark.parametrize(
    ("mode", "warmups"),
    [("exact", 2), ("varying_result", 4), ("varying_args", 7)],
    ids=["exact_2", "varying_result_4", "varying_args_7"],
)
async def test_call_is_allowed_one_step_before_the_limit(mode, warmups):
    """阈值前一步（exact=2 / varying_result=4 / varying_args=7）：放行，不阻断。"""
    state: dict = {}
    for index in range(warmups):
        name, args, content = _repeated_call(index, mode=mode)
        _, handler = await _call_guard(
            state, name=name, args=args, content=content
        )
        handler.assert_awaited_once()

    # 再来一次——应该仍然放行（还没到阈值）
    name, args, content = _repeated_call(warmups, mode=mode)
    result, handler = await _call_guard(
        state, name=name, args=args, content=content
    )

    handler.assert_awaited_once()
    # 放行返回的 messages 里是正常 ToolMessage（非 error）
    msg = result.update["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert msg.status != "error"
    # tracker 仍在更新
    assert result.update["tool_storm"] is not None


# ─── 阻断后重置 ───────────────────────────────────────────


async def test_block_then_switch_tool_is_allowed():
    """阻断后（tracker 置 None）换一个工具：正常放行。"""
    state = {
        "tool_storm": _tracker(
            _snapshot(
                "read_file",
                _dump({"file_path": "same.py"}),
                "result",
            ),
            exact=3,
        )
    }

    _, handler = await _call_guard(
        state,
        name="grep",
        args={"pattern": "new"},
        content="new result",
    )

    handler.assert_awaited_once()
    assert state["tool_storm"] is not None


async def test_block_then_deliberate_retry_is_allowed():
    """阻断后（tracker 置 None）立即重试同一调用：第二次放行（给了模型一次纠正机会）。"""
    state: dict = {}
    for _ in range(3):
        await _call_guard(
            state,
            name="read_file",
            args={"file_path": "same.py"},
            content="same result",
        )

    # 第 4 次：被阻断
    _, blocked_handler = await _call_guard(
        state,
        name="read_file",
        args={"file_path": "same.py"},
        content="unused",
    )
    # 第 5 次：tracker 已重置，放行
    _, retry_handler = await _call_guard(
        state,
        name="read_file",
        args={"file_path": "same.py"},
        content="same result",
    )

    blocked_handler.assert_not_awaited()
    retry_handler.assert_awaited_once()


# ─── 工具异常也计数 ────────────────────────────────────────


async def test_converted_tool_errors_are_counted_and_then_blocked():
    """工具抛异常被 handle_tool_errors 转成 error ToolMessage，storm 照常计数并最终阻断。"""
    state: dict = {}
    execute = AsyncMock(side_effect=RuntimeError("same failure"))

    async def converted_error_handler(request):
        return await handle_tool_errors.awrap_tool_call(request, execute)

    for _ in range(3):
        request = SimpleNamespace(
            state=state,
            tool_call={"name": "bash", "args": {"command": "bad"}, "id": "call"},
        )
        result = await tool_call_storm_block.awrap_tool_call(
            request, converted_error_handler
        )
        state["tool_storm"] = result.update["tool_storm"]

    request = SimpleNamespace(
        state=state,
        tool_call={"name": "bash", "args": {"command": "bad"}, "id": "call"},
    )
    blocked = await tool_call_storm_block.awrap_tool_call(
        request, converted_error_handler
    )

    assert execute.await_count == 3
    assert _blocked_message(blocked).status == "error"
    assert blocked.update["tool_storm"] is None


# ─── Command 透传（native command 不参与风暴计数）──────────


async def test_native_command_is_returned_unchanged():
    """工具返回 Command（控制流指令）：原样透传，不更新 tracker。"""
    command = Command(
        graph=Command.PARENT,
        update={"custom": "preserve"},
        goto="next",
    )
    handler = AsyncMock(return_value=command)
    request = _make_request({}, name="custom", args={})

    result = await tool_call_storm_block.awrap_tool_call(request, handler)

    assert result is command


# ─── 并行批量工具调用 ──────────────────────────────────────


def _parallel_state() -> dict:
    """构造一个并行批量的 state：最近一条 AIMessage 含 2 个 tool_calls。"""
    args = {"file_path": "same.py"}
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": args,
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "grep",
                        "args": {"pattern": "new"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            )
        ],
        "tool_storm": _tracker(
            _snapshot(
                "read_file",
                _dump(args),
                "same-result",
            ),
            exact=3,
        ),
    }


async def test_parallel_tool_calls_execute_and_clear_old_tracker():
    """并行批量下即使已有 exact_streak=3，也不阻断：执行并重置 tracker 为 None。"""
    cases = [
        ("read_file", {"file_path": "same.py"}, "call-1"),
        ("grep", {"pattern": "new"}, "call-2"),
    ]

    for name, args, call_id in cases:
        state = _parallel_state()
        handler = AsyncMock(
            return_value=ToolMessage(content="result", tool_call_id=call_id)
        )
        request = _make_request(state, name=name, args=args, tool_call_id=call_id)

        result = await tool_call_storm_block.awrap_tool_call(request, handler)

        handler.assert_awaited_once()
        assert isinstance(result, Command)
        assert result.update["tool_storm"] is None


async def test_parallel_native_command_is_not_blocked_or_rewritten():
    """并行批量下返回 Command：原样透传（不包成带 tracker 的 Command）。"""
    command = Command(update={"custom": "preserve"}, goto="next")
    handler = AsyncMock(return_value=command)
    request = _make_request(
        _parallel_state(),
        name="read_file",
        args={"file_path": "same.py"},
        tool_call_id="call-1",
    )

    result = await tool_call_storm_block.awrap_tool_call(request, handler)

    handler.assert_awaited_once()
    assert result is command


# ─── 阻断提示文案 ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("reason", "hint", "tracker"),
    [
        (
            "exact",
            "工具参数和返回结果完全相同",
            _tracker(
                _snapshot("read_file", _dump({"file_path": "same.py"}), "r"),
                exact=3,
            ),
        ),
        (
            "varying_result",
            "返回结果持续波动",
            _tracker(
                _snapshot("read_file", _dump({"file_path": "same.py"}), "r"),
                varying_result=5,
            ),
        ),
        (
            "varying_args",
            "频繁更换参数",
            _tracker(
                _snapshot("read_file", _dump({"file_path": "old.py"}), "r"),
                varying_args=8,
            ),
        ),
    ],
)
async def test_blocked_message_is_chinese_guidance(reason, hint, tracker):
    """阻断时返回的 error ToolMessage content 含正确的中文提示。"""
    args = {"file_path": "new.py"} if reason == "varying_args" else {"file_path": "same.py"}
    request = _make_request(
        {"messages": [], "tool_storm": tracker},
        name="read_file",
        args=args,
    )

    result = await tool_call_storm_block.awrap_tool_call(request, AsyncMock())

    message = _blocked_message(result)
    assert message.status == "error"
    assert "【工具调用风暴限制触发】" in message.content
    assert hint in message.content
    assert "请复盘当前解题思路" in message.content


# ─── emit_tool_events 错误上报 ────────────────────────────


async def test_emit_tool_events_reports_error_inside_command_update():
    """storm 阻断返回的 Command 含 error ToolMessage 时，emit_tool_events 应上报 success=False。"""
    error_message = ToolMessage(
        content="blocked",
        tool_call_id="call",
        status="error",
    )
    command = Command(update={"messages": [error_message]})
    handler = AsyncMock(return_value=command)
    request = SimpleNamespace(
        tool_call={"name": "read_file", "args": {}, "id": "call"},
    )

    with patch("chcode.agent_setup._ipc_send") as mock_send:
        result = await emit_tool_events.awrap_tool_call(request, handler)

    assert result is command
    end_event = mock_send.call_args_list[-1].args[0]
    assert end_event["type"] == "tool_end"
    assert end_event["success"] is False


# ─── checkpoint 持久化 ────────────────────────────────────


async def test_tool_storm_tracker_survives_sqlite_checkpoint_resume(tmp_path):
    """tracker 存进 SQLite checkpoint，跨 turn 恢复后值不变。"""
    graph_builder = StateGraph(State)
    graph_builder.add_node("pass_through", lambda _state: {})
    graph_builder.add_edge(START, "pass_through")
    graph_builder.add_edge("pass_through", END)

    checkpointer = await create_checkpointer(tmp_path / "storm.db")
    graph = graph_builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "storm-resume"}}
    tracker = _tracker(
        _snapshot(
            "read_file",
            _dump({"file_path": "checkpoint.py"}),
            _dump("checkpoint result"),
        ),
        exact=3,
    )

    try:
        first = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="first turn")],
                "tool_storm": tracker,
            },
            config,
        )
        assert first["tool_storm"] == tracker

        resumed = await graph.ainvoke(
            {"messages": [HumanMessage(content="second turn")]},
            config,
        )
        assert resumed["tool_storm"] == tracker
    finally:
        await checkpointer.conn.close()


# ─── 静默性 ───────────────────────────────────────────────


async def test_guard_does_not_print_to_stdout(capsys):
    """中间件不向 stdout/stderr 打印任何东西。"""
    _, handler = await _call_guard(
        {},
        name="secret_lookup",
        args={"token": "argument-secret-marker"},
        content="result",
    )

    handler.assert_awaited_once()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ─── middleware 注册顺序 ──────────────────────────────────


def test_build_agent_places_storm_guard_outside_error_conversion():
    """storm 中间件必须注册在 handle_tool_errors 外层（数组顺序在前）。"""
    with (
        patch("chcode.agent_setup._dummy_model"),
        patch("chcode.agent_setup.create_agent") as mock_create,
        patch("chcode.agent_setup._get_all_tools", return_value=[]),
        patch("chcode.config.load_model_json", return_value={}),
        patch("chcode.agent_setup.EnhancedChatOpenAI"),
    ):
        build_agent(model_config={"model": "gpt-4"})

    middleware = mock_create.call_args.kwargs["middleware"]
    assert middleware.index(tool_call_storm_block) < middleware.index(
        handle_tool_errors
    )


# ─── 端到端集成测试 ────────────────────────────────────────


class _ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *args, **kwargs):
        return self


async def test_real_agent_graph_blocks_fourth_identical_tool_call():
    """真实 agent graph：连续 4 次同参调用，前 3 次执行、第 4 次被阻断。"""
    executions = 0

    @tool
    def repeated_lookup(query: str) -> str:
        """Return a deterministic result while recording actual executions."""
        nonlocal executions
        executions += 1
        return f"result for {query}"

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "repeated_lookup",
                    "args": {"query": "same"},
                    "id": f"call-{index}",
                    "type": "tool_call",
                }
            ],
        )
        for index in range(4)
    ]
    responses.append(AIMessage(content="done"))
    agent = create_agent(
        _ToolCallingFakeModel(responses=responses),
        [repeated_lookup],
        middleware=[tool_call_storm_block],
        state_schema=State,
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="run the lookup")]}
    )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert executions == 3
    assert len(tool_messages) == 4
    assert [message.status for message in tool_messages] == [
        "success",
        "success",
        "success",
        "error",
    ]
    assert result["tool_storm"] is None
