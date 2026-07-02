from unittest.mock import MagicMock, patch

import chcode.display as display


class TestFormatTokens:
    def test_small(self):
        assert display._format_tokens(999) == "999"

    def test_thousands(self):
        assert display._format_tokens(123456) == "123.5K"

    def test_exact_thousand(self):
        assert display._format_tokens(1000) == "1.0K"


class TestGetContextUsageText:
    def _ai_msg(self, input_tokens=None):
        from langchain_core.messages import AIMessage

        msg = MagicMock(spec=AIMessage)
        msg.usage_metadata = {"input_tokens": input_tokens} if input_tokens else None
        return msg

    def test_no_ai_messages(self):
        result = display.get_context_usage_text([], 128000)
        assert result == ""

    def test_with_tokens(self):
        msgs = [self._ai_msg(input_tokens=50000)]
        result = display.get_context_usage_text(msgs, 128000)
        assert "50.0K" in result
        assert "128.0K" in result
        assert "yellow" in result

    def test_high_usage(self):
        msgs = [self._ai_msg(input_tokens=120000)]
        result = display.get_context_usage_text(msgs, 128000)
        assert "bold red" in result

    def test_medium_usage(self):
        msgs = [self._ai_msg(input_tokens=100000)]
        result = display.get_context_usage_text(msgs, 128000)
        assert "bold yellow" in result


class TestRenderFunctions:
    def test_render_human(self):
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_human("Hello")
            mock_console.print.assert_called_once()

    def test_render_ai_chunk_suppressed_in_subagent(self):
        display._subagent_parallel = True
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_ai_chunk("text")
            mock_console.print.assert_not_called()
        display._subagent_parallel = False

    def test_render_ai_end(self):
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_ai_end()
            mock_console.print.assert_called_once()

    def test_render_success(self):
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_success("Done")
            mock_console.print.assert_called_once()

    def test_render_error(self):
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_error("Fail")
            mock_console.print.assert_called_once()

    def test_render_warning(self):
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_warning("Careful")
            mock_console.print.assert_called_once()

    def test_render_info(self):
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_info("FYI")
            mock_console.print.assert_called_once()

    def test_render_separator(self):
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_separator()
            mock_console.print.assert_called_once()

    def test_render_welcome(self):
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_welcome()
            assert mock_console.print.call_count >= 2
            # 校验 Panel 内容：含 ChCode 与翻译标题，且 "ChCode — " 前缀不重复
            from rich.panel import Panel

            panel = next(
                (c.args[0] for c in mock_console.print.call_args_list if c.args and isinstance(c.args[0], Panel)),
                None,
            )
            assert panel is not None, "should print a Panel"
            content = panel.renderable
            assert "ChCode" in content
            assert "终端 AI 编程助手" in content  # 默认中文
            assert "ChCode — ChCode" not in content  # 防止前缀重复
