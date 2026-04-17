"""
LangChain Tools 定义 (通用工具+skill工具)    后续补充web_search工具

使用 LangChain 1.0 的 @tool 装饰器和 ToolRuntime 定义工具：
- load_skill: 加载 Skill 详细指令（Level 2）
- bash: 执行命令/脚本（Level 3）
- read_file: 读取文件

ToolRuntime 提供访问运行时信息的统一接口：
- state: 可变的执行状态
- context: 不可变的配置（如 skill_loader）
"""

import asyncio
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import httpx
from langchain.tools import tool, ToolRuntime
from pydantic import BaseModel, BeforeValidator, Field
from rich.console import Console
from rich.text import Text
from chcode.display import render_tool_call

from chcode.utils.shell import (
    BashProvider,
    PowerShellProvider,
    ShellSession,
    interpret_command_result,
)
from chcode.utils.skill_loader import SkillAgentContext
from tavily import TavilyClient

console = Console()

CONFIG_DIR = Path.home() / ".chat"
SETTING_JSON = CONFIG_DIR / "chagent.json"

_tavily_api_key = os.getenv("TAVILY_API_KEY", "")
if SETTING_JSON.exists():
    try:
        data = json.loads(SETTING_JSON.read_text(encoding="utf-8"))
        api_key = data.get("tavily_api_key", "")
        if api_key:
            _tavily_api_key = api_key
    except Exception:
        pass

_tavily_client: TavilyClient | None = None


def get_tavily_client() -> TavilyClient | None:
    """获取 Tavily 客户端（懒加载）"""
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client
    if not _tavily_api_key:
        return None
    _tavily_client = TavilyClient(api_key=_tavily_api_key)
    return _tavily_client


def update_tavily_api_key(api_key: str) -> None:
    """运行时更新 Tavily API Key"""
    global _tavily_api_key, _tavily_client
    _tavily_api_key = api_key
    if api_key:
        _tavily_client = TavilyClient(api_key=api_key)
    else:
        _tavily_client = None


def resolve_path(file_path: str, working_directory: Path) -> Path:  # type: ignore[assignment]
    """
    解析文件路径，处理相对路径和 ~ 展开

    Args:
        file_path: 文件路径（绝对或相对，支持 ~ 表示用户主目录）
        working_directory: 工作目录

    Returns:
        解析后的绝对路径
    """
    path = Path(file_path).expanduser()  # 处理 ~ 展开
    if not path.is_absolute():
        path = working_directory / path
    return path


@tool
def load_skill(skill_name: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Load a skill's detailed instructions.

    This tool reads the SKILL.md file for the specified skill and returns
    its complete instructions. Use this when the user's request matches
    a skill's description from the available skills list.

    The skill's instructions will guide you on how to complete the task,
    which may include running scripts via the bash tool.

    Args:
        skill_name: Name of the skill to load (e.g., 'news-extractor')
    """
    loader = runtime.context.skill_loader
    render_tool_call("load_skill", skill_name)

    # 尝试加载 skill
    skill_content = loader.load_skill(skill_name)

    if not skill_content:
        # # 列出可用的 skills（从已扫描的元数据中获取）
        skills = loader.scan_skills()
        if skills:
            available = [s.name for s in skills]
            return f"Skill '{skill_name}' not found. Available skills: {', '.join(available)}"
        return f"Skill '{skill_name}' not found. No skills are currently available."

    # 获取 skill 路径信息
    skill_path = skill_content.metadata.skill_path
    scripts_dir = skill_path / "scripts"

    scripts_info = (
        f"""
- **Scripts Directory**: `{scripts_dir}`

**Important**: When running scripts, use absolute paths like:
```bash
uv run {scripts_dir}/script_name.py [args]
```"""
        if scripts_dir.exists()
        else ""
    )

    # 构建路径信息
    path_info = (
        f"""
## Skill Path Info

- **Skill Directory**: `{skill_path}`"""
        + scripts_info
    )

    # 返回 instructions 和路径信息
    return f"""# Skill: {skill_name}

## Instructions

{skill_content.instructions}
{path_info}
"""


def _create_shell_session(workdir: str) -> ShellSession:
    is_windows = platform.system() == "Windows"
    if is_windows:
        bash_provider = BashProvider()
        if bash_provider.is_available:
            session = ShellSession(bash_provider)
            session.cwd = workdir
            return session
        ps_provider = PowerShellProvider()
        if ps_provider.is_available:
            session = ShellSession(ps_provider)
            session.cwd = workdir
            return session
    else:
        bash_provider = BashProvider()
        if bash_provider.is_available:
            session = ShellSession(bash_provider)
            session.cwd = workdir
            return session

    return None


_shell_sessions: dict[str, ShellSession] = {}


def _get_shell_session(workdir: str) -> ShellSession:
    key = workdir
    session = _shell_sessions.get(key)
    if session is not None:
        return session
    session = _create_shell_session(workdir)
    if session is not None:
        _shell_sessions[key] = session
        return session
    return None


@tool
def bash(
    command: str,
    runtime: ToolRuntime[SkillAgentContext],
    timeout: int = 300,
    workdir: str | None = None,
) -> str:
    """
    Execute a shell command with automatic platform detection and CWD tracking.

    On Windows: uses Git Bash if available, falls back to PowerShell.
    On Linux/Mac: uses the system shell (bash/zsh).

    The working directory is tracked across commands within the same session.
    Use 'workdir' to override the working directory for a specific command
    without affecting the session's tracked CWD.

    Output is automatically truncated if it exceeds 2000 lines or 51200 bytes.
    Certain exit codes are interpreted semantically (e.g., grep exit 1 = no matches).

    Args:
        command: The shell command to execute
        timeout: Timeout in seconds (default 300, max 600)
        workdir: Working directory override (default: project root)
    """
    cwd = str(runtime.context.working_directory)
    render_tool_call("bash", command)

    timeout = min(timeout, 600)
    timeout_ms = timeout * 1000

    session = _get_shell_session(cwd)
    if session is None:
        return "bash:\n[FAILED] No shell available on this system"

    exec_workdir = workdir if workdir else None
    result, truncated = session.execute(
        command, timeout=timeout_ms, workdir=exec_workdir
    )

    interpretation = interpret_command_result(command, result.exit_code)

    parts = []
    if result.exit_code == 0:
        parts.append(f"[OK] ({session.provider_name})")
    elif interpretation.message and not interpretation.is_error:
        parts.append(f"[OK] ({session.provider_name}) {interpretation.message}")
    else:
        parts.append(
            f"[FAILED] Exit code: {result.exit_code} ({session.provider_name})"
        )
    parts.append("")

    output = truncated.content if truncated.truncated else result.stdout
    if output and output.strip():
        parts.append(output.rstrip())

    if result.stderr and result.stderr.strip():
        if output and output.strip():
            parts.append("")
        parts.append("--- stderr ---")
        parts.append(result.stderr.rstrip())

    if result.timed_out:
        parts.append("")
        parts.append(f"Command timed out after {timeout}s")

    if not output or not output.strip():
        if not result.stderr or not result.stderr.strip():
            parts.append("(no output)")

    return "bash:\n" + "\n".join(parts)


@tool
def read_file(file_path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Read the contents of a file.

    Use this to:
    - Read skill documentation files
    - View script output files
    - Inspect any text file

    Args:
        file_path: Path to the file (absolute or relative to working directory)
    """
    path = resolve_path(file_path, runtime.context.working_directory)
    render_tool_call("read_file", file_path)

    if not path.exists():
        return f"read:\n[FAILED] File not found: {file_path}"

    if not path.is_file():
        return f"read:\n[FAILED] Not a file: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        numbered_lines = []
        for i, line in enumerate(lines[:2000], 1):
            numbered_lines.append(f"{i:4d}| {line}")

        if len(lines) > 2000:
            numbered_lines.append(f"... ({len(lines) - 2000} more lines)")

        result = "\n".join(numbered_lines)
        if len(lines) > 2000:
            return f"read:\n[OK] ({len(lines)} lines, showing first 2000)\n\n{result}"
        return f"read:\n[OK]\n\n{result}"

    except UnicodeDecodeError:
        return f"read:\n[FAILED] Cannot read file (binary or unknown encoding): {file_path}"
    except Exception as e:
        return f"read:\n[FAILED] Failed to read file: {str(e)}"


@tool
def write_file(
    file_path: str, content: str, runtime: ToolRuntime[SkillAgentContext]
) -> str:
    """
    Write content to a file.

    Use this to:
    - Save generated content
    - Create new files
    - Modify existing files

    Args:
        file_path: Path to the file (absolute or relative to working directory)
        content: Content to write to the file
    """
    path = resolve_path(file_path, runtime.context.working_directory)
    render_tool_call("write_file", file_path)

    try:
        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding="utf-8")
        return f"write:\n[OK] File written: {path}"

    except Exception as e:
        return f"write:\n[FAILED] Failed to write file: {str(e)}"


@tool
def glob(pattern: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Find files matching a glob pattern.

    Use this to:
    - Find files by name pattern (e.g., "**/*.py" for all Python files)
    - List files in a directory with wildcards
    - Discover project structure

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts", "*.md")
    """
    cwd = runtime.context.working_directory
    render_tool_call("glob", pattern)

    try:
        # 使用 Path.glob 进行匹配
        matches = sorted(cwd.glob(pattern))

        if not matches:
            return f"glob:\n[FAILED] No files matching pattern: {pattern}"

        max_results = 100
        result_lines = []

        for path in matches[:max_results]:
            try:
                rel_path = path.relative_to(cwd)
                result_lines.append(str(rel_path))
            except ValueError:
                result_lines.append(str(path))

        result = "\n".join(result_lines)

        if len(matches) > max_results:
            result += f"\n... and {len(matches) - max_results} more files"

        return f"glob:\n[OK] ({len(matches)} matches)\n\n{result}"

    except Exception as e:
        return f"glob:\n[FAILED] {str(e)}"


_GREP_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".cache",
        ".sass-cache",
        "target",
        "Pods",
    }
)
_GREP_BINARY_EXT = frozenset(
    {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".obj",
        ".o",
        ".a",
        ".lib",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wav",
        ".flac",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".war",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".otf",
        ".sqlite",
        ".db",
    }
)
_GREP_MAX_FILE_SIZE = 1 * 1024 * 1024


@tool
def grep(pattern: str, path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Search for a pattern in files.

    Use this to:
    - Find code containing specific text or regex
    - Search for function/class definitions
    - Locate usages of variables or imports

    Args:
        pattern: Regular expression pattern to search for
        path: File or directory path to search in (use "." for current directory)
    """
    cwd = runtime.context.working_directory
    render_tool_call("grep", pattern)
    search_path = resolve_path(path, cwd)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"grep:\n[FAILED] Invalid regex pattern: {e}"

    results = []
    max_results = 50
    files_searched = 0

    def _search_file(file_path: Path) -> None:
        nonlocal files_searched
        try:
            size = file_path.stat().st_size
            if size > _GREP_MAX_FILE_SIZE:
                return
        except OSError:
            return

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")
            files_searched += 1

            for line_num, line in enumerate(lines, 1):
                if regex.search(line):
                    try:
                        rel_path = file_path.relative_to(cwd)
                    except ValueError:
                        rel_path = file_path
                    results.append(f"{rel_path}:{line_num}: {line.strip()[:100]}")

                    if len(results) >= max_results:
                        return
        except (PermissionError, IsADirectoryError):
            pass

    try:
        if search_path.is_file():
            _search_file(search_path)
        else:
            for p in search_path.rglob("*"):
                if len(results) >= max_results:
                    break
                if not p.is_file():
                    continue
                parts = p.parts
                if any(
                    part.startswith(".") or part in _GREP_EXCLUDED_DIRS
                    for part in parts
                ):
                    continue
                if p.suffix.lower() in _GREP_BINARY_EXT:
                    continue
                _search_file(p)

        if not results:
            return f"grep:\n[FAILED] No matches found for pattern: {pattern} (searched {files_searched} files)"

        output = "\n".join(results)
        if len(results) >= max_results:
            output += f"\n... (truncated, showing first {max_results} matches)"

        return f"grep:\n[OK] ({len(results)} matches in {files_searched} files)\n\n{output}"

    except Exception as e:
        return f"grep:\n[FAILED] {str(e)}"


@tool
def edit(
    file_path: str,
    old_string: str,
    new_string: str,
    runtime: ToolRuntime[SkillAgentContext],
) -> str:
    """
    Edit a file by replacing text.

    Use this to:
    - Modify existing code
    - Fix bugs by replacing incorrect code
    - Update configuration values

    The old_string must match exactly (including whitespace/indentation).
    For safety, the old_string must be unique in the file.

    Args:
        file_path: Path to the file to edit
        old_string: The exact text to find and replace
        new_string: The text to replace it with
    """
    path = resolve_path(file_path, runtime.context.working_directory)
    render_tool_call("edit", file_path)

    if not path.exists():
        return f"edit:\n[FAILED] File not found: {file_path}"

    if not path.is_file():
        return f"edit:\n[FAILED] Not a file: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")

        count = content.count(old_string)

        if count == 0:
            return "edit:\n[FAILED] String not found in file. Make sure the text matches exactly including whitespace."

        if count > 1:
            return f"edit:\n[FAILED] String appears {count} times in file. Please provide more context to make it unique."

        new_content = content.replace(old_string, new_string, 1)
        path.write_text(new_content, encoding="utf-8")

        old_lines = len(old_string.split("\n"))
        new_lines = len(new_string.split("\n"))

        return f"edit:\n[OK] Edited {path.name}: replaced {old_lines} lines with {new_lines} lines"

    except UnicodeDecodeError:
        return f"edit:\n[FAILED] Cannot edit file (binary or unknown encoding): {file_path}"
    except Exception as e:
        return f"edit:\n[FAILED] {str(e)}"


@tool
def list_dir(path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    List contents of a directory.

    Use this to:
    - Explore directory structure
    - See what files exist in a folder
    - Check if files/folders exist

    Args:
        path: Directory path (use "." for current directory)
    """
    dir_path = resolve_path(path, runtime.context.working_directory)
    render_tool_call("list_dir", path)

    if not dir_path.exists():
        return f"ls:\n[FAILED] Directory not found: {path}"

    if not dir_path.is_dir():
        return f"ls:\n[FAILED] Not a directory: {path}"

    try:
        entries = sorted(
            dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )

        result_lines = []
        for entry in entries[:100]:  # 限制数量
            if entry.is_dir():
                result_lines.append(f"{entry.name}/")
            else:
                # 显示文件大小
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size // 1024}KB"
                else:
                    size_str = f"{size // (1024 * 1024)}MB"
                result_lines.append(f"   {entry.name} ({size_str})")

        if len(entries) > 100:
            result_lines.append(f"... and {len(entries) - 100} more entries")

        return f"ls:\n[OK] ({len(entries)} entries)\n\n{chr(10).join(result_lines)}"

    except PermissionError:
        return f"ls:\n[FAILED] Permission denied: {path}"
    except Exception as e:
        return f"ls:\n[FAILED] {str(e)}"


@tool
async def web_search(
    query: str,
    runtime: ToolRuntime[SkillAgentContext],
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    render_tool_call("web_search", query)
    client = get_tavily_client()
    if client is None:
        return "[ERROR] Tavily API Key 未配置，请使用 /search 命令配置"
    return await asyncio.to_thread(
        client.search,
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


FETCH_TIMEOUT = 60.0
MAX_MARKDOWN_LENGTH = 100_000
MAX_URL_LENGTH = 2000


def _html_to_markdown(html: str) -> str:
    try:
        from markdownify import markdownify as md

        return md(html)
    except ImportError:
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def _is_binary_content_type(content_type: str) -> bool:
    binary_types = [
        "application/pdf",
        "application/zip",
        "application/x-tar",
        "application/gzip",
        "application/x-bzip2",
        "image/",
        "video/",
        "audio/",
    ]
    return any(bt in content_type.lower() for bt in binary_types)


@tool
async def web_fetch(url: str) -> dict:
    """Fetches content from a specified URL and converts it to text."""
    render_tool_call("web_fetch", url)
    start = time.time()

    if len(url) > MAX_URL_LENGTH:
        return {
            "url": url,
            "bytes": 0,
            "code": 0,
            "code_text": "Error",
            "result": f"URL exceeds maximum length of {MAX_URL_LENGTH} characters",
            "duration_ms": int((time.time() - start) * 1000),
        }

    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {
                "url": url,
                "bytes": 0,
                "code": 0,
                "code_text": "Error",
                "result": f"Invalid URL: {url}",
                "duration_ms": int((time.time() - start) * 1000),
            }

        if parsed.scheme == "http":
            url = url.replace("http://", "https://", 1)

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=FETCH_TIMEOUT,
            max_redirects=10,
            headers={
                "Accept": "text/markdown, text/html, */*",
                "User-Agent": "ClaudeToolkit/1.0",
            },
        ) as client:
            response = await client.get(url)

        content_type = response.headers.get("content-type", "")
        raw_bytes = len(response.content)

        if _is_binary_content_type(content_type):
            return {
                "url": url,
                "bytes": raw_bytes,
                "code": response.status_code,
                "code_text": response.reason_phrase,
                "result": f"Binary content ({content_type}, {raw_bytes} bytes). Cannot extract text.",
                "duration_ms": int((time.time() - start) * 1000),
            }

        html_content = response.text

        if "text/html" in content_type:
            markdown_content = _html_to_markdown(html_content)
        else:
            markdown_content = html_content

        if len(markdown_content) > MAX_MARKDOWN_LENGTH:
            markdown_content = (
                markdown_content[:MAX_MARKDOWN_LENGTH]
                + "\n\n[Content truncated due to length...]"
            )

        result = f"Content from {url}:\n\n{markdown_content}\n\n---"
        # resp=model.invoke(f"Extract effective message from {url}:\n\n{markdown_content}")
        # result=resp.content

        return {
            "url": url,
            "bytes": raw_bytes,
            "code": response.status_code,
            "code_text": response.reason_phrase,
            "result": result,
            "duration_ms": int((time.time() - start) * 1000),
        }

    except httpx.TimeoutException:
        return {
            "url": url,
            "bytes": 0,
            "code": 0,
            "code_text": "Timeout",
            "result": f"Request timed out after {FETCH_TIMEOUT}s",
            "duration_ms": int((time.time() - start) * 1000),
        }
    except httpx.HTTPError as e:
        return {
            "url": url,
            "bytes": 0,
            "code": 0,
            "code_text": "Error",
            "result": f"HTTP error: {e}",
            "duration_ms": int((time.time() - start) * 1000),
        }
    except Exception as e:
        return {
            "url": url,
            "bytes": 0,
            "code": 0,
            "code_text": "Error",
            "result": f"Error fetching URL: {e}",
            "duration_ms": int((time.time() - start) * 1000),
        }


async def _checkbox_with_other_async(
    question: str, options: list[str]
) -> list[str] | None:
    """
    多选 + 自定义输入框（异步版本）

    空格切换选中，Tab 切换列表/输入框焦点，Enter 提交。
    输入行始终可见，用于输入不在列表中的自定义选项。
    """
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, UIContent
    from prompt_toolkit.layout.controls import (
        FormattedTextControl,
        BufferControl,
        UIControl,
    )
    from prompt_toolkit.layout.containers import HSplit, Window

    input_buffer = Buffer()
    input_row_idx = len(options)

    class _CheckboxControl(UIControl):
        def __init__(self, opts: list[str]):
            self.opts = opts
            self.selected = 0
            self.checked: set[int] = set()

        def is_focusable(self) -> bool:
            return True

        def get_invalidate_events(self):
            yield input_buffer.on_text_changed

        def preferred_height(
            self, width, max_available_height, wrap_lines, get_line_prefix
        ):
            return len(self.opts) + 1

        def create_content(self, width: int, height: int) -> UIContent:
            lines = []
            for i, opt in enumerate(self.opts):
                marker = "[√]" if i in self.checked else "[ ]"
                prefix = "  ❯ " if i == self.selected else "    "
                line = f"{prefix}{marker} {opt}"
                style = "bold" if i == self.selected else ""
                lines.append([(style, line)])

            input_text = input_buffer.text or ""
            input_prefix = "  ❯ " if self.selected == input_row_idx else "    "
            input_line = f"{input_prefix}> {input_text}"
            input_style = "bold" if self.selected == input_row_idx else ""
            lines.append([(input_style, input_line)])

            def get_line(i):
                return lines[i] if i < len(lines) else [("", "")]

            return UIContent(
                get_line=get_line,
                line_count=len(lines),
            )

    control = _CheckboxControl(options)

    question_window = Window(
        height=1,
        content=FormattedTextControl(text=f"? {question}"),
        dont_extend_height=True,
    )
    control_window = Window(content=control)

    input_edit = Window(
        content=BufferControl(buffer=input_buffer),
        height=1,
        dont_extend_height=True,
        char=" ",
    )

    kb = KeyBindings()
    _exiting = False

    @kb.add("up")
    def _up(e):
        nonlocal _exiting
        if _exiting:
            return
        if control.selected > 0:
            control.selected -= 1
            if control.selected < input_row_idx:
                e.app.layout.focus(control_window)
            else:
                e.app.layout.focus(input_edit)
        e.app.invalidate()

    @kb.add("down")
    def _down(e):
        nonlocal _exiting
        if _exiting:
            return
        if control.selected < input_row_idx:
            control.selected += 1
            if control.selected == input_row_idx:
                e.app.layout.focus(input_edit)
        e.app.invalidate()

    @kb.add("tab")
    def _tab(e):
        nonlocal _exiting
        if _exiting:
            return
        if control.selected == input_row_idx:
            control.selected = 0
            e.app.layout.focus(control_window)
        else:
            control.selected = input_row_idx
            e.app.layout.focus(input_edit)
        e.app.invalidate()

    @kb.add(" ")
    def _space(e):
        nonlocal _exiting
        if _exiting:
            return
        if control.selected < input_row_idx:
            control.checked ^= {control.selected}
        e.app.invalidate()

    @kb.add("enter")
    def _enter(e):
        nonlocal _exiting
        if _exiting:
            return
        _exiting = True
        selected_names = [control.opts[i] for i in sorted(control.checked)]
        custom = input_buffer.text.strip()
        if custom:
            selected_names.append(custom)
        try:
            e.app.exit(result=selected_names if selected_names else None)
        except Exception:
            pass

    @kb.add("escape")
    def _esc(e):
        nonlocal _exiting
        if _exiting:
            return
        _exiting = True
        try:
            e.app.exit(result=None)
        except Exception:
            pass

    @kb.add("c-c")
    def _cancel(e):
        nonlocal _exiting
        if _exiting:
            return
        _exiting = True
        try:
            e.app.exit(result=None)
        except Exception:
            pass

    layout = Layout(HSplit([question_window, control_window, input_edit]))
    app = Application(
        layout=layout, key_bindings=kb, full_screen=False, erase_when_done=True
    )
    result = await app.run_async()
    if result is not None:
        console.print(f"[cyan]?[/cyan] {question} [bold]{', '.join(result)}[/bold]")
    return result


async def _select_with_other_async(question: str, options: list[str]) -> str | None:
    """
    下拉选择 + 自定义输入框（异步版本）。
    输入行始终可见，用上下箭头或 Tab 移动到输入行直接输入。
    """
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, UIContent
    from prompt_toolkit.layout.controls import (
        FormattedTextControl,
        BufferControl,
        UIControl,
    )
    from prompt_toolkit.layout.containers import HSplit, Window

    input_buffer = Buffer()
    input_row_idx = len(options)

    class _SelectControl(UIControl):
        def __init__(self, opts: list[str]):
            self.opts = opts
            self.selected = 0

        def is_focusable(self) -> bool:
            return True

        def get_invalidate_events(self):
            yield input_buffer.on_text_changed

        def preferred_height(
            self, width, max_available_height, wrap_lines, get_line_prefix
        ):
            return len(self.opts) + 1

        def create_content(self, width: int, height: int) -> UIContent:
            lines = []
            for i, opt in enumerate(self.opts):
                prefix = "  ❯ " if i == self.selected else "    "
                line = f"{prefix}{opt}"
                style = "bold" if i == self.selected else ""
                lines.append([(style, line)])

            input_text = input_buffer.text or ""
            input_prefix = "  ❯ " if self.selected == input_row_idx else "    "
            input_line = f"{input_prefix}> {input_text}"
            input_style = "bold" if self.selected == input_row_idx else ""
            lines.append([(input_style, input_line)])

            def get_line(i):
                return lines[i] if i < len(lines) else [("", "")]

            return UIContent(
                get_line=get_line,
                line_count=len(lines),
            )

    control = _SelectControl(options)

    question_window = Window(
        height=1,
        content=FormattedTextControl(text=f"? {question}"),
        dont_extend_height=True,
    )
    control_window = Window(content=control)

    input_edit = Window(
        content=BufferControl(buffer=input_buffer),
        height=1,
        dont_extend_height=True,
        char=" ",
    )

    kb = KeyBindings()
    _exiting = False

    @kb.add("up")
    def _up(e):
        nonlocal _exiting
        if _exiting:
            return
        if control.selected > 0:
            control.selected -= 1
            if control.selected < input_row_idx:
                e.app.layout.focus(control_window)
            else:
                e.app.layout.focus(input_edit)
        e.app.invalidate()

    @kb.add("down")
    def _down(e):
        nonlocal _exiting
        if _exiting:
            return
        if control.selected < input_row_idx:
            control.selected += 1
            if control.selected == input_row_idx:
                e.app.layout.focus(input_edit)
        e.app.invalidate()

    @kb.add("tab")
    def _tab(e):
        nonlocal _exiting
        if _exiting:
            return
        if control.selected == input_row_idx:
            control.selected = 0
            e.app.layout.focus(control_window)
        else:
            control.selected = input_row_idx
            e.app.layout.focus(input_edit)
        e.app.invalidate()

    @kb.add("enter")
    def _enter(e):
        nonlocal _exiting
        if _exiting:
            return
        _exiting = True
        if control.selected == input_row_idx:
            text = input_buffer.text.strip()
            if text:
                try:
                    e.app.exit(result=text)
                except Exception:
                    pass
            else:
                _exiting = False
        else:
            try:
                e.app.exit(result=control.opts[control.selected])
            except Exception:
                pass

    @kb.add("escape")
    def _esc(e):
        nonlocal _exiting
        if _exiting:
            return
        _exiting = True
        try:
            e.app.exit(result=None)
        except Exception:
            pass

    @kb.add("c-c")
    def _cancel(e):
        nonlocal _exiting
        if _exiting:
            return
        _exiting = True
        try:
            e.app.exit(result=None)
        except Exception:
            pass

    layout = Layout(HSplit([question_window, control_window, input_edit]))
    app = Application(
        layout=layout, key_bindings=kb, full_screen=False, erase_when_done=True
    )
    result = await app.run_async()
    if result is not None:
        console.print(f"[cyan]?[/cyan] {question} [bold]{result}[/bold]")
    return result


def _coerce_json_list(v: Any) -> Any:
    """将 JSON 字符串解析为 list，兼容 LLM 把数组参数传为字符串的情况"""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return v
    return v


@tool
async def ask_user(
    question: str = "",
    options: Annotated[list[str] | None, BeforeValidator(_coerce_json_list)] = None,
    is_multiple: bool = False,
    questions: Annotated[list[dict] | None, BeforeValidator(_coerce_json_list)] = None,
) -> str:
    """
    Ask the user one or more questions interactively with predefined options.

    Use this when you need clarification, user preferences, or choices
    before proceeding. The user will see dropdown menus or checkboxes in the terminal.

    Single question mode (default):
        ask_user(question="What's your preference?", options=["A", "B", "C"])

    Batch questions mode (asks all questions sequentially):
        ask_user(questions=[
            {"question": "What database?", "options": ["PostgreSQL", "MySQL"]},
            {"question": "What framework?", "options": ["React", "Vue"], "is_multiple": true},
        ])

    Args:
        question: The question to ask (ignored if questions is provided)
        options: List of options for single question mode (ignored if questions is provided)
        is_multiple: If True in single-question mode, allow selecting multiple options
        questions: List of question dicts for batch mode. Each dict must have:
                   - "question": the question text (required)
                   - "options": list of choices (optional, falls back to text input)
                   - "is_multiple": bool for checkbox vs single select (default: false)
                   When provided, all questions are asked sequentially. Overrides question/options.
    """
    import questionary

    if questions:
        return await _ask_multi_questions(questions)

    render_tool_call("ask_user", question)

    if not options:
        answer = await asyncio.to_thread(lambda: questionary.text("请输入: ").ask())
        if answer is None:
            return "user_answer:\n(用户取消)"
        return f"user_answer:\n{answer}"

    try:
        if is_multiple:
            selected = await _checkbox_with_other_async(question, options)
            if selected is None:
                return "user_answer:\n(用户取消)"
            result = ", ".join(selected)
        else:
            answer = await _select_with_other_async(question, options)
            if answer is None:
                return "user_answer:\n(用户取消)"
            result = answer
        return f"user_answer:\n{result}"
    except Exception as e:
        return f"user_answer:\n(询问失败: {e})"


async def _ask_multi_questions(questions: list[dict]) -> str:
    """
    多问题批量提问界面

    Args:
        questions: List of {"question": "...", "options": ["..."], "is_multiple": false}
    Returns:
        Formatted string with all answers
    """
    import questionary

    console.print()
    console.print(f"[bold cyan]📋 批量提问 ({len(questions)} 个问题)[/bold cyan]")
    console.print()

    answers = []
    for i, q in enumerate(questions, 1):
        q_text = q.get("question", "")
        q_options = q.get("options", [])
        q_multiple = q.get("is_multiple", False)

        console.print(f"[dim]问题 {i}/{len(questions)}: {q_text}[/dim]")

        if not q_options:
            answer = await asyncio.to_thread(lambda: questionary.text("请输入: ").ask())
            if answer is None:
                answers.append(f"Q{i}: (用户取消)")
                continue
            answers.append(f"Q{i}: {answer}")
        else:
            try:
                if q_multiple:
                    selected = await _checkbox_with_other_async(
                        "选择（空格选择，回车确认）:", q_options
                    )
                    if selected is None:
                        answers.append(f"Q{i}: (用户取消)")
                        continue
                    result = ", ".join(selected)
                else:
                    selected = await _select_with_other_async(q_text, q_options)
                    if selected is None:
                        answers.append(f"Q{i}: (用户取消)")
                        continue
                    result = selected
                answers.append(f"Q{i}: {result}")
            except Exception as e:
                answers.append(f"Q{i}: (询问失败: {e})")

        console.print()

    # 汇总结果
    result_lines = ["=== 批量提问结果 ==="]
    for i, q in enumerate(questions, 1):
        result_lines.append(f"问题: {q.get('question', '')}\n回答: {answers[i - 1]}")
    return "\n\n".join(result_lines)


@tool
async def agent(
    prompt: str,
    subagent_type: str = "general-purpose",
    description: str = "",
    timeout_seconds: int = 300,
    runtime: ToolRuntime[SkillAgentContext] = None,
) -> str:
    """
    Launch a sub-agent to perform a task autonomously.

    Sub-agents have their own tool set and system prompt. They run independently
    and return a text report when finished. Use this for complex, multi-step tasks
    that benefit from focused execution.

    Available sub-agent types:
    - "general-purpose": Full tool access, can read/write files and run commands. Use for multi-step coding tasks, complex research, or when you need autonomous execution.
    - "Explore": Read-only agent specialized in codebase exploration. Use for finding files by pattern, searching code, answering questions about the codebase. Fast and efficient.
    - "Plan": Read-only agent for designing implementation plans. Use for architectural analysis, step-by-step implementation strategies, identifying critical files.

    Custom agents can also be loaded from .chat/agents/*.md files.

    Args:
        prompt: The task description for the sub-agent. Be specific about what to find, analyze, or do.
        subagent_type: Type of sub-agent to launch ("general-purpose", "Explore", "Plan", or a custom agent name).
        description: Short description of what this sub-agent invocation does (for display purposes).
        timeout_seconds: Maximum seconds the sub-agent can run before being terminated. Default 300 (5 minutes). Must be greater than 300 (5 minutes) to allow sufficient execution time.
    """
    import chcode.display as _display
    from chcode.agents.loader import load_agents
    from chcode.agents.runner import run_subagent

    tag = f"{subagent_type}: {(description or '')[:30]}"
    render_tool_call("agent", f"{subagent_type}: {description or prompt[:60]}")

    all_agents = load_agents()
    agent_def = all_agents.get(subagent_type)

    if agent_def is None:
        available = ", ".join(sorted(all_agents.keys()))
        return f"Unknown agent type '{subagent_type}'. Available types: {available}"

    model_config = runtime.context.model_config
    working_directory = runtime.context.working_directory
    skill_loader = runtime.context.skill_loader

    _display._current_agent_tag.set(tag)
    with _display._agent_progress_lock:
        _display._agent_progress[tag] = {
            "calls": 0,
            "start": time.time(),
            "timeout": timeout_seconds,
            "failed": False,
        }

    with _display._subagent_count_lock:
        _display._subagent_count += 1
        if _display._subagent_count >= 2:
            _display._subagent_parallel = True
            _display._start_progress()
            if _display._progress_task is None or _display._progress_task.done():
                _display._progress_task = asyncio.ensure_future(
                    _display._progress_updater()
                )

    try:
        result = await run_subagent(
            prompt=prompt,
            agent_def=agent_def,
            model_config=model_config,
            working_directory=working_directory,
            skill_loader=skill_loader,
            timeout_seconds=timeout_seconds,
            description=description,
        )

        with _display._agent_progress_lock:
            if tag in _display._agent_progress:
                # 检查是否超时或出错
                if result and ("timed out" in result or "error:" in result.lower()):
                    _display._agent_progress[tag]["failed"] = True
                else:
                    _display._agent_progress[tag]["done"] = True
        # 触发一次立即更新
        _display._update_progress()

        if not _display._subagent_parallel and result:
            for line in result.splitlines():
                _display.console.print(Text(f"  {line}", style="dim"))
    finally:
        _display._current_agent_tag.set(None)
        with _display._subagent_count_lock:
            _display._subagent_count -= 1
            if _display._subagent_count == 0:
                _display._subagent_parallel = False
                _display._finalize_progress()

    return result


# ---------------------------------------------------------------------------
# todo_write — 创建和管理结构化任务列表
# ---------------------------------------------------------------------------


class TodoItem(BaseModel):
    """单个任务项"""

    content: str = Field(description="Brief description of the task")
    status: str = Field(
        default="pending",
        description="Current status: pending, in_progress, completed, cancelled",
    )
    priority: str = Field(
        default="medium",
        description="Priority level: high, medium, low",
    )


_TODO_STORAGE_DIR = os.path.join(
    os.environ.get(
        "XDG_DATA_HOME",
        os.path.join(os.path.expanduser("~"), ".local", "share"),
    ),
    "chcode",
    "todo",
)


def _todo_path(session_id: str) -> str:
    return os.path.join(_TODO_STORAGE_DIR, f"ses_{session_id}.json")


def _save_todos(session_id: str, todos: list[dict]) -> None:
    os.makedirs(_TODO_STORAGE_DIR, exist_ok=True)
    now = int(time.time() * 1000)
    for i, todo in enumerate(todos):
        todo["position"] = i
        todo["time_updated"] = now
        if "time_created" not in todo:
            todo["time_created"] = now
    with open(_todo_path(session_id), "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)


@tool
def todo_write(
    todos: list[TodoItem],
    runtime: ToolRuntime[SkillAgentContext],
) -> str:
    """\
Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

When to use:
- Complex multistep tasks (3+ steps)
- User provides multiple tasks
- After receiving new instructions
- After completing a task (mark complete, add follow-ups)

When NOT to use:
- Single, straightforward tasks
- Trivial tasks with no organizational benefit
- Purely conversational or informational requests

Task states: pending, in_progress, completed, cancelled
Priority levels: high, medium, low
Only ONE task should be in_progress at any time. Mark tasks complete immediately after finishing.

Args:
        todos: The updated todo list. Each item has content (str), status (str), and priority (str).
    """
    session_id = runtime.context.thread_id or "default"

    # Convert Pydantic models to dicts
    todo_dicts = [t.model_dump() for t in todos]

    # Persist (delete file if empty, matching opencode's delete-then-insert pattern)
    path = _todo_path(session_id)
    if os.path.isfile(path) and len(todo_dicts) == 0:
        os.remove(path)
    else:
        _save_todos(session_id, todo_dicts)

    active = sum(1 for t in todo_dicts if t.get("status") != "completed")

    render_tool_call("todo_write", f"{active} active todos")

    lines = []
    for t in todo_dicts:
        status = t.get("status", "pending")
        content = t.get("content", "")
        priority = t.get("priority", "medium")
        marker = {"completed": "[x]", "in_progress": "[>]", "cancelled": "[-]"}.get(
            status, "[ ]"
        )
        lines.append(f"  {marker} {content} (priority: {priority})")

    output = "\n".join(lines)
    if active > 0:
        output = f"{active} active todo(s):\n{output}"
    else:
        output = "All todos completed." if todo_dicts else "Todo list cleared."

    # 直接打印到终端（chat.py 流式循环不渲染 ToolMessage 结果）
    # 使用 Text 对象避免 [x] 等方括号被 Rich markup 解析器吞掉
    if todo_dicts:
        console.print(Text(f"\n  {active} active todo(s):", style="bold green"))
        for t in todo_dicts:
            status = t.get("status", "pending")
            content = t.get("content", "")
            priority = t.get("priority", "medium")
            marker_map = {
                "completed": "[x]",
                "in_progress": "[>]",
                "cancelled": "[-]",
                "pending": "[ ]",
            }
            marker = marker_map.get(status, "[ ]")
            ps = {"high": "red bold", "medium": "yellow", "low": "dim"}.get(
                priority, ""
            )
            line = Text(f"    {marker} {content} ")
            line.append(f"({priority})", style=ps)
            console.print(line)
    else:
        console.print(Text("  Todo list cleared.", style="dim"))

    return output


ALL_TOOLS = [
    load_skill,
    bash,
    read_file,
    write_file,
    glob,
    grep,
    edit,
    list_dir,
    web_search,
    web_fetch,
    ask_user,
    agent,
    todo_write,
]
