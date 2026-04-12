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

import locale
import os
import shutil
import subprocess
import re
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from langchain.tools import tool, ToolRuntime
from charset_normalizer import from_bytes
from pydantic import BaseModel, Field
from rich.console import Console
from rich.text import Text

from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI
from chcode.utils.skill_loader import SkillAgentContext
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
console = Console()


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


def _find_git_bash() -> str:
    """通过环境变量 PATH 查找 Git Bash (bash.exe)"""
    # 优先通过 git.exe 所在目录推导 bash.exe，避免命中 WSL 的 bash
    git_path = shutil.which("git")
    if git_path:
        git_bin = os.path.dirname(git_path)
        bash_candidate = os.path.join(git_bin, "bash.exe")
        if os.path.isfile(bash_candidate):
            return bash_candidate
        # cmd 子目录下也可能有
        bash_candidate = os.path.join(git_bin, "..", "bin", "bash.exe")
        if os.path.isfile(bash_candidate):
            return os.path.normpath(bash_candidate)

    # 最后兜底：PATH 中查找 bash.exe
    bash_path = shutil.which("bash")
    if bash_path and os.path.isfile(bash_path):
        return bash_path

    return "bash"


def _get_shell_command(platform: str, command: str):
    if platform == "Windows":
        git_bash = _find_git_bash()
        return git_bash, ["-c", command]
    else:
        return "/bin/sh", ["-c", command]


@tool
def bash(
    command: str,
    platform: Literal["Windows", "Linux", "Mac"],
    runtime: ToolRuntime[SkillAgentContext],
) -> str:
    """
    Execute a shell command with robust multi-encoding output handling.

    Uses Git Bash on Windows, sh on Linux/Mac.
    """
    cwd = str(runtime.context.working_directory)
    system_encoding = locale.getpreferredencoding() or "utf-8"

    def robust_decode(data: bytes) -> str:
        if not data:
            return ""
        if len(data) >= 4:
            bom = data[:4]
            if bom[:3] == b"\xef\xbb\xbf":
                return data[3:].decode("utf-8", errors="replace")
            if bom[:2] in (b"\xff\xfe", b"\xfe\xff"):
                return data.decode("utf-16", errors="replace")
        result = from_bytes(data)
        best = result.best() if result else None
        if best and best.coherence > 0.5:
            return str(best)
        for enc in ["utf-8", "gb18030", system_encoding, "latin-1"]:
            try:
                return data.decode(enc, errors="strict")
            except (UnicodeDecodeError, LookupError):
                continue
        return data.decode(system_encoding, errors="replace")

    try:
        shell_exec, shell_args = _get_shell_command(platform, command)

        proc = subprocess.run(
            [shell_exec] + shell_args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )

        stdout_decoded = robust_decode(proc.stdout)
        stderr_decoded = robust_decode(proc.stderr)

        parts = []
        if proc.returncode == 0:
            parts.append(f"[OK] {command} 执行成功")
        else:
            parts.append(f"[FAILED] Exit code: {proc.returncode}")
        parts.append("")

        if stdout_decoded.strip():
            parts.append(stdout_decoded.rstrip())

        if stderr_decoded.strip():
            if stdout_decoded.strip():
                parts.append("")
            parts.append("--- stderr ---")
            parts.append(stderr_decoded.rstrip())

        if not stdout_decoded.strip() and not stderr_decoded.strip():
            parts.append("(no output)")

        return "bash:\n" + "\n".join(parts)

    except subprocess.TimeoutExpired as e:
        stdout_partial = robust_decode(e.stdout) if e.stdout else ""
        stderr_partial = robust_decode(e.stderr) if e.stderr else ""
        msg = ["bash:\n[FAILED] Command timed out after 300 seconds."]
        if stdout_partial or stderr_partial:
            msg.append("Partial output captured:")
            if stdout_partial:
                msg.append(stdout_partial.rstrip())
            if stderr_partial:
                if stdout_partial:
                    msg.append("")
                msg.append("--- stderr (partial) ---")
                msg.append(stderr_partial.rstrip())
        return "\n".join(msg)

    except Exception as e:
        return f"bash:\n[FAILED] Execution error: {str(e)}"


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
    search_path = resolve_path(path, cwd)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"grep:\n[FAILED] Invalid regex pattern: {e}"

    results = []
    max_results = 50
    files_searched = 0

    try:
        if search_path.is_file():
            files = [search_path]
        else:
            # 搜索所有文本文件，排除常见的二进制/隐藏目录
            files = []
            for p in search_path.rglob("*"):
                if p.is_file():
                    # 排除隐藏文件和常见的非代码目录
                    parts = p.parts
                    if any(
                        part.startswith(".")
                        or part
                        in ("node_modules", "__pycache__", ".git", "venv", ".venv")
                        for part in parts
                    ):
                        continue
                    files.append(p)

        for file_path in files:
            if len(results) >= max_results:
                break

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
                            break

            except (UnicodeDecodeError, PermissionError, IsADirectoryError):
                continue

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

    if not path.exists():
        return f"edit:\n[FAILED] File not found: {file_path}"

    if not path.is_file():
        return f"edit:\n[FAILED] Not a file: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")

        count = content.count(old_string)

        if count == 0:
            return f"edit:\n[FAILED] String not found in file. Make sure the text matches exactly including whitespace."

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
def web_search(
    query: str,
    runtime: ToolRuntime[SkillAgentContext],
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


###  ————————————————————————WebFetch——————————————————————————————————————
# class WebFetchInput(BaseModel):
#     url: str = Field(description="The URL to fetch content from")
#     prompt: str = Field(description="What information to extract from the page")
#
#
# class WebFetchOutput(BaseModel):
#     url: str
#     bytes: int
#     code: int
#     code_text: str
#     result: str
#     duration_ms: int


MAX_CONTENT_LENGTH = 10 * 1024 * 1024
FETCH_TIMEOUT = 60.0
MAX_MARKDOWN_LENGTH = 100_000
MAX_URL_LENGTH = 2000


def _html_to_markdown(html: str) -> str:
    try:
        from markdownify import markdownify as md

        return md(html)
    except ImportError:
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
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
def web_fetch(url: str) -> dict:
    """Fetches content from a specified URL and converts it to text. """
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

        with httpx.Client(
            follow_redirects=True,
            timeout=FETCH_TIMEOUT,
            max_redirects=10,
            headers={
                "Accept": "text/markdown, text/html, */*",
                "User-Agent": "ClaudeToolkit/1.0",
            },
        ) as client:
            response = client.get(url)

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
                markdown_content[:MAX_MARKDOWN_LENGTH] + "\n\n[Content truncated due to length...]"
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


def _select_with_other(question: str, options: list[str]) -> str | None:
    """
    下拉选择 + 「其它」行内输入。

    UX:
      - 上下键选择选项，Enter 确认
      - 「其它」行始终显示输入框: 其它: [自定义输入]
      - 选中「其它」时，输入直接写入右侧输入框
    """
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, UIContent
    from prompt_toolkit.layout.controls import FormattedTextControl, UIControl
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.utils import get_cwidth

    # ─── 自定义控件：选择列表 + 行内输入 ───
    class _SelectWithOtherControl(UIControl):
        def __init__(self, opts: list[str]):
            self.opts = opts
            self.all_opts = opts + ["\u5176\u5b83"]  # options + 其它
            self.selected = 0
            self.buffer = Buffer()

        def is_focusable(self) -> bool:
            return True

        def get_invalidate_events(self):
            yield self.buffer.on_text_changed

        def preferred_height(self, width, max_available_height, wrap_lines, get_line_prefix):
            return len(self.all_opts)

        def create_content(self, width: int, height: int) -> UIContent:
            lines = []
            other_idx = len(self.all_opts) - 1
            other_prefix = "  \u276f \u5176\u5b83: "

            for i, opt in enumerate(self.all_opts):
                if i == self.selected:
                    prefix = "  \u276f "
                else:
                    prefix = "    "

                if opt == "\u5176\u5b83":
                    # 始终显示输入框
                    if self.buffer.text:
                        input_display = self.buffer.text
                    else:
                        input_display = "[\u81ea\u5b9a\u4e49\u8f93\u5165]"
                    line = f"{prefix}{opt}: {input_display}"
                else:
                    line = f"{prefix}{opt}"

                # 选中行高亮
                if i == self.selected:
                    lines.append([("bold", line)])
                else:
                    lines.append([("", line)])

            def get_line(i):
                if i < len(lines):
                    return lines[i]
                return [("", "")]

            # 计算光标位置
            cursor_pos = None
            if self.selected == other_idx:
                # 光标在"其它"输入框内
                # 关键：用 len() 而非 get_cwidth() 计算前缀宽度
                # 因为其他选项可能含 emoji，但"其它: "前缀是纯文本，宽度固定
                prefix_width = len(other_prefix)
                cursor_x = prefix_width + self.buffer.cursor_position
                cursor_pos = Point(x=cursor_x, y=other_idx)

            return UIContent(
                get_line=get_line,
                line_count=len(lines),
                show_cursor=True,
                cursor_position=cursor_pos,
            )

    control = _SelectWithOtherControl(options)

    # ─── 问题标签 ─────────────────────────
    question_text = f"? {question}"
    question_window = Window(
        height=1,
        content=FormattedTextControl(text=question_text),
    )
    control_window = Window(content=control)

    # ─── 按键绑定 ─────────────────────────
    kb = KeyBindings()
    _exiting = False

    @kb.add("up")
    def _up(e):
        nonlocal _exiting
        if _exiting:
            return
        control.selected = max(0, control.selected - 1)
        e.app.invalidate()

    @kb.add("down")
    def _down(e):
        nonlocal _exiting
        if _exiting:
            return
        control.selected = min(len(control.all_opts) - 1, control.selected + 1)
        e.app.invalidate()

    @kb.add("tab")
    def _tab(e):
        nonlocal _exiting
        if _exiting:
            return
        control.selected = (control.selected + 1) % len(control.all_opts)
        e.app.invalidate()

    @kb.add("enter")
    def _enter(e):
        nonlocal _exiting
        _exiting = True
        chosen = control.all_opts[control.selected]
        if chosen == "\u5176\u5b83":
            # 其它选项：提交输入框内容
            text = control.buffer.text.strip()
            if text:
                e.app.exit(result=text)
            else:
                _exiting = False
        else:
            e.app.exit(result=chosen)

    @kb.add("escape")
    def _esc(e):
        e.app.exit(result=None)

    @kb.add("c-c")
    def _cancel(e):
        e.app.exit(result=None)

    # 捕获所有按键 — 选中"其它"时转发到 buffer
    @kb.add(Keys.Any)
    def _any(e):
        nonlocal _exiting
        if _exiting:
            return
        other_idx = len(control.all_opts) - 1
        # 只有选中"其它"时才处理输入
        if control.selected != other_idx:
            return

        data = e.data
        if data == "\r":  # enter 已处理
            return

        buf = control.buffer
        # 处理退格
        if data == "\x7f" or data == "\x08":
            if buf.cursor_position > 0:
                buf.delete_before_cursor()
            e.app.invalidate()
            return
        # 可打印字符
        if len(data) == 1 and data >= " ":
            buf.insert_text(data)
            e.app.invalidate()

    # ─── 构建并运行 ───────────────────────
    layout = Layout(HSplit([question_window, control_window]))
    app = Application(layout=layout, key_bindings=kb, full_screen=False)
    return app.run()


@tool
def ask_user(
    question: str,
    options: list[str],
    is_multiple: bool = False,
) -> str:
    """
    Ask the user a question interactively with predefined options.

    Use this when you need clarification, user preferences, or choices
    before proceeding. The user will see a dropdown or checkbox in the terminal.

    Args:
        question: The question to ask the user
        options: List of options for the user to choose from
        is_multiple: If True, allow selecting multiple options; otherwise single select
    """
    import questionary

    console.print(Text(f"\n[ask_user] {question}", style="bold yellow"))

    if not options:
        answer = questionary.text("请输入: ").ask()
        if answer is None:
            return "user_answer:\n(用户取消)"
        return f"user_answer:\n{answer}"

    try:
        if is_multiple:
            selected = questionary.checkbox(
                "选择（空格选择，回车确认）:",
                choices=options,
            ).ask()
            if selected is None:
                return "user_answer:\n(用户取消)"
            result = ", ".join(selected)
        else:
            answer = _select_with_other(question, options)
            if answer is None:
                return "user_answer:\n(用户取消)"
            result = answer
        return f"user_answer:\n{result}"
    except Exception as e:
        return f"user_answer:\n(询问失败: {e})"


ALL_TOOLS = [load_skill, bash, read_file, write_file, glob, grep, edit, list_dir, web_search, web_fetch, ask_user]
