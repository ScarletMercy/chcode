"""tests for chcode.utils.skill_hub — 用假 httpx 客户端,不发真实网络请求。"""

from __future__ import annotations

import io
import zipfile

import httpx
import pytest

from chcode.utils import skill_hub


@pytest.fixture(autouse=True)
def _reset_preferred_host(monkeypatch):
    """隔离模块级 jsdelivr 镜像偏好,避免跨测试泄漏。"""
    monkeypatch.setattr(skill_hub, "_preferred_jsdelivr_host", None)


# ───────────────────────── fakes ─────────────────────────


class _FakeResp:
    def __init__(
        self,
        status_code: int = 200,
        json_data=None,
        text: str = "",
        content: bytes = b"",
    ):
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self.content = content
        self.headers: dict = {}

    def json(self):
        return self._json


class _FakeClient:
    """按 URL 子串匹配路由(插入顺序)的假 AsyncClient。"""

    def __init__(self, routes: dict[str, _FakeResp]):
        self.routes = routes
        self.calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url: str, **kwargs):
        self.calls.append(url)
        for key, resp in self.routes.items():
            if key in url:
                return resp
        return _FakeResp(status_code=404)


class _BoomClient:
    """clawhub 一律抛异常,skills.sh 正常返回。"""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url: str, **kwargs):
        if "clawhub" in url:
            raise httpx.ConnectError("boom")
        return _FakeResp(
            json_data={"skills": [{"id": "o/r/gamma", "name": "Gamma"}]}
        )


def _make_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    return buf.getvalue()


# ───────────────────────── 搜索 ─────────────────────────


async def test_search_skills_merges_and_orders(monkeypatch):
    routes = {
        "clawhub.ai/api/v1/skills": _FakeResp(
            json_data={
                "items": [
                    {"slug": "git-helper", "displayName": "Git Helper", "summary": "g"},
                ]
            }
        ),
        "skills.sh/api/search": _FakeResp(
            json_data={
                "skills": [
                    {"id": "o/r/git-flow", "name": "Git Flow"},
                    {"id": "o/r/git-helper", "name": "Git Helper"},  # 与 clawhub 重名,应去重
                ]
            }
        ),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    results, errors = await skill_hub.search_skills("git")
    names = [r["name"] for r in results]

    assert names == ["Git Helper", "Git Flow"]  # clawhub 在前,重名已去重
    assert errors == []
    assert all(r["source"] in ("clawhub", "skills-sh") for r in results)


async def test_search_ranks_by_relevance_across_sources(monkeypatch):
    """跨源按相关性统一排序:skills.sh 强匹配应排在 clawhub 弱匹配之前。"""
    routes = {
        "clawhub.ai": _FakeResp(
            json_data={
                "items": [
                    {
                        "slug": "weak-desc",
                        "displayName": "Weak",
                        "summary": "something about git inside description",
                    },
                ]
            }
        ),
        "skills.sh": _FakeResp(
            json_data={"skills": [{"id": "o/r/git-commit", "name": "git-commit"}]}
        ),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    results, errors = await skill_hub.search_skills("git")
    names = [r["name"] for r in results]

    assert names == ["git-commit", "Weak"]  # 名称匹配 > 描述匹配,与来源无关
    assert errors == []


async def test_search_skills_clawhub_filters_irrelevant(monkeypatch):
    """clawhub 服务端不搜索(参数被忽略),客户端必须过滤不相关结果。"""
    routes = {
        "clawhub.ai": _FakeResp(
            json_data={
                "items": [
                    {
                        "slug": "campus-pm-coach",
                        "displayName": "校招产品经理教练",
                        "summary": "为校招候选人",
                    },
                    {"slug": "git-helper", "displayName": "Git Helper", "summary": "g"},
                ]
            }
        ),
        "skills.sh": _FakeResp(status_code=404),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    results, errors = await skill_hub.search_skills("git")
    names = [r["name"] for r in results]

    assert names == ["Git Helper"]  # 不相关的校招教练被过滤
    assert errors and "skills.sh" in errors[0]  # skills.sh 404 被如实上报


async def test_search_skills_one_source_fails(monkeypatch):
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _BoomClient()
    )

    results, errors = await skill_hub.search_skills("x")

    assert len(results) == 1
    assert results[0]["name"] == "Gamma"
    assert results[0]["source"] == "skills-sh"
    assert errors and "clawhub" in errors[0]  # 失败源被如实上报


async def test_search_skills_skills_sh_missing_fields(monkeypatch):
    """skills.sh 项缺 id 但有 source+skillId 时应拼出 identifier。"""
    routes = {
        "clawhub.ai": _FakeResp(json_data={"items": []}),
        "skills.sh": _FakeResp(
            json_data={
                "skills": [{"source": "owner/repo", "skillId": "path/to/my-skill"}]
            }
        ),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    results, _ = await skill_hub.search_skills("x")
    assert len(results) == 1
    assert results[0]["identifier"] == "owner/repo/path/to/my-skill"
    assert results[0]["name"] == "my-skill"


async def test_search_skills_non_string_fields(monkeypatch):
    """API 字段为非字符串(数字/嵌套 dict)或纯空格时应优雅清洗或跳过,而非崩溃。"""
    routes = {
        "clawhub.ai": _FakeResp(
            json_data={
                "items": [
                    {"slug": 123},  # slug 非字符串 → 跳过该条
                    {
                        "slug": "git-helper",
                        "displayName": 456,  # 非字符串 → 回退到 slug
                        "summary": {"nested": True},  # 非字符串 → 视为缺失
                    },
                    {
                        "slug": "gitflow",  # 纯空格 displayName → 回退到 slug,不遮蔽回退链
                        "displayName": "   ",
                    },
                ]
            }
        ),
        # id 非字符串且无 source+skillId fallback → 跳过该条
        "skills.sh": _FakeResp(json_data={"skills": [{"id": 789, "name": ["x"]}]}),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    results, errors = await skill_hub.search_skills("git")

    assert len(results) == 2
    assert results[0]["name"] == "git-helper"  # displayName 无效回退 slug
    assert results[0]["description"] == ""
    assert results[1]["name"] == "gitflow"  # 纯空格被视为缺失,回退到 slug
    assert errors == []


async def test_search_skills_malformed_response(monkeypatch):
    """API 返回 200 但 JSON 结构异常(缺 items/skills 键)时应优雅降级,而非解包崩溃。"""
    routes = {
        "clawhub.ai": _FakeResp(json_data={"error": "odd"}),  # dict 无 items 键
        "skills.sh": _FakeResp(json_data="not a dict"),  # 非 dict JSON
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    results, errors = await skill_hub.search_skills("git")

    assert results == []
    assert errors == []


# ───────────────────────── 下载并安装 ─────────────────────────


async def test_download_and_install_clawhub(tmp_path, monkeypatch):
    zip_bytes = _make_zip(
        {"my-skill/SKILL.md": "---\nname: my-skill\ndescription: d\n---\n# body"}
    )
    routes = {
        "/skills/my-skill": _FakeResp(
            json_data={"latestVersion": {"version": "1.0.0"}}
        ),
        "/download": _FakeResp(content=zip_bytes),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    ok, name = await skill_hub.download_and_install("my-skill", "clawhub", tmp_path)

    assert ok is True
    assert name == "my-skill"
    assert (tmp_path / "my-skill" / "SKILL.md").exists()


async def test_download_and_install_clawhub_ambiguous(tmp_path, monkeypatch):
    """409 AMBIGUOUS_SKILL_SLUG:多 owner 同名,应让用户选择后带 owner 重试。"""
    zip_bytes = _make_zip(
        {"my-skill/SKILL.md": "---\nname: my-skill\ndescription: d\n---\n# body"}
    )
    calls: list[tuple[str, dict]] = []

    class _AmbigClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, **kw):
            params = dict(params or {})
            calls.append((url, params))
            if url.endswith("/download"):
                return _FakeResp(content=zip_bytes)
            if params.get("owner"):
                return _FakeResp(json_data={"latestVersion": {"version": "0.1.0"}})
            return _FakeResp(
                status_code=409,
                json_data={
                    "matches": [
                        {"ownerHandle": "alice", "slug": "my-skill"},
                        {"ownerHandle": "bob", "slug": "my-skill"},
                    ]
                },
            )

    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _AmbigClient()
    )

    async def fake_select(message, choices, **kw):
        return choices[0]  # 模拟用户选第一个 @alice/my-skill

    monkeypatch.setattr(skill_hub, "select", fake_select)

    ok, name = await skill_hub.download_and_install("my-skill", "clawhub", tmp_path)

    assert ok is True
    assert name == "my-skill"
    assert (tmp_path / "my-skill" / "SKILL.md").exists()
    # 消歧后,详情与下载请求都带了 owner
    detail_calls = [c for c in calls if c[0].endswith("/skills/my-skill")]
    assert any(c[1].get("owner") == "alice" for c in detail_calls)
    dl_calls = [c for c in calls if c[0].endswith("/download")]
    assert dl_calls and dl_calls[0][1].get("owner") == "alice"


async def test_download_and_install_clawhub_rejects_traversal_name(
    tmp_path, monkeypatch
):
    """zip 内 frontmatter name 含路径穿越(../evil)时应拒绝安装,不创建任何目录。"""
    zip_bytes = _make_zip(
        {"evil/SKILL.md": "---\nname: ../evil\ndescription: d\n---\n# body"}
    )
    routes = {
        "/skills/evil": _FakeResp(
            json_data={"latestVersion": {"version": "1.0.0"}}
        ),
        "/download": _FakeResp(content=zip_bytes),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    ok, msg = await skill_hub.download_and_install("evil", "clawhub", tmp_path)

    assert ok is False
    assert "名称非法" in msg
    assert not any(tmp_path.iterdir())  # 穿越目录未被创建


async def test_download_and_install_clawhub_network_error(tmp_path, monkeypatch):
    """clawhub 网络异常应报「暂不可达」而非「未找到」,便于排查方向。"""

    class _DeadClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            raise httpx.ConnectError("dead")

    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _DeadClient()
    )

    ok, msg = await skill_hub.download_and_install("my-skill", "clawhub", tmp_path)

    assert ok is False
    assert "不可达" in msg


async def test_download_and_install_clawhub_download_network_error(
    tmp_path, monkeypatch
):
    """详情接口正常、下载阶段网络异常,也应报「暂不可达」而非裸异常文本。"""

    class _DownloadDeadClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            if url.endswith("/download"):
                raise httpx.ConnectError("dead")
            return _FakeResp(json_data={"latestVersion": {"version": "1.0.0"}})

    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _DownloadDeadClient()
    )

    ok, msg = await skill_hub.download_and_install("my-skill", "clawhub", tmp_path)

    assert ok is False
    assert "不可达" in msg


async def test_download_and_install_skills_sh(tmp_path, monkeypatch):
    skill_md = "---\nname: my-sh-skill\ndescription: d\n---\n# body"
    routes = {
        "cdn.jsdelivr.net": _FakeResp(text=skill_md),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    ok, name = await skill_hub.download_and_install(
        "owner/repo/some-skill", "skills-sh", tmp_path
    )

    assert ok is True
    assert name == "my-sh-skill"
    written = (tmp_path / "my-sh-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert written == skill_md


async def test_download_and_install_skills_sh_multi_segment_path(
    tmp_path, monkeypatch
):
    """多段 skillId(搜索 fallback 构造的 owner/repo/path/to/my-skill)应完整取 repo 后全部路径。"""
    skill_md = "---\nname: nested-skill\ndescription: d\n---\n# body"
    # 只有完整 skillId 路径 @main/path/to/my-skill/SKILL.md 命中(路由按插入顺序匹配)
    routes = {
        "path/to/my-skill/SKILL.md": _FakeResp(text=skill_md),
        "cdn.jsdelivr.net": _FakeResp(status_code=404),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    ok, name = await skill_hub.download_and_install(
        "owner/repo/path/to/my-skill", "skills-sh", tmp_path
    )

    assert ok is True
    assert name == "nested-skill"  # frontmatter name 优先
    assert (tmp_path / "nested-skill" / "SKILL.md").exists()


async def test_download_and_install_skills_sh_repo_root(tmp_path, monkeypatch):
    """skill 位于仓库根(单 skill 仓库,如 halthelobster/proactive-agent)时回退到根。"""
    skill_md = "---\nname: my-skill\ndescription: d\n---\n# body"
    # 子路径一律 404,只有仓库根 @main/SKILL.md 命中(路由按插入顺序匹配)
    routes = {
        "@main/SKILL.md": _FakeResp(text=skill_md),
        "cdn.jsdelivr.net": _FakeResp(status_code=404),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    ok, name = await skill_hub.download_and_install(
        "owner/repo/my-skill", "skills-sh", tmp_path
    )

    assert ok is True
    assert name == "my-skill"
    assert (tmp_path / "my-skill" / "SKILL.md").exists()


async def test_download_and_install_skills_sh_mirror_fallback(tmp_path, monkeypatch):
    """cdn.jsdelivr.net 不可达时应自动切换到其他 jsdelivr 镜像。"""
    skill_md = "---\nname: my-skill\ndescription: d\n---\n# body"

    class _MirrorClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            if "cdn.jsdelivr.net" in url:
                raise httpx.ConnectError("blocked")
            if "@main/SKILL.md" in url:
                return _FakeResp(text=skill_md)
            return _FakeResp(status_code=404)

    monkeypatch.setattr(skill_hub.httpx, "AsyncClient", lambda **kw: _MirrorClient())

    ok, name = await skill_hub.download_and_install(
        "owner/repo/my-skill", "skills-sh", tmp_path
    )

    assert ok is True
    assert name == "my-skill"
    assert (tmp_path / "my-skill" / "SKILL.md").exists()
    # 成功的镜像被记住,后续优先使用
    assert skill_hub._preferred_jsdelivr_host in ("gcore.jsdelivr.net", "fastly.jsdelivr.net")


async def test_download_and_install_skills_sh_bad_name_falls_back(
    tmp_path, monkeypatch
):
    """frontmatter 的 name 非法时,回退用 skill_id 末段清洗。"""
    skill_md = "---\nname: Bad Name!!\ndescription: d\n---\n# body"
    routes = {
        "cdn.jsdelivr.net": _FakeResp(text=skill_md),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    ok, name = await skill_hub.download_and_install(
        "owner/repo/my-skill", "skills-sh", tmp_path
    )

    assert ok is True
    assert name == "my-skill"  # 回退到 skill_id 末段
    assert (tmp_path / "my-skill" / "SKILL.md").exists()


async def test_download_and_install_skills_sh_reinstall_cleans_stale(
    tmp_path, monkeypatch
):
    """重装同名 skill 先清空旧目录,过期残留文件不保留(对齐 zip 安装语义)。"""
    skill_md = "---\nname: my-skill\ndescription: d\n---\n# body v2"
    stale_dir = tmp_path / "my-skill" / "scripts"
    stale_dir.mkdir(parents=True)
    (stale_dir / "old.py").write_text("old", encoding="utf-8")
    routes = {
        "cdn.jsdelivr.net": _FakeResp(text=skill_md),
    }
    monkeypatch.setattr(
        skill_hub.httpx, "AsyncClient", lambda **kw: _FakeClient(routes)
    )

    ok, name = await skill_hub.download_and_install(
        "owner/repo/my-skill", "skills-sh", tmp_path
    )

    assert ok is True
    assert name == "my-skill"
    assert not (stale_dir / "old.py").exists()  # 旧文件被清理
    written = (tmp_path / "my-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert written == skill_md


async def test_download_and_install_skills_sh_domain_unsupported(tmp_path):
    """skills.volces.com/name 这类非 GitHub 域名源应被明确拒绝(无网络调用)。"""
    ok, msg = await skill_hub.download_and_install(
        "skills.volces.com/proactive-agent", "skills-sh", tmp_path
    )
    assert ok is False
    assert "非 GitHub 源" in msg
    # 3+ 段域名标识同样归入「非 GitHub 源」,而非「无法识别」
    ok, msg = await skill_hub.download_and_install(
        "skills.volces.com/a/b", "skills-sh", tmp_path
    )
    assert ok is False
    assert "非 GitHub 源" in msg


async def test_download_and_install_unknown_source(tmp_path):
    ok, msg = await skill_hub.download_and_install("x", "unknown", tmp_path)
    assert ok is False
    assert "未知来源" in msg


# ───────────────────────── name 清洗 ─────────────────────────


def test_sanitize_name():
    assert skill_hub._sanitize_name("My-Cool_Skill") == "my-cool_skill"
    assert skill_hub._sanitize_name("Skill Name!") == "skill-name"
    assert skill_hub._sanitize_name("123abc") is None  # 不能以数字开头
    assert skill_hub._sanitize_name("---") is None
    assert skill_hub._sanitize_name("!!!") is None  # 全符号 → 空
    # 路径穿越字符被清洗为安全的单段名(无法逃逸目录)
    assert skill_hub._sanitize_name("../../etc") == "etc"
