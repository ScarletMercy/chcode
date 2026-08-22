"""Skill Hub — 从远程 hub(clawhub.ai / skills.sh)搜索并安装 skill

经 /skill 菜单的「技能市场」入口调用,无 LLM 参与,不污染主对话上下文。
下载后复用 skill_loader 已有的安装逻辑(含路径穿越防护)。
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

import httpx
from rich.table import Table

from chcode.display import console
from chcode.i18n import t
from chcode.prompts import select, text
from chcode.utils.frontmatter import parse_frontmatter

# 复用技能管理里「选择安装位置」的公共交互(skill_manager 对本模块是函数内惰性导入,无循环)
from chcode.utils.skill_manager import select_install_root
from chcode.utils.skill_loader import install_skill, validate_skill_package

CLAWHUB_BASE = "https://clawhub.ai/api/v1"
SKILLS_SH_SEARCH = "https://skills.sh/api/search"
# clawhub /skills 不支持服务端搜索(参数被忽略),拉全量目录后客户端过滤
CLAWHUB_CATALOG_LIMIT = 200
# 用 jsdelivr CDN 取 GitHub 内容(raw.githubusercontent.com 常不可达);
# cdn 在部分网络也不稳定,依次尝试多镜像,记住成功的 host
_JSDELIVR_HOSTS = ["cdn.jsdelivr.net", "gcore.jsdelivr.net", "fastly.jsdelivr.net"]
_preferred_jsdelivr_host: str | None = None
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def _term_score(term: str, field: str) -> int:
    """单个查询词与单字段的匹配强度:完全相等 > 前缀 > 整词 > 子串。"""
    if not field:
        return 0
    if field == term:
        return 8
    if field.startswith(term):
        return 6
    tokens = re.split(r"[^a-z0-9\u4e00-\u9fff]+", field)
    if term in tokens:
        return 5
    if term in field:
        return 3
    return 0


def _relevance_score(query: str, name: str, identifier: str, description: str) -> int:
    """客户端相关性打分,0 = 不相关(clawhub 服务端不搜索,必须本地过滤)。"""
    terms = [
        term for term in re.split(r"[^a-z0-9\u4e00-\u9fff]+", query.lower()) if term
    ]
    if not terms:
        return 1
    score = 0
    for term in terms:
        s = _term_score(term, (name or "").lower())
        if not s:
            s = _term_score(term, (identifier or "").lower())
        if not s and term in (description or "").lower():
            s = 1  # 描述命中最弱
        score += s
    return score


def _clean_str(value) -> str:
    """API 字段清洗:非字符串一律视为缺失,避免 .lower()/.count() 等崩溃。"""
    return value.strip() if isinstance(value, str) else ""


# ───────────────────────── 搜索 ─────────────────────────


_SOURCE_RANK = {"clawhub": 0, "skills-sh": 1}


async def search_skills(query: str, limit: int = 20) -> tuple[list[dict], list[str]]:
    """并发搜索 clawhub 与 skills.sh,跨源按相关性统一排序后合并去重。

    Returns:
        (结果列表, 失败数据源描述列表)——单源失败不影响另一源,但会如实上报。
    """
    (clawhub_results, cl_err), (sh_results, sh_err) = await asyncio.gather(
        _search_clawhub(query, limit),
        _search_skills_sh(query, limit),
    )
    errors = [e for e in (cl_err, sh_err) if e]
    merged = clawhub_results + sh_results
    merged.sort(
        key=lambda r: (
            -r.get("_score", 0),
            _SOURCE_RANK.get(r["source"], 9),
            r["name"].lower(),
        )
    )
    seen: set[str] = set()
    out: list[dict] = []
    for r in merged:
        key = r["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        r.pop("_score", None)
        out.append(r)
    return out[:limit], errors


async def _search_clawhub(query: str, limit: int) -> tuple[list[dict], str]:
    try:
        async with httpx.AsyncClient() as client:
            # search 参数被服务端忽略,拉全量目录客户端过滤
            resp = await client.get(
                f"{CLAWHUB_BASE}/skills",
                params={"limit": CLAWHUB_CATALOG_LIMIT},
                timeout=20,
            )
        if resp.status_code != 200:
            return [], f"clawhub: HTTP {resp.status_code}"
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        return [], f"clawhub: {type(e).__name__}"

    items = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], ""
    scored: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        name = (
            _clean_str(item.get("displayName")) or _clean_str(item.get("name")) or slug
        )
        desc = _clean_str(item.get("summary")) or _clean_str(item.get("description"))
        score = _relevance_score(query, name, slug, desc)
        if score <= 0:
            continue  # 不相关结果直接丢弃,避免垃圾挤掉 skills.sh 相关结果
        scored.append(
            {
                "name": name,
                "description": desc,
                "identifier": slug,
                "source": "clawhub",
                "_score": score,
            }
        )
    return scored[:limit], ""


async def _search_skills_sh(query: str, limit: int) -> tuple[list[dict], str]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                SKILLS_SH_SEARCH,
                params={"q": query, "limit": limit},
                timeout=15,
            )
        if resp.status_code != 200:
            return [], f"skills.sh: HTTP {resp.status_code}"
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        return [], f"skills.sh: {type(e).__name__}"

    items = data.get("skills", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], ""
    results: list[dict] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        ident = item.get("id")
        if not isinstance(ident, str):
            ident = ""
        if not ident or ident.count("/") < 2:
            src = item.get("source")
            sid = item.get("skillId")
            if isinstance(src, str) and isinstance(sid, str):
                ident = f"{src}/{sid}"
            else:
                continue
        name = _clean_str(item.get("name")) or ident.split("/")[-1]
        desc = _clean_str(item.get("description")) or "from skills.sh"
        # skills.sh 服务端已过滤,客户端只打分用于排序,不丢弃
        results.append(
            {
                "name": name,
                "description": desc,
                "identifier": ident,
                "source": "skills-sh",
                "_score": _relevance_score(query, name, ident, desc),
            }
        )
    return results, ""


# ───────────────────────── 下载并安装 ─────────────────────────


async def download_and_install(
    identifier: str, source: str, target_root: Path
) -> tuple[bool, str]:
    """下载并安装 skill。

    成功返回 (True, name),失败返回 (False, 原因)。
    clawhub: 下载 zip,复用 validate_skill_package + install_skill(含路径穿越防护)。
    skills.sh: 抓取 SKILL.md,name 经正则校验后写入单文件。
    """
    try:
        if source == "clawhub":
            return await _install_from_clawhub(identifier, target_root)
        if source == "skills-sh":
            return await _install_from_skills_sh(identifier, target_root)
        return False, t("skillhub.err_unknown_source", source=source)
    except Exception as e:
        return False, str(e)


async def _get_detail(client: httpx.AsyncClient, slug: str, owner: str):
    try:
        return await client.get(
            f"{CLAWHUB_BASE}/skills/{slug}",
            params={"owner": owner} if owner else None,
            timeout=20,
        )
    except httpx.HTTPError:
        return None


async def _resolve_ambiguous_slug(slug: str, resp) -> str:
    """409 AMBIGUOUS_SKILL_SLUG:多个 owner 同名,让用户选择。返回 ownerHandle,取消返回空串。"""
    try:
        matches = resp.json().get("matches", [])
    except ValueError:
        return ""
    owners: list[str] = []
    options: list[str] = []
    for m in matches:
        if isinstance(m, dict) and m.get("ownerHandle"):
            owners.append(m["ownerHandle"])
            options.append(f"@{m['ownerHandle']}/{m.get('slug', slug)}")
    if not options:
        return ""
    back_label = t("common.back")
    chosen = await select(t("skillhub.select_owner"), options + [back_label])
    if not chosen or chosen == back_label:
        return ""
    idx = (options + [back_label]).index(chosen)
    return owners[idx]


async def _install_from_clawhub(slug: str, target_root: Path) -> tuple[bool, str]:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        owner = ""
        resp = await _get_detail(client, slug, owner)
        if resp is not None and resp.status_code == 409:
            # 多个 owner 同名,让用户选择后带 owner 重试
            owner = await _resolve_ambiguous_slug(slug, resp)
            if not owner:
                return False, t("skillhub.err_ambiguous_cancelled", slug=slug)
            resp = await _get_detail(client, slug, owner)
        if resp is None:
            return False, t("skillhub.err_network", slug=slug)
        if resp.status_code != 200:
            return False, t("skillhub.err_not_found", slug=slug)

        try:
            skill_data = resp.json()
        except ValueError:
            return False, t("skillhub.err_invalid_response", slug=slug)
        # 响应可能是 {"skill": {...}, "latestVersion": {...}} 嵌套结构
        if isinstance(skill_data, dict) and isinstance(skill_data.get("skill"), dict):
            merged = dict(skill_data["skill"])
            if skill_data.get("latestVersion") is not None:
                merged.setdefault("latestVersion", skill_data["latestVersion"])
            skill_data = merged
        if not isinstance(skill_data, dict):
            return False, t("skillhub.err_invalid_response", slug=slug)

        version = _resolve_latest_version(skill_data)
        if not version:
            versions = await _get_json(
                client,
                f"{CLAWHUB_BASE}/skills/{slug}/versions",
                params={"owner": owner} if owner else None,
            )
            if (
                isinstance(versions, list)
                and versions
                and isinstance(versions[0], dict)
            ):
                version = versions[0].get("version")
        if not version:
            return False, t("skillhub.err_no_version", slug=slug)

        # 下载 zip(带 owner 消歧,处理 429 限流)
        dl_params: dict = {"slug": slug, "version": version}
        if owner:
            dl_params["owner"] = owner
        resp = None
        for attempt in range(3):
            try:
                resp = await client.get(
                    f"{CLAWHUB_BASE}/download",
                    params=dl_params,
                    timeout=30,
                )
            except httpx.HTTPError:
                return False, t("skillhub.err_network", slug=slug)
            if resp.status_code != 429:
                break
            if attempt < 2:  # 最后一次 429 后不再空等
                try:
                    delay = int(resp.headers.get("retry-after", "5"))
                except (ValueError, TypeError):
                    delay = 5
                await asyncio.sleep(min(delay, 15))
        if resp is None or resp.status_code != 200:
            code = resp.status_code if resp is not None else "?"
            return False, t("skillhub.err_download_failed", slug=slug, code=code)

    # 创建/写入/校验均在 try 内,中途任何异常 finally 都能清理临时文件
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
            tmp_path = tf.name
            tf.write(resp.content)
        info = validate_skill_package(tmp_path)
        if not info:
            return False, t("skillhub.err_invalid_package", slug=slug)
        # frontmatter name 会成为目标目录名,必须落在 target_root 内(防路径穿越)
        if (
            not (target_root / info["name"])
            .resolve()
            .is_relative_to(target_root.resolve())
        ):
            return False, t("skillhub.err_invalid_package_name", slug=slug)
        if not install_skill(tmp_path, target_root):
            return False, t("skillhub.err_write_failed", slug=slug)
        return True, info["name"]
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


async def _install_from_skills_sh(
    identifier: str, target_root: Path
) -> tuple[bool, str]:
    global _preferred_jsdelivr_host
    parts = identifier.split("/")
    # GitHub 源: owner/repo/skillId(3 段,首段无点)。
    # 域名源(如 skills.volces.com/name,首段含点)无 GitHub 路径,暂不支持直接安装。
    if len(parts) >= 3 and "." not in parts[0]:
        repo = f"{parts[0]}/{parts[1]}"
        skill_id = "/".join(parts[2:])  # skillId 可能是多段路径(如 path/to/my-skill)
    elif "." in parts[0]:
        return False, t("skillhub.err_non_github", host=parts[0])
    else:
        return False, t("skillhub.err_unrecognized_id", identifier=identifier)

    base_name = skill_id.split("/")[-1]
    # skill 在仓库里可能位于多个约定路径,逐一尝试;
    # 仓库根放最后(整个仓库即一个 skill 的场景,如 halthelobster/proactive-agent)
    candidate_paths: list[str] = []
    for p in (
        f"skills/{base_name}",
        f".agents/skills/{base_name}",
        f".claude/skills/{base_name}",
        skill_id,
        "",  # 仓库根
    ):
        rel = p.strip("/")
        if rel not in candidate_paths:
            candidate_paths.append(rel)

    hosts = list(_JSDELIVR_HOSTS)
    if _preferred_jsdelivr_host in hosts:
        hosts.remove(_preferred_jsdelivr_host)
        hosts.insert(0, _preferred_jsdelivr_host)

    content = None
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for host in hosts:
            host_dead = False
            for branch in ("main", "master"):
                for cpath in candidate_paths:
                    prefix = f"{cpath}/" if cpath else ""
                    url = f"https://{host}/gh/{repo}@{branch}/{prefix}SKILL.md"
                    try:
                        resp = await client.get(url, timeout=12)
                    except httpx.HTTPError:
                        host_dead = True  # 该镜像不可达,立即换下一个
                        break
                    if resp.status_code == 200 and resp.text.strip():
                        content = resp.text
                        break
                if host_dead or content:
                    break
            if content:
                _preferred_jsdelivr_host = host
                break
    if content is None:
        return False, t("skillhub.err_skillmd_not_found", identifier=identifier)

    fm = parse_frontmatter(content)
    name = None
    if fm and isinstance(fm.frontmatter.get("name"), str):
        candidate = fm.frontmatter["name"].strip().lower()
        if _NAME_RE.match(candidate):
            name = candidate
    if not name:
        name = _sanitize_name(base_name)
    if not name:
        return False, t("skillhub.err_invalid_skill_name", name=identifier)

    # name 已经正则校验,无斜杠/..,不会逃逸 target_root
    skill_dir = target_root / name
    if skill_dir.exists():
        # 对齐 zip 安装语义:重装先清空旧目录,避免残留过期文件
        import shutil

        shutil.rmtree(skill_dir)
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return True, name


def _resolve_latest_version(skill_data: dict) -> str | None:
    latest = skill_data.get("latestVersion")
    if isinstance(latest, dict):
        v = latest.get("version")
        if isinstance(v, str) and v:
            return v
    tags = skill_data.get("tags")
    if isinstance(tags, dict):
        v = tags.get("latest")
        if isinstance(v, str) and v:
            return v
    return None


async def _get_json(
    client: httpx.AsyncClient, url: str, timeout: int = 20, params: dict | None = None
):
    try:
        resp = await client.get(url, params=params, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except (httpx.HTTPError, ValueError):
        return None


def _sanitize_name(raw: str) -> str | None:
    """把任意字符串清洗成合法 skill name(小写、[a-z0-9_-]),不合法返回 None。"""
    candidate = re.sub(r"[^a-z0-9_-]", "-", raw.strip().lower())
    candidate = re.sub(r"-+", "-", candidate).strip("-")
    if candidate and _NAME_RE.match(candidate):
        return candidate
    return None


# ───────────────────────── 交互 UI ─────────────────────────


async def manage_skill_hub(workplace_path: Path) -> None:
    """Skill Hub 主流程:搜索 → 选择 → 安装。"""
    query = await text(t("skillhub.input_query"))
    if not query or not query.strip():
        return
    query = query.strip()

    console.print(f"[yellow]{t('skillhub.searching')}[/yellow]")
    results, source_errors = await search_skills(query)
    if source_errors:
        console.print(
            f"[yellow]{t('skillhub.source_failed', sources='; '.join(source_errors))}[/yellow]"
        )
    if not results:
        console.print(f"[yellow]{t('skillhub.no_results')}[/yellow]")
        return

    table = Table(title=t("skillhub.results_title"))
    table.add_column(t("skillhub.col_name"), style="cyan")
    table.add_column(t("skillhub.col_source"), style="green")
    table.add_column(t("skillhub.col_desc"), style="white")
    for r in results:
        desc = r["description"]
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(r["name"], r["source"], desc)
    console.print(table)

    back_label = t("common.back")
    options = [f"{r['name']} ({r['source']})" for r in results] + [back_label]
    selected = await select(t("skillhub.select_to_install"), options)
    if not selected or selected == back_label:
        return
    # 选项后缀固定为 " ({source})";skill 名不含 ASCII " ("(即便含中文括号也不冲突)
    selected_name = selected.split(" (")[0]
    skill = next((r for r in results if r["name"] == selected_name), None)
    if not skill:
        return

    target_root = await select_install_root(workplace_path)
    if target_root is None:
        return

    console.print(f"[yellow]{t('skill.installing')}[/yellow]")
    ok, msg = await download_and_install(
        skill["identifier"], skill["source"], target_root
    )
    if ok:
        console.print(f"[green]{t('skill.install_success', name=msg)}[/green]")
    else:
        console.print(f"[red]{t('skillhub.install_failed', error=msg)}[/red]")
