"""
UI 文案目录（i18n）— zh / en 内联在同一 key 下。

结构：每个 key 对应 {"zh": ..., "en": ...}，两种语言紧挨在一起，避免漏译/漂移。
约定：
- key 用稳定英文点分命名，如 "cmd.new_session"、"model.manage"
- t() 通过 CATALOGS 查表；缺失时回退到中文再回退到 key 本身（永不抛异常）
- 选项类字符串须与历史完全一致（测试 mock 依赖）
"""

MESSAGES = {
    # ─── 语言选择（双语展示，两个 catalog 相同）───
    "lang.saved_zh": {
        "zh": "已设置语言：中文",
        "en": "已设置语言：中文",
    },
    "lang.saved_en": {
        "zh": "Language set to: English",
        "en": "Language set to: English",
    },
    # ─── 首次运行向导 ───
    "firstrun.panel": {
        "zh": "[bold]ChCode[/bold] — 终端 AI 编程助手\n\n首次运行需要配置 AI 模型连接。\n设置环境变量后可自动检测（推荐），或手动填写配置。",
        "en": "[bold]ChCode[/bold] — Terminal AI coding assistant\n\nFirst run requires configuring an AI model connection.\nSet environment variables for auto-detection (recommended), or fill in manually.",
    },
    "firstrun.select_method": {
        "zh": "选择配置方式:",
        "en": "Select configuration method:",
    },
    "firstrun.env_detected": {
        "zh": "{name} (检测到 {env_var})",
        "en": "{name} (detected {env_var})",
    },
    "firstrun.modelscope_quick": {
        "zh": "魔搭快捷配置...",
        "en": "ModelScope quick setup...",
    },
    "firstrun.manual": {
        "zh": "手动配置...",
        "en": "Manual setup...",
    },
    "firstrun.exit": {
        "zh": "退出",
        "en": "Exit",
    },
    "firstrun.exit_hint": {
        "zh": "设置环境变量后重新运行，或执行 chcode config new 手动配置",
        "en": "Set environment variables and rerun, or run 'chcode config new' to configure manually",
    },
    "firstrun.no_env_key": {
        "zh": "未检测到环境变量中的 API Key",
        "en": "No API keys detected in environment variables",
    },
    "firstrun.select": {
        "zh": "选择:",
        "en": "Select:",
    },
    "firstrun.env_hint": {
        "zh": "提示: 在环境变量中设置 API Key 后重新运行，例如:",
        "en": "Tip: set an API key in environment variables and rerun, for example:",
    },
    "firstrun.env_example": {
        "zh": "  set BIGMODEL_API_KEY=your_key",
        "en": "  set BIGMODEL_API_KEY=your_key",
    },
    "firstrun.manual_cmd_hint": {
        "zh": "或执行 chcode config new 手动配置",
        "en": "or run 'chcode config new' to configure manually",
    },
    "firstrun.config_done": {
        "zh": "配置完成: {model}",
        "en": "Configuration complete: {model}",
    },
    # ─── 模型配置 ───
    "model.method_select": {
        "zh": "配置方式:",
        "en": "Configuration method:",
    },
    "model.context_length": {
        "zh": "上下文长度（token 数，默认 1M）:",
        "en": "Context length (tokens, default 1M):",
    },
    "model.saved": {
        "zh": "模型配置已保存: {model}",
        "en": "Model configuration saved: {model}",
    },
    "model.multimodal_ask": {
        "zh": "该模型是否为多模态（视觉）模型？",
        "en": "Is this a multimodal (vision) model?",
    },
    "model.vision_added_default": {
        "zh": "已加入视觉模型清单(默认): {model}（图片将直接处理、vision 工具将停用）",
        "en": "Added to vision model list (default): {model} (images processed directly, vision tool disabled)",
    },
    "model.vision_added_fallback": {
        "zh": "已加入视觉模型清单(备用): {model}（图片将直接处理、vision 工具将停用）",
        "en": "Added to vision model list (fallback): {model} (images processed directly, vision tool disabled)",
    },
    "model.vision_duplicate": {
        "zh": "未写入视觉配置（已存在相同配置）",
        "en": "Not written to vision config (identical config already exists)",
    },
    "model.vision_failed": {
        "zh": "视觉模型配置失败（不影响主配置）: {error}",
        "en": "Vision model configuration failed (does not affect main config): {error}",
    },
    "model.edit_none": {
        "zh": "没有当前模型配置，请新建",
        "en": "No current model configuration, please create one",
    },
    "model.updated": {
        "zh": "模型配置已更新: {model}",
        "en": "Model configuration updated: {model}",
    },
    "model.switch_no_default": {
        "zh": "请先配置默认模型",
        "en": "Please configure a default model first",
    },
    "model.switch_no_fallback": {
        "zh": "没有备用模型可切换",
        "en": "No fallback models available to switch",
    },
    "model.current_default_tag": {
        "zh": " (当前默认)",
        "en": " (current default)",
    },
    "model.select_to_use": {
        "zh": "选择要使用的模型:",
        "en": "Select a model to use:",
    },
    "model.switch_confirm": {
        "zh": "确定切换到 {model}？当前默认将移至备用列表",
        "en": "Confirm switch to {model}? Current default will move to fallback list",
    },
    "model.switched": {
        "zh": "已切换到: {model}",
        "en": "Switched to: {model}",
    },
    # ─── ModelScope 快捷配置 ───
    "modelscope.testing": {
        "zh": "测试连接中...",
        "en": "Testing connection...",
    },
    "modelscope.connect_failed": {
        "zh": "连接测试失败: {detail}",
        "en": "Connection test failed: {detail}",
    },
    "modelscope.config_done": {
        "zh": "配置完成: {model} (默认)",
        "en": "Configuration complete: {model} (default)",
    },
    "modelscope.fallback_count": {
        "zh": "备用模型 ({count} 个): {names}",
        "en": "Fallback models ({count}): {names}",
    },
    "modelscope.vision_auto": {
        "zh": "视觉模型已自动配置: {model}",
        "en": "Vision model auto-configured: {model}",
    },
    "modelscope.vision_unknown": {
        "zh": "未知",
        "en": "unknown",
    },
    # ─── 连接测试 / 加载错误 ───
    "connection.testing": {
        "zh": "测试连接中...",
        "en": "Testing connection...",
    },
    "connection.failed": {
        "zh": "连接测试失败: {error}",
        "en": "Connection test failed: {error}",
    },
    "config.load_failed": {
        "zh": "Warning: 加载 {path} 失败: {e}",
        "en": "Warning: failed to load {path}: {e}",
    },
    # ─── Tavily ───
    "tavily.detected_env": {
        "zh": "检测到 TAVILY_API_KEY 环境变量，已自动配置 Tavily",
        "en": "Detected TAVILY_API_KEY environment variable, Tavily auto-configured",
    },
    "tavily.configured": {
        "zh": "已配置 Tavily: {key}",
        "en": "Tavily configured: {key}",
    },
    "tavily.ask_configure": {
        "zh": "是否配置 Tavily 搜索引擎?",
        "en": "Configure Tavily search engine?",
    },
    "tavily.skipped": {
        "zh": "已跳过，后续可通过 /search 命令配置",
        "en": "Skipped, configure later via the /search command",
    },
    "tavily.input_key": {
        "zh": "请输入 Tavily API Key:",
        "en": "Enter Tavily API Key:",
    },
    "tavily.saved": {
        "zh": "Tavily API Key 已保存并生效",
        "en": "Tavily API Key saved and active",
    },
    # ─── LangSmith ───
    "langsmith.detected_env": {
        "zh": "检测到 LANGSMITH_API_KEY 环境变量，已自动配置 LangSmith",
        "en": "Detected LANGSMITH_API_KEY environment variable, LangSmith auto-configured",
    },
    "langsmith.ask_configure": {
        "zh": "是否配置 LangSmith 追踪?",
        "en": "Configure LangSmith tracing?",
    },
    "langsmith.skipped": {
        "zh": "已跳过，后续可通过 /langsmith 命令配置",
        "en": "Skipped, configure later via the /langsmith command",
    },
    "langsmith.input_project": {
        "zh": "请输入 LangSmith 项目名称:",
        "en": "Enter LangSmith project name:",
    },
    "langsmith.input_key": {
        "zh": "请输入 LangSmith API Key:",
        "en": "Enter LangSmith API Key:",
    },
    "langsmith.saved": {
        "zh": "LangSmith 配置已写入环境变量，重启后生效",
        "en": "LangSmith configuration written to environment variables, takes effect after restart",
    },
    # ─── 通用 ───
    "common.yes": {
        "zh": "是",
        "en": "Yes",
    },
    "common.no": {
        "zh": "否",
        "en": "No",
    },
    "common.cancelled": {
        "zh": "已取消",
        "en": "Cancelled",
    },
    "common.back": {
        "zh": "返回",
        "en": "Back",
    },
    # ─── 视觉模型配置 ───
    "vision.menu": {
        "zh": "视觉模型配置:",
        "en": "Vision model config:",
    },
    "vision.view": {
        "zh": "查看当前配置",
        "en": "View current config",
    },
    "vision.reconfigure": {
        "zh": "重新配置",
        "en": "Reconfigure",
    },
    "vision.switch": {
        "zh": "切换模型",
        "en": "Switch model",
    },
    "vision.configure": {
        "zh": "配置视觉模型",
        "en": "Configure vision model",
    },
    "vision.unconfigured_ask": {
        "zh": "视觉模型未配置，是否现在配置？",
        "en": "Vision model not configured. Configure now?",
    },
    "vision.select_key_source": {
        "zh": "选择 API Key 来源:",
        "en": "Select API Key source:",
    },
    "vision.use_env_token": {
        "zh": "使用环境变量 ModelScopeToken ({masked})",
        "en": "Use env var ModelScopeToken ({masked})",
    },
    "vision.manual_key": {
        "zh": "手动输入 API Key",
        "en": "Enter API Key manually",
    },
    "vision.input_key": {
        "zh": "输入 ModelScope API Key:",
        "en": "Enter ModelScope API Key:",
    },
    "vision.select_default": {
        "zh": "选择默认视觉模型:",
        "en": "Select default vision model:",
    },
    "vision.config_done": {
        "zh": "视觉模型配置完成: {model} (默认)",
        "en": "Vision model configured: {model} (default)",
    },
    "vision.fallback_count": {
        "zh": "备用模型 ({count} 个): {names}",
        "en": "Fallback models ({count}): {names}",
    },
    "vision.no_default": {
        "zh": "请先配置默认视觉模型",
        "en": "Please configure a default vision model first",
    },
    "vision.no_fallback": {
        "zh": "没有备用视觉模型可切换",
        "en": "No fallback vision models to switch",
    },
    "vision.select_to_use": {
        "zh": "选择要使用的视觉模型:",
        "en": "Select a vision model to use:",
    },
    "vision.switch_confirm": {
        "zh": "确定切换到 {model}？当前默认将移至备用列表",
        "en": "Confirm switch to {model}? Current default will move to fallback list",
    },
    "vision.switched": {
        "zh": "已切换到: {model}",
        "en": "Switched to: {model}",
    },
    "vision.not_configured": {
        "zh": "未配置视觉模型",
        "en": "Vision model not configured",
    },
    "vision.default_label": {
        "zh": "默认视觉模型:",
        "en": "Default vision model:",
    },
    "vision.fallback_table_title": {
        "zh": "备用视觉模型",
        "en": "Fallback vision models",
    },
    "vision.col_model": {
        "zh": "模型",
        "en": "Model",
    },
    "vision.col_status": {
        "zh": "状态",
        "en": "Status",
    },
    "vision.no_fallback_dim": {
        "zh": "无备用模型",
        "en": "No fallback models",
    },
    "vision.unknown": {
        "zh": "未知",
        "en": "unknown",
    },
    # ─── 技能管理 ───
    "skill.menu": {
        "zh": "技能管理:",
        "en": "Skill management:",
    },
    "skill.view_installed": {
        "zh": "查看已安装技能",
        "en": "View installed skills",
    },
    "skill.install_new": {
        "zh": "安装新技能",
        "en": "Install a new skill",
    },
    "skill.none_installed": {
        "zh": "没有发现已安装的技能",
        "en": "No installed skills found",
    },
    "skill.installed_table_title": {
        "zh": "已安装技能",
        "en": "Installed skills",
    },
    "skill.col_name": {
        "zh": "名称",
        "en": "Name",
    },
    "skill.col_type": {
        "zh": "类型",
        "en": "Type",
    },
    "skill.col_desc": {
        "zh": "描述",
        "en": "Description",
    },
    "skill.col_path": {
        "zh": "路径",
        "en": "Path",
    },
    "skill.select_to_operate": {
        "zh": "选择技能进行操作:",
        "en": "Select a skill to operate on:",
    },
    "skill.action_on": {
        "zh": "对技能 '{name}' 的操作:",
        "en": "Action on skill '{name}':",
    },
    "skill.view_detail": {
        "zh": "查看详情",
        "en": "View details",
    },
    "skill.delete": {
        "zh": "删除技能",
        "en": "Delete skill",
    },
    "skill.skill_file_not_exist": {
        "zh": "技能文件不存在",
        "en": "Skill file does not exist",
    },
    "skill.detail_title": {
        "zh": "技能: {name}",
        "en": "Skill: {name}",
    },
    "skill.delete_confirm": {
        "zh": "确定删除技能 '{name}'？此操作不可撤销！",
        "en": "Confirm delete skill '{name}'? This cannot be undone!",
    },
    "skill.deleted": {
        "zh": "技能 '{name}' 已删除",
        "en": "Skill '{name}' deleted",
    },
    "skill.delete_failed": {
        "zh": "删除失败: {error}",
        "en": "Delete failed: {error}",
    },
    "skill.input_archive_path": {
        "zh": "输入技能压缩包路径 (.zip/.tar.gz/.tgz):",
        "en": "Enter skill archive path (.zip/.tar.gz/.tgz):",
    },
    "skill.file_not_exist": {
        "zh": "文件不存在",
        "en": "File does not exist",
    },
    "skill.validating": {
        "zh": "验证技能包...",
        "en": "Validating skill package...",
    },
    "skill.invalid_package": {
        "zh": "无效的技能包，必须包含 SKILL.md",
        "en": "Invalid skill package, must contain SKILL.md",
    },
    "skill.validate_failed": {
        "zh": "验证技能包失败: {error}",
        "en": "Skill package validation failed: {error}",
    },
    "skill.install_error": {
        "zh": "安装技能失败: {error}",
        "en": "Skill installation failed: {error}",
    },
    "skill.select_install_location": {
        "zh": "选择安装位置:",
        "en": "Select install location:",
    },
    "skill.install_project": {
        "zh": "项目级 (当前工作目录)",
        "en": "Project-level (current workdir)",
    },
    "skill.install_global": {
        "zh": "全局级 (用户目录)",
        "en": "Global-level (user dir)",
    },
    "skill.installing": {
        "zh": "安装中...",
        "en": "Installing...",
    },
    "skill.install_success": {
        "zh": "技能 '{name}' 安装成功！",
        "en": "Skill '{name}' installed successfully!",
    },
    "skill.install_failed": {
        "zh": "安装失败",
        "en": "Installation failed",
    },
    "skill.type.project": {
        "zh": "项目",
        "en": "Project",
    },
    "skill.type.global": {
        "zh": "全局",
        "en": "Global",
    },
    # ─── chat REPL ───
    "chat.empty_group": {
        "zh": "(空消息组)",
        "en": "(empty message group)",
    },
    "chat.goodbye": {
        "zh": "再见！",
        "en": "Goodbye!",
    },
    "chat.unknown_command": {
        "zh": "未知命令: {command}，输入 /help 查看帮助",
        "en": "Unknown command: {command}, type /help for help",
    },
    "chat.new_session_started": {
        "zh": "新会话已开始",
        "en": "New session started",
    },
    "chat.status.model_unset": {
        "zh": "未设置",
        "en": "not set",
    },
    "chat.status.common_mode": {
        "zh": "普通模式",
        "en": "Common mode",
    },
    "chat.status.yolo_mode": {
        "zh": "YOLO 模式",
        "en": "YOLO mode",
    },
    "chat.status.modelscope_quota": {
        "zh": "\n<ansicyan>魔搭今日免费额度剩余: 全局 {total} │ 模型({model}) {model_rl}</ansicyan>",
        "en": "\n<ansicyan>ModelScope daily free quota left: global {total} | model({model}) {model_rl}</ansicyan>",
    },
    "chat.interrupted": {
        "zh": "[已中断]",
        "en": "[interrupted]",
    },
    "chat.media_embedded": {
        "zh": "[已嵌入 {count} 个媒体文件]",
        "en": "[embedded {count} media files]",
    },
    "chat.opening": {
        "zh": "正在打开: {url}",
        "en": "Opening: {url}",
    },
    "chat.switching_fallback": {
        "zh": "正在切换到备用模型: {model}",
        "en": "Switching to fallback model: {model}",
    },
    "chat.switched_fallback_retry": {
        "zh": "已切换到备用模型，自动重试中...",
        "en": "Switched to fallback model, retrying...",
    },
    "chat.switch_failed": {
        "zh": "切换模型失败: {error}",
        "en": "Model switch failed: {error}",
    },
    "chat.no_more_fallback": {
        "zh": "没有更多备用模型可用",
        "en": "No more fallback models available",
    },
    "chat.agent_error": {
        "zh": "Agent 执行错误: {error}",
        "en": "Agent error: {error}",
    },
    "chat.agent_stopped": {
        "zh": "该消息意外停止",
        "en": "This message stopped unexpectedly",
    },
    "chat.load_conv_failed": {
        "zh": "加载对话失败: {error}",
        "en": "Failed to load conversation: {error}",
    },
    # ─── /model ───
    "cmd.model": {
        "zh": "模型管理（新建/编辑/切换）",
        "en": "Model management (new/edit/switch)",
    },
    "chat.model.menu": {
        "zh": "模型管理:",
        "en": "Model management:",
    },
    "chat.model.new": {
        "zh": "新建模型 (/model new)",
        "en": "New model (/model new)",
    },
    "chat.model.edit": {
        "zh": "编辑当前模型 (/model edit)",
        "en": "Edit current model (/model edit)",
    },
    "chat.model.switch": {
        "zh": "切换模型 (/model switch)",
        "en": "Switch model (/model switch)",
    },
    # ─── /vision / /skill ───
    "cmd.vision": {
        "zh": "视觉模型配置",
        "en": "Vision model config",
    },
    "cmd.skill": {
        "zh": "技能管理",
        "en": "Skill management",
    },
    "chat.skill.init_first": {
        "zh": "请先初始化工作目录",
        "en": "Please initialize the workdir first",
    },
    # ─── /tools ───
    "cmd.tools": {
        "zh": "显示内置工具",
        "en": "Show built-in tools",
    },
    "chat.tools.title": {
        "zh": "内置工具",
        "en": "Built-in tools",
    },
    "chat.tools.native_vision": {
        "zh": "当前模型支持原生视觉，图片/视频将直接嵌入消息",
        "en": "Current model supports native vision; images/videos will be embedded directly",
    },
    "chat.tools.disabled": {
        "zh": " (已禁用)",
        "en": " (disabled)",
    },
    # ─── /history ───
    "cmd.history": {
        "zh": "历史会话",
        "en": "Session history",
    },
    "chat.history.none": {
        "zh": "没有历史会话",
        "en": "No session history",
    },
    "chat.history.select": {
        "zh": "选择历史会话:",
        "en": "Select a session:",
    },
    "chat.history.operation": {
        "zh": "操作:",
        "en": "Action:",
    },
    "chat.history.load": {
        "zh": "加载此会话",
        "en": "Load this session",
    },
    "chat.history.rename": {
        "zh": "重命名此会话",
        "en": "Rename this session",
    },
    "chat.history.delete": {
        "zh": "删除此会话",
        "en": "Delete this session",
    },
    "chat.history.rename_prompt": {
        "zh": "输入新名称（留空恢复默认）:",
        "en": "Enter a new name (leave empty to restore default):",
    },
    "chat.history.renamed": {
        "zh": "会话已重命名",
        "en": "Session renamed",
    },
    "chat.history.delete_confirm": {
        "zh": "确定删除会话 {tid}？",
        "en": "Delete session {tid}?",
    },
    "chat.history.deleted": {
        "zh": "会话已删除",
        "en": "Session deleted",
    },
    # ─── /compress ───
    "cmd.compress": {
        "zh": "压缩会话",
        "en": "Compress session",
    },
    "chat.compress.no_model": {
        "zh": "请先配置模型",
        "en": "Please configure a model first",
    },
    "chat.compress.confirm": {
        "zh": "确定压缩当前会话？",
        "en": "Compress the current session?",
    },
    "chat.compress.working": {
        "zh": "压缩中...",
        "en": "Compressing...",
    },
    "chat.compress.failed_prefix": {
        "zh": "会话压缩失败",
        "en": "Session compression failed",
    },
    "chat.compress.failed_detail": {
        "zh": "会话压缩失败: {error}",
        "en": "Session compression failed: {error}",
    },
    "chat.compress.failed_no_summary": {
        "zh": "会话压缩失败: LLM 返回结果缺少 summary 字段",
        "en": "Session compression failed: LLM response missing summary field",
    },
    "chat.compress.done_prefix": {
        "zh": "历史对话已压缩: ",
        "en": "Conversation compressed: ",
    },
    "chat.compress.complete": {
        "zh": "会话压缩完成",
        "en": "Session compression complete",
    },
    "chat.compress.error": {
        "zh": "压缩失败: {error}",
        "en": "Compression failed: {error}",
    },
    # ─── /git ───
    "cmd.git": {
        "zh": "Git 状态",
        "en": "Git status",
    },
    "chat.git.unavailable": {
        "zh": "Git 不可用: {status}",
        "en": "Git unavailable: {status}",
    },
    "chat.git.repo_init": {
        "zh": "Git 仓库已初始化 ({count} 个检查点)",
        "en": "Git repository initialized ({count} checkpoints)",
    },
    "chat.git.repo_not_init": {
        "zh": "Git 仓库未初始化",
        "en": "Git repository not initialized",
    },
    "chat.git.rollback_failed": {
        "zh": "Git 回滚失败: {error}",
        "en": "Git rollback failed: {error}",
    },
    # ─── /search ───
    "cmd.search": {
        "zh": "配置 Tavily 搜索 API Key",
        "en": "Configure Tavily search API Key",
    },
    "chat.search.unset": {
        "zh": "未配置",
        "en": "not configured",
    },
    "chat.search.current_key": {
        "zh": "当前 Tavily API Key: {key}",
        "en": "Current Tavily API Key: {key}",
    },
    "chat.search.operation": {
        "zh": "操作:",
        "en": "Action:",
    },
    "chat.search.configure": {
        "zh": "配置 API Key",
        "en": "Configure API Key",
    },
    "chat.search.clear": {
        "zh": "清除 API Key",
        "en": "Clear API Key",
    },
    "chat.search.input_key": {
        "zh": "请输入 Tavily API Key:",
        "en": "Enter Tavily API Key:",
    },
    "chat.search.cleared": {
        "zh": "Tavily API Key 已清除",
        "en": "Tavily API Key cleared",
    },
    "chat.search.saved": {
        "zh": "Tavily API Key 已保存",
        "en": "Tavily API Key saved",
    },
    "chat.search.cancelled": {
        "zh": "未输入，已取消",
        "en": "No input, cancelled",
    },
    # ─── /mode ───
    "cmd.mode": {
        "zh": "切换 Common/Yolo 模式",
        "en": "Toggle Common/Yolo mode",
    },
    "chat.mode.select": {
        "zh": "选择模式:",
        "en": "Select mode:",
    },
    "chat.mode.common": {
        "zh": "Common (手动批准风险操作)",
        "en": "Common (manually approve risky actions)",
    },
    "chat.mode.yolo": {
        "zh": "Yolo (自动批准所有操作)",
        "en": "Yolo (auto-approve all actions)",
    },
    "chat.mode.switched": {
        "zh": "已切换到 {mode} 模式",
        "en": "Switched to {mode} mode",
    },
    # ─── /workdir ───
    "cmd.workdir": {
        "zh": "切换工作目录",
        "en": "Switch workdir",
    },
    "chat.workdir.select": {
        "zh": "选择工作目录:",
        "en": "Select workdir:",
    },
    "chat.workdir.custom": {
        "zh": "自定义路径...",
        "en": "Custom path...",
    },
    "chat.workdir.custom_prompt": {
        "zh": "请输入工作目录路径: ",
        "en": "Enter workdir path: ",
    },
    "chat.workdir.not_exist": {
        "zh": "路径不存在",
        "en": "Path does not exist",
    },
    "chat.workdir.current": {
        "zh": "工作目录: {path}",
        "en": "Workdir: {path}",
    },
    # ─── /messages ───
    "cmd.messages": {
        "zh": "管理历史消息（编辑/分叉/删除）",
        "en": "Manage history messages (edit/fork/delete)",
    },
    "chat.messages.no_agent": {
        "zh": "Agent 未初始化",
        "en": "Agent not initialized",
    },
    "chat.messages.none": {
        "zh": "没有可管理的消息",
        "en": "No messages to manage",
    },
    "chat.messages.select_op": {
        "zh": "选择操作:",
        "en": "Select action:",
    },
    "chat.messages.edit": {
        "zh": "编辑消息",
        "en": "Edit message",
    },
    "chat.messages.fork": {
        "zh": "分叉消息",
        "en": "Fork message",
    },
    "chat.messages.delete": {
        "zh": "删除消息",
        "en": "Delete message",
    },
    "chat.messages.select_delete": {
        "zh": "选择要删除的消息组（空格选择，回车确认）:",
        "en": "Select message groups to delete (space to select, enter to confirm):",
    },
    "chat.messages.delete_confirm": {
        "zh": "确定删除 {count} 个消息组？",
        "en": "Delete {count} message groups?",
    },
    "chat.messages.no_valid": {
        "zh": "没有有效的选择",
        "en": "No valid selection",
    },
    "chat.messages.deleted_groups": {
        "zh": "已删除 {count} 个消息组",
        "en": "Deleted {count} message groups",
    },
    "chat.messages.edit_hint": {
        "zh": "选择要编辑的消息组（编辑后将删除此消息组之后的所有内容）:",
        "en": "Select a message group to edit (everything after it will be deleted):",
    },
    "chat.messages.fork_hint": {
        "zh": "选择 Fork 点（此消息组将保留在分支中）:",
        "en": "Select the fork point (this group will be kept in the branch):",
    },
    "chat.messages.invalid": {
        "zh": "无效的选择",
        "en": "Invalid selection",
    },
    "chat.messages.no_human": {
        "zh": "该组没有 HumanMessage",
        "en": "This group has no HumanMessage",
    },
    "chat.messages.edit_confirm": {
        "zh": "确定编辑此消息组？编辑后将删除此消息组之后的所有内容。",
        "en": "Edit this message group? Everything after it will be deleted.",
    },
    "chat.messages.loaded_to_input": {
        "zh": "消息已加载到输入框，修改后发送即可重新生成",
        "en": "Message loaded into the input box; edit and send to regenerate",
    },
    "chat.messages.fork_confirm": {
        "zh": "确定从第 {idx} 条消息组创建分支？",
        "en": "Create a branch from message group #{idx}?",
    },
    "chat.messages.select_new_workdir": {
        "zh": "选择新工作目录:",
        "en": "Select a new workdir:",
    },
    "chat.messages.copying": {
        "zh": "复制工作目录文件...",
        "en": "Copying workdir files...",
    },
    "chat.messages.copy_failed": {
        "zh": "复制文件失败:\n{tb}",
        "en": "Copy failed:\n{tb}",
    },
    "chat.messages.fork_done": {
        "zh": "分支已创建！工作目录: {path}",
        "en": "Branch created! Workdir: {path}",
    },
    # ─── 复制目录（Windows 保留名等）───
    "chat.copy.skip_reserved": {
        "zh": "跳过 Windows 保留名: {name}",
        "en": "Skipping Windows reserved name: {name}",
    },
    "chat.copy.dir_failed": {
        "zh": "复制目录失败: {name}, {error}",
        "en": "Copy dir failed: {name}, {error}",
    },
    "chat.copy.file_failed": {
        "zh": "复制文件失败: {name}, {error}",
        "en": "Copy file failed: {name}, {error}",
    },
    # ─── HITL ───
    "hitl.action": {
        "zh": "操作:",
        "en": "Action:",
    },
    "hitl.approve": {
        "zh": "approve (批准)",
        "en": "approve",
    },
    "hitl.reject": {
        "zh": "reject (拒绝)",
        "en": "reject",
    },
    "hitl.write_file": {
        "zh": "写入文件: {path}\n内容: {content}",
        "en": "Write file: {path}\nContent: {content}",
    },
    "hitl.edit_modify": {
        "zh": "[HITL] edit  修改文件: {path}",
        "en": "[HITL] edit  modify file: {path}",
    },
    "hitl.user_rejected": {
        "zh": "用户已拒绝",
        "en": "user rejected",
    },
    # ─── /help 表格 ───
    "chat.help.title": {
        "zh": "命令列表",
        "en": "Commands",
    },
    "chat.help.col_cmd": {
        "zh": "命令",
        "en": "Command",
    },
    "chat.help.col_desc": {
        "zh": "说明",
        "en": "Description",
    },
    # ─── 斜杠命令描述（其余）───
    "cmd.new": {
        "zh": "新会话",
        "en": "New session",
    },
    "cmd.langsmith": {
        "zh": "LangSmith 追踪",
        "en": "LangSmith tracing",
    },
    "cmd.homepage": {
        "zh": "打开项目主页",
        "en": "Open project homepage",
    },
    "cmd.help": {
        "zh": "显示帮助",
        "en": "Show help",
    },
    "cmd.quit": {
        "zh": "退出",
        "en": "Quit",
    },
    "cmd.lang": {
        "zh": "选择语言 / Select language",
        "en": "选择语言 / Select language",
    },
    # ─── /langsmith（chat 命令）───
    "chat.langsmith.state_on": {
        "zh": "开启",
        "en": "On",
    },
    "chat.langsmith.state_off": {
        "zh": "关闭",
        "en": "Off",
    },
    "chat.langsmith.tracing_line": {
        "zh": "LangSmith 追踪: {state}",
        "en": "LangSmith tracing: {state}",
    },
    "chat.langsmith.project_line": {
        "zh": "  项目: {project}",
        "en": "  Project: {project}",
    },
    "chat.langsmith.operation": {
        "zh": "操作:",
        "en": "Action:",
    },
    "chat.langsmith.open_panel": {
        "zh": "打开面板",
        "en": "Open dashboard",
    },
    "chat.langsmith.enable": {
        "zh": "开启追踪",
        "en": "Enable tracing",
    },
    "chat.langsmith.disable": {
        "zh": "关闭追踪",
        "en": "Disable tracing",
    },
    "chat.langsmith.rename_project": {
        "zh": "修改项目名称",
        "en": "Rename project",
    },
    "chat.langsmith.change_key": {
        "zh": "修改 API Key",
        "en": "Change API Key",
    },
    "chat.langsmith.set_key_first": {
        "zh": "请先设置 LangSmith API Key",
        "en": "Please set the LangSmith API Key first",
    },
    "chat.langsmith.input_project": {
        "zh": "请输入项目名称:",
        "en": "Enter project name:",
    },
    "chat.langsmith.input_key": {
        "zh": "请输入 LangSmith API Key:",
        "en": "Enter LangSmith API Key:",
    },
    "chat.langsmith.config_updated": {
        "zh": "LangSmith 配置已更新，重启后生效",
        "en": "LangSmith config updated, takes effect after restart",
    },
    # ─── 工具（ask_user / web_search / vision）───
    "tools.tavily_not_configured": {
        "zh": "[ERROR] Tavily API Key 未配置，请使用 /search 命令配置",
        "en": "[ERROR] Tavily API Key not configured. Use the /search command to configure it.",
    },
    "tools.please_input": {
        "zh": "请输入: ",
        "en": "Please enter: ",
    },
    "tools.user_cancelled": {
        "zh": "(用户取消)",
        "en": "(user cancelled)",
    },
    "tools.query_failed": {
        "zh": "(询问失败: {error})",
        "en": "(query failed: {error})",
    },
    "tools.batch_questions": {
        "zh": "📋 批量提问 ({count} 个问题)",
        "en": "📋 Batch questions ({count} questions)",
    },
    "tools.question_progress": {
        "zh": "问题 {i}/{total}: {text}",
        "en": "Question {i}/{total}: {text}",
    },
    "tools.select_hint": {
        "zh": "选择（空格选择，回车确认）:",
        "en": "Select (space to choose, enter to confirm):",
    },
    "tools.batch_results": {
        "zh": "=== 批量提问结果 ===",
        "en": "=== Batch Question Results ===",
    },
    "tools.batch_qa": {
        "zh": "问题: {question}\n回答: {answer}",
        "en": "Question: {question}\nAnswer: {answer}",
    },
    "tools.vision_not_configured": {
        "zh": "vision:\n[FAILED] 视觉模型未配置。\n请使用 /vision 命令配置 ModelScope API Key，\n或设置环境变量 ModelScopeToken。",
        "en": "vision:\n[FAILED] Vision model not configured.\nUse the /vision command to configure the ModelScope API Key,\nor set the ModelScopeToken environment variable.",
    },
    "tools.vision_call_failed": {
        "zh": "视觉模型 {model} 调用失败: {error}",
        "en": "Vision model {model} call failed: {error}",
    },
    "tools.vision_all_failed": {
        "zh": "vision:\n[FAILED] 所有视觉模型均调用失败\n最后错误: {error}",
        "en": "vision:\n[FAILED] All vision models failed\nLast error: {error}",
    },
    # ─── agent_setup 重试/回退 ───
    "agent.vision_native_filter": {
        "zh": "当前模型支持原生视觉，图片/视频已直接嵌入消息，无需调用 vision 工具。请直接分析消息中的图片/视频内容。",
        "en": "The current model supports native vision; image/video is already embedded in the message, so the vision tool is not needed. Please analyze the image/video content in the message directly.",
    },
    "agent.switch_to_fallback": {
        "zh": "主模型重试{count}次失败，切换到备用模型...",
        "en": "Main model failed after {count} retries, switching to fallback model...",
    },
    "agent.no_fallback_giveup": {
        "zh": "请求失败，无备用模型可用，放弃请求\n  {error}",
        "en": "Request failed, no fallback model available, giving up\n  {error}",
    },
    "agent.retry_in": {
        "zh": "请求失败 ({count}/{max}), {delay}秒后重试...\n  {error}",
        "en": "Request failed ({count}/{max}), retrying in {delay}s...\n  {error}",
    },
    "agent.switch_error": {
        "zh": "切换到备用模型",
        "en": "switching to fallback model",
    },
    # ─── runner ───
    "runner.fallback_switched": {
        "zh": "Agent {type} 主模型失败，已切换备用模型，请重试",
        "en": "Agent {type} main model failed, switched to fallback model, please retry",
    },
    # ─── display 欢迎页 / spinner ───
    "display.welcome_title": {
        "zh": "终端 AI 编程助手",
        "en": "Terminal-based AI Coding Agent",
    },
    "display.welcome_hint": {
        "zh": "Enter 发送 | Ctrl+Enter 换行 | /help 查看命令\nCtrl+C 中断生成 | Tab 切换模式 | /quit 退出",
        "en": "Enter to send | Ctrl+Enter for newline | /help for commands\nCtrl+C to interrupt | Tab to toggle mode | /quit to exit",
    },
    "display.organizing": {
        "zh": "正在整理结果...",
        "en": "Organizing results...",
    },
    # ─── git_checker ───
    "git.available": {
        "zh": "Git可用",
        "en": "Git available",
    },
    "git.cmd_failed": {
        "zh": "Git命令执行失败: {error}",
        "en": "Git command failed: {error}",
    },
    "git.unknown_error": {
        "zh": "未知错误",
        "en": "unknown error",
    },
    "git.not_found": {
        "zh": "未找到Git命令，请确保Git已安装并添加到PATH环境变量中",
        "en": "Git command not found. Please ensure Git is installed and added to PATH",
    },
    "git.timeout": {
        "zh": "Git命令执行超时",
        "en": "Git command timed out",
    },
    "git.exception": {
        "zh": "检查Git时发生异常: {error}",
        "en": "Git check exception: {error}",
    },
    # ─── cli ───
    "cli.init_failed": {
        "zh": "初始化失败",
        "en": "Initialization failed",
    },
    "cli.unknown_action": {
        "zh": "未知操作: {action}",
        "en": "Unknown action: {action}",
    },
    "cli.available_actions": {
        "zh": "可用操作: new, edit, switch",
        "en": "Available actions: new, edit, switch",
    },
    "cli.opening": {
        "zh": "正在打开: {url}",
        "en": "Opening: {url}",
    },
    # ─── prompts / 表单 ───
    "prompt.skip": {
        "zh": "跳过 (不设置)",
        "en": "Skip (not set)",
    },
    "prompt.custom_input": {
        "zh": "自定义输入...",
        "en": "Custom input...",
    },
    "prompt.please_input": {
        "zh": "请输入: ",
        "en": "Please enter: ",
    },
    "form.model_name": {
        "zh": "输入模型名称: ",
        "en": "Enter model name: ",
    },
    "form.base_url": {
        "zh": "输入 API Base URL:",
        "en": "Enter API Base URL:",
    },
    "form.api_key_source": {
        "zh": "选择 API Key 来源:",
        "en": "Select API Key source:",
    },
    "form.keep_current_key": {
        "zh": "保持当前 Key ({masked})",
        "en": "Keep current key ({masked})",
    },
    "form.reinput_key": {
        "zh": "重新输入 API Key...",
        "en": "Re-enter API Key...",
    },
    "form.input_key": {
        "zh": "输入 API Key: ",
        "en": "Enter API Key: ",
    },
    "form.api_key_empty": {
        "zh": "API Key 不能为空",
        "en": "API Key cannot be empty",
    },
    "form.configure_hyperparams": {
        "zh": "配置超参数？",
        "en": "Configure hyperparameters?",
    },
    "form.input_temperature": {
        "zh": "输入 temperature: ",
        "en": "Enter temperature: ",
    },
    "form.input_top_p": {
        "zh": "输入 top_p: ",
        "en": "Enter top_p: ",
    },
    "form.input_top_k": {
        "zh": "输入 top_k: ",
        "en": "Enter top_k: ",
    },
    "form.input_max_tokens": {
        "zh": "输入 max_completion_tokens: ",
        "en": "Enter max_completion_tokens: ",
    },
    "form.input_stop": {
        "zh": "输入停止序列 (逗号分隔): ",
        "en": "Enter stop sequences (comma-separated): ",
    },
    "form.input_freq_penalty": {
        "zh": "输入 frequency_penalty: ",
        "en": "Enter frequency_penalty: ",
    },
    "form.input_presence_penalty": {
        "zh": "输入 presence_penalty: ",
        "en": "Enter presence_penalty: ",
    },
    "modelscope.detected_token": {
        "zh": "检测到 ModelScope Token，是否使用？",
        "en": "ModelScope token detected. Use it?",
    },
    "modelscope.manual_input": {
        "zh": "手动输入...",
        "en": "Manual input...",
    },
    "modelscope.input_key": {
        "zh": "输入 ModelScope API Key: ",
        "en": "Enter ModelScope API Key: ",
    },
    "provider.bigmodel": {
        "zh": "智谱 GLM",
        "en": "Zhipu GLM",
    },
    "provider.qwen": {
        "zh": "通义千问",
        "en": "Qwen",
    },
    # ─── 工具描述（/tools 命令显示）───
    "tool_desc.load_skill": {
        "zh": "加载技能的详细指令。当用户请求匹配某个技能描述时使用。",
        "en": "Load a skill's detailed instructions. Use when the user's request matches a skill description.",
    },
    "tool_desc.bash": {
        "zh": "执行 Shell 命令（自动检测平台、跟踪工作目录）。",
        "en": "Execute a shell command with automatic platform detection and CWD tracking.",
    },
    "tool_desc.read_file": {
        "zh": "读取文件内容。",
        "en": "Read the contents of a file.",
    },
    "tool_desc.write_file": {
        "zh": "写入内容到文件。",
        "en": "Write content to a file.",
    },
    "tool_desc.glob": {
        "zh": "按 glob 模式查找文件。",
        "en": "Find files matching a glob pattern.",
    },
    "tool_desc.grep": {
        "zh": "在文件中搜索正则表达式模式。",
        "en": "Search for a pattern in files.",
    },
    "tool_desc.edit": {
        "zh": "通过替换文本编辑文件。",
        "en": "Edit a file by replacing text.",
    },
    "tool_desc.list_dir": {
        "zh": "列出目录内容。",
        "en": "List contents of a directory.",
    },
    "tool_desc.web_search": {
        "zh": "执行网络搜索。",
        "en": "Run a web search.",
    },
    "tool_desc.web_fetch": {
        "zh": "从指定 URL 获取内容并转换为文本。",
        "en": "Fetch content from a specified URL and convert it to text.",
    },
    "tool_desc.ask_user": {
        "zh": "向用户提出一个或多个交互式问题。",
        "en": "Ask the user one or more questions interactively with predefined options.",
    },
    "tool_desc.agent": {
        "zh": "启动子代理自主执行任务。",
        "en": "Launch a sub-agent to perform a task autonomously.",
    },
    "tool_desc.todo_write": {
        "zh": "创建和管理结构化的任务列表。",
        "en": "Create and manage a structured task list for the current coding session.",
    },
    "tool_desc.vision": {
        "zh": "使用视觉模型分析图片或视频。",
        "en": "Analyze an image or video using a vision model.",
    },
}

CATALOGS = {
    "zh": {k: v["zh"] for k, v in MESSAGES.items()},
    "en": {k: v["en"] for k, v in MESSAGES.items()},
}
