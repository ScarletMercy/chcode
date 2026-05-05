def get_text_content(content: str | list) -> str:
    """将消息内容转为纯文本，处理多模态消息的列表格式"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts).strip()
    return str(content)


def mask_api_key(key: str, mask: str = "...", short_mask: str = "***") -> str:
    if not key:
        return "未配置"
    if len(key) <= 10:
        return short_mask
    return f"{key[:6]}{mask}{key[-4:]}"
