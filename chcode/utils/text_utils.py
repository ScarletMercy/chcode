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
                else:
                    text_parts.append(f"[{part.get('type', 'image')}]")
            else:
                text_parts.append(str(part))
        return " ".join(text_parts).strip()
    return str(content)
