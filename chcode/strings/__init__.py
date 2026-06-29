"""
文案目录 — zh / en 两套平铺字典，key 一一对应。

约定：
- key 用稳定英文点分命名，如 "cmd.new_session"、"model.manage"
- 两个文件的 key 集合保持一致；缺失时 t() 回退到中文再回退到 key
"""

from chcode.strings.en import MESSAGES as _EN
from chcode.strings.zh import MESSAGES as _ZH

CATALOGS = {"zh": _ZH, "en": _EN}
