"""命名空间层（单一应用）。

去除多租户 app_id 后，系统只服务一个应用：知识库名全局唯一即为其存储标识。
本模块仅保留「名字合法性校验」与「拼接底层存储 key」两个职责，
未来若再需物理隔离（独立向量库实例 / 独立 embedding / 独立权限域），
只需替换 compose / 校验实现，上层 router 与 service 无需改动。
"""

import re

# 名字非法字符（路径分隔符与文件系统保留字符），避免底层存储 key 产生歧义
_INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def is_valid_name(name: str) -> bool:
    """校验知识库名是否合法：非空、不含路径/文件系统保留字符、不含目录逃逸段。"""
    if not name or not name.strip():
        return False
    if _INVALID_NAME_CHARS.search(name):
        return False
    norm = name.strip()
    # 拒绝 "." / ".." 以及任何含 ".." 的路径段，防止拼接存储路径时目录逃逸
    if norm in (".", ".."):
        return False
    for seg in re.split(r"[/\\]", norm):
        if seg in (".", ".."):
            return False
    return True


def compose(kb_name: str) -> str:
    """把知识库名组合成底层存储的唯一 key（当前即知识库名本身）。"""
    return (kb_name or "").strip()
