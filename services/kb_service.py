"""知识库服务：封装底层 VectorStoreManager / DocumentLoader。

职责：
- 屏蔽底层存储路径约定（kb_ 前缀、空格转义等），对上层只暴露干净的业务语义
- 单一应用模型：知识库名全局唯一，无需任何命名空间前缀
- 后续换向量库（Milvus/Weaviate/Qdrant）只需替换本类实现，router 不变
"""

import os
from typing import List, Optional

from api.namespace import compose, is_valid_name
from api.schemas import KBSummary
from src.vector_store import VectorStoreManager
from src.document_loader import DocumentLoader


class KBService:
    def __init__(self):
        self._vs = VectorStoreManager()
        self._loader = DocumentLoader()

    # ---------- 知识库级 ----------

    def list_kbs(self) -> List[KBSummary]:
        """列出全部知识库。"""
        result = []
        for kb in self._vs.list_knowledge_bases():
            # 从真实路径解析 raw 名，避免底层把 '_' 还原成空格带来的歧义
            raw = os.path.basename(kb.get("path", ""))[3:]  # 去掉 "kb_" 前缀
            result.append(KBSummary(
                name=raw,
                document_count=kb.get("document_count", 0),
                chunk_count=kb.get("chunk_count", 0),
            ))
        return result

    def create_kb(self, name: str, description: Optional[str] = None) -> bool:
        """创建知识库（重名返回 False）。"""
        if not is_valid_name(name):
            raise ValueError("知识库名含非法字符（不能包含 / \\ : * ? \" < > |）")
        return self._vs.create_knowledge_base(compose(name))

    def delete_kb(self, name: str) -> bool:
        if not is_valid_name(name):
            raise ValueError("知识库名含非法字符（不能包含 / \\ : * ? \" < > |）")
        return self._vs.delete_knowledge_base(compose(name))

    def get_vectorstore(self, name: str):
        """获取底层向量库句柄（供 Agent 检索使用）。"""
        if not is_valid_name(name):
            raise ValueError("知识库名含非法字符（不能包含 / \\ : * ? \" < > |）")
        return self._vs.get_knowledge_base(compose(name))

    # ---------- 文档级 ----------

    def add_file(self, name: str, content: bytes, filename: str) -> int:
        """导入上传文件，返回写入的片段数。

        替换式导入（避免重复上传同名文件导致片段 n+n）：先备份同名旧片段，
        再清旧写新；若写入阶段失败则回滚恢复旧片段，保证不丢数据。
        """
        if not is_valid_name(name):
            raise ValueError("知识库名含非法字符（不能包含 / \\ : * ? \" < > |）")
        chunks = self._loader.load_uploaded_file(content, filename)
        if not chunks:
            return 0
        kb = compose(name)
        # 备份同名旧片段，用于失败回滚
        old_ids, old_docs, old_metas = self._vs.get_by_source(kb, filename)
        if old_ids:
            self._vs.remove_document(kb, filename)
        try:
            return self._vs.add_documents(kb, chunks)
        except Exception:
            # 回滚：恢复旧片段，保证不丢数据
            if old_ids:
                try:
                    self._vs.add_raw(kb, old_ids, old_docs, old_metas)
                except Exception:
                    pass
            raise

    def add_text(self, name: str, text: str, source: str = "手动输入") -> int:
        if not is_valid_name(name):
            raise ValueError("知识库名含非法字符（不能包含 / \\ : * ? \" < > |）")
        chunks = self._loader.load_text(text, source=source)
        if not chunks:
            return 0
        return self._vs.add_documents(compose(name), chunks)

    def remove_document(self, name: str, source: str) -> int:
        if not is_valid_name(name):
            raise ValueError("知识库名含非法字符（不能包含 / \\ : * ? \" < > |）")
        return self._vs.remove_document(compose(name), source)

    def list_documents(self, name: str) -> List[str]:
        """列出某知识库内已导入的文档来源（去重）。"""
        if not is_valid_name(name):
            return []
        vs = self._vs.get_knowledge_base(compose(name))
        if not vs:
            return []
        try:
            collection = vs._collection
            results = collection.get(include=["metadatas"])
            sources, seen = [], set()
            for meta in results["metadatas"]:
                if meta and "source" in meta:
                    s = meta["source"]
                    if s not in seen:
                        seen.add(s)
                        sources.append(s)
            return sources
        except Exception:
            return []
