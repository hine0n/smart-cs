"""Reranker 模块：将检索到的 Top N 文档重排序为最相关的 Top K"""

import json
import re
from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_API_KEY,
    LLM_BASE_URL,
    RERANK_TOP_K,
    RERANK_FINAL_K,
    RERANK_MODE,
    RERANK_LLM_MODEL,
    RERANK_MODEL,
)


class Reranker:
    """文档重排序器

    支持两种模式：
    - "llm": 用 LLM 对每个文档相关性打分（无需额外依赖）
    - "cross-encoder": 使用 cross-encoder 模型打分（更精准，需下载模型）
    """

    def __init__(self, mode: str = RERANK_MODE):
        self.mode = mode
        self._llm = None
        self._cross_model = None

        if mode not in ("llm", "cross-encoder", "none"):
            raise ValueError(f"不支持的 rerank 模式: {mode}，可选 'none' / 'llm' / 'cross-encoder'")

    def _get_llm(self):
        """延迟初始化 LLM（仅在实际需要打分时创建，使用轻模型提速）"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=RERANK_LLM_MODEL,
                temperature=0,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
            )
        return self._llm

    def _get_cross_encoder(self):
        """延迟加载 cross-encoder 模型"""
        if self._cross_model is None:
            self._cross_model = self._load_cross_encoder()
        return self._cross_model

    def _load_cross_encoder(self):
        """加载 cross-encoder 模型"""
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder(RERANK_MODEL)
        except ImportError:
            raise ImportError(
                "需要安装 sentence-transformers 来使用 cross-encoder 模式\n"
                "运行: pip install sentence-transformers"
            )

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = None,
    ) -> List[Document]:
        """对文档列表按相关性重排序

        Args:
            query: 用户问题
            docs: 待排序的文档列表
            top_k: 返回前 K 个文档（默认使用配置值）

        Returns:
            按相关性降序排列的文档列表
        """
        if top_k is None:
            top_k = RERANK_FINAL_K

        if self.mode == "none":
            # 关闭重排序：直接取前 top_k 个检索结果，避免额外 LLM/模型调用带来的延迟
            return docs[:top_k]

        if not docs:
            return []

        if len(docs) <= top_k:
            return docs

        if self.mode == "cross-encoder":
            return self._rerank_cross_encoder(query, docs, top_k)
        else:
            return self._rerank_llm(query, docs, top_k)

    def _rerank_llm(
        self, query: str, docs: List[Document], top_k: int
    ) -> List[Document]:
        """使用 LLM 对文档相关性打分"""
        if len(docs) <= top_k:
            return docs

        # 构造打分 prompt
        docs_text = []
        for i, doc in enumerate(docs, 1):
            snippet = doc.page_content[:300].replace("\n", " ")
            docs_text.append(f"[{i}] {snippet}")

        prompt = f"""请对以下文档片段与用户问题的相关性打分（1-10 分，10 分最相关）。
只返回 JSON 数组，不要包含其他内容。

用户问题：{query}

文档片段：
{chr(10).join(docs_text)}

请返回打分结果，JSON 格式：
[
  {{"id": 1, "score": 9, "reason": "简要理由"}},
  {{"id": 2, "score": 3, "reason": "简要理由"}}
]"""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            scores = self._parse_scores(response.content)
        except Exception:
            # LLM 打分失败，回退到原始顺序
            return docs[:top_k]

        # 按分数排序
        scored_docs = []
        for s in scores:
            idx = s["id"] - 1
            if 0 <= idx < len(docs):
                doc = docs[idx]
                doc.metadata["rerank_score"] = s["score"]
                scored_docs.append(doc)

        scored_docs.sort(key=lambda d: d.metadata.get("rerank_score", 0), reverse=True)
        return scored_docs[:top_k]

    def _rerank_cross_encoder(
        self, query: str, docs: List[Document], top_k: int
    ) -> List[Document]:
        """使用 cross-encoder 模型重排序"""
        model = self._get_cross_encoder()
        if not model:
            return docs[:top_k]

        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)

        # 组合文档和分数
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in scored[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            result.append(doc)

        return result

    def _parse_scores(self, text: str) -> List[dict]:
        """解析 LLM 返回的打分结果"""
        # 尝试提取 JSON
        text = text.strip()
        if "```" in text:
            text = re.sub(r"```\w*\n?", "", text)
            text = text.replace("```", "")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取数组部分
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return []
