"""Agent 模块：检索 Top5 → Rerank Top3 → Agent 回答"""

import re
from typing import List, Generator

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_API_KEY,
    LLM_BASE_URL,
    RETRIEVAL_K,
    AGENT_SYSTEM_PROMPT,
)
from src.reranker import Reranker


class CustomerAgent:
    """智能客服 Agent

    工作流: 用户提问 → 检索 Top5 → Rerank Top3 → Agent 分析回答
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            streaming=True,
        )
        self.reranker = Reranker()
        self._vectorstore = None  # 由外部设置

    def set_knowledge_base(self, vectorstore: Chroma):
        """设置当前使用的知识库"""
        self._vectorstore = vectorstore

    def _build_search_tool(self, top_k: int = None):
        """构建知识库搜索工具"""

        vectorstore = self._vectorstore
        reranker = self.reranker
        k = top_k or RETRIEVAL_K

        @tool
        def search_knowledge_base(query: str) -> str:
            """搜索知识库中的文档内容。

            当需要查找具体信息、文档内容、政策说明、操作指南等时使用此工具。
            参数 query 应为具体的搜索问题或关键词。
            """
            if vectorstore is None:
                return "错误：未设置知识库，请先选择或导入知识库文档"

            # 步骤 1: 检索 Top k（可被 top_k 参数动态覆盖）
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
            docs = retriever.invoke(query)

            if not docs:
                return "未在知识库中找到相关信息"

            # 步骤 2: Rerank → Top 3
            docs = reranker.rerank(query, docs)

            if not docs:
                return "重排序后无相关结果"

            # 步骤 3: 格式化结果（只保留纯文本内容，过滤掉任何 JSON/评分/内部数据）
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知来源")
                content = doc.page_content.strip()[:500]

                # 清理内容中可能残留的 JSON 评分数据
                content = re.sub(r'\[\{.*?\}\]\s*', '', content, flags=re.DOTALL).strip()

                results.append(
                    f"[来源 {i}] 文件: {source}\n{content}"
                )

            return "\n\n---\n\n".join(results)

        return search_knowledge_base

    def answer(self, question: str, system_prompt: str = None, top_k: int = None) -> dict:
        """同步回答用户问题

        Returns:
            dict: {"answer": str, "sources": List[dict], "steps": List[str]}
        """
        tools = [self._build_search_tool(top_k)]
        agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=system_prompt or AGENT_SYSTEM_PROMPT,
        )

        result = agent.invoke({"messages": [HumanMessage(content=question)]})

        # 提取最终回答和中间步骤
        messages = result["messages"]
        answer = messages[-1].content if messages else "抱歉，无法生成回答"

        # 提取搜索工具调用的返回结果作为 sources
        sources = []
        steps = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                steps.append("searching")
            if hasattr(msg, "content") and hasattr(msg, "name") and msg.name == "search_knowledge_base":
                sources = self._extract_sources(msg.content)

        return {"answer": answer, "sources": sources, "steps": steps}

    def answer_stream(self, question: str, system_prompt: str = None, top_k: int = None) -> Generator:
        """流式回答用户问题

        Yields:
            逐步产生回答文本 token（stream_mode="messages" 抽 AIMessageChunk 增量）；
            最后附带一个 {"sources": [...]} 字典。
            回答文本已自动清理：去除 LLM 可能输出的 JSON 数组/评分/内部数据片段。
        """
        tools = [self._build_search_tool(top_k)]
        agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=system_prompt or AGENT_SYSTEM_PROMPT,
        )

        final_sources = []
        text_buffer = []  # 缓存所有文本，用于最终清理

        # 同时开启 messages（逐 token 流式）与 values（完整消息，用于提取来源）。
        # stream_mode 为列表时，每次产出 (mode, data) 元组；一次执行同时拿到
        # 流式文本与检索来源，不额外消耗 API 调用。
        for mode, data in agent.stream(
            {"messages": [HumanMessage(content=question)]},
            stream_mode=["messages", "values"],
        ):
            if mode == "messages":
                chunk, _metadata = data
                # 仅输出最终回答阶段的纯文本增量，跳过工具调用（决策）阶段
                if isinstance(chunk, AIMessageChunk):
                    # 带 tool_call_chunks / tool_calls 的是「调用工具」的中间决策，不输出；
                    # additional_kwargs 里的 reasoning_content（思考）我们从不取，天然不泄露。
                    if getattr(chunk, "tool_call_chunks", []):
                        continue
                    if getattr(chunk, "tool_calls", []):
                        continue
                    if chunk.content:
                        # 清理 LLM 可能夹带的 JSON / 内部数据，避免思考过程泄露
                        cleaned = self._clean_json_leak(chunk.content)
                        if cleaned:
                            text_buffer.append(cleaned)
                            yield cleaned
            elif mode == "values":
                # 从完整 state 消息里找 search 工具返回，提取来源
                for msg in data.get("messages", []):
                    if getattr(msg, "name", None) == "search_knowledge_base" and getattr(msg, "content", ""):
                        final_sources = self._extract_sources(msg.content)

        yield {"sources": final_sources}

    @staticmethod
    def _clean_json_leak(text: str) -> str:
        """清理回答中 LLM 可能泄露的 JSON/内部数据片段"""
        # 移除含 "id"/"score" 的类 JSON 数组片段（reranker 输出泄露）
        text = re.sub(r'\[\s*\{[^]]*?"id"\s*:\s*\d+[^]]*?\}\s*(?:,\s*\{[^]]*?"id"[^]]*?\})*\s*\]', '', text, flags=re.DOTALL)
        text = re.sub(r'\[[^\]]*?"score"\s*:\s*\d+[^\]]*?\]', '', text, flags=re.DOTALL)
        # 清理多余空行
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text

    @staticmethod
    def _clean_stream_frame(text: str) -> str:
        """流式渲染前清理：移除完整 JSON 片段，并剥离尚未闭合的 JSON 残尾，
        避免打字机过程中短暂闪现 reranker / 工具返回的 JSON。"""
        text = CustomerAgent._clean_json_leak(text)
        last_open = text.rfind("[")
        if last_open != -1:
            tail = text[last_open:]
            # 仅当尾部是以 [ 开头的 JSON 数组残尾（[ 后紧跟 { 或 "）且方括号
            # 未闭合时才剥离，避免误伤 [文本] / 列表序号 / [数字] 等正常写法。
            if re.match(r'^\[\s*[\{"]', tail) and tail.count("[") > tail.count("]"):
                text = text[:last_open].rstrip()
        return text

    @staticmethod
    def _extract_sources(text: str) -> List[dict]:
        """从搜索工具返回的文本中提取来源信息"""
        sources = []
        seen = set()

        for line in text.split("\n"):
            if "[来源" in line and "文件:" in line:
                # 提取文件名
                parts = line.split("文件:", 1)
                if len(parts) > 1:
                    src = parts[1].strip()
                    # 去掉相关度评分部分
                    if "(" in src:
                        src = src.split("(")[0].strip()

                    if src and src not in seen:
                        seen.add(src)
                        sources.append({"source": src, "snippet": ""})

        return sources
