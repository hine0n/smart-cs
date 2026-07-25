"""问答服务：封装 CustomerAgent，提供流式回答。

设计要点（可拓展）：
- 每次请求新建 CustomerAgent 实例，避免共享 _vectorstore 状态，天然支持并发
- 流式输出保持与底层 Generator 一致（token / sources 两类事件）
- 未来换异步 LLM 或 Agent 框架时，只需替换 stream_answer 的内部实现
"""

from typing import Generator, Union, Dict, Any

from services.kb_service import KBService
from src.agent import CustomerAgent


class AgentService:
    def __init__(self, kb_service: KBService):
        self._kb_service = kb_service

    def stream_answer(
        self, kb_name: str, question: str, system_prompt: str = None, top_k: int = None
    ) -> Generator[Union[str, Dict[str, Any]], None, None]:
        """流式回答。

        Yields:
            str  -> 回答文本增量（token）
            dict -> {"sources": [...]} 检索来源（结束前产出一次）
            或由上层约定的错误 dict
        """
        vectorstore = self._kb_service.get_vectorstore(kb_name)
        if vectorstore is None:
            yield {"error": f"知识库 '{kb_name}' 不存在，请先创建并导入文档"}
            return

        # 每次新建 Agent，避免共享状态（单 worker 并发安全）
        agent = CustomerAgent()
        agent.set_knowledge_base(vectorstore)

        for chunk in agent.answer_stream(question, system_prompt=system_prompt, top_k=top_k):
            yield chunk
