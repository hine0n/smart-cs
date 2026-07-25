"""问答路由：SSE 流式回答。

端点：
  POST /api/v1/chat/stream       按智能体（agent_id）问答（前台使用）
  POST /api/v1/chat/stream/kb    按知识库名直接问答（管理/调试用）
返回 text/event-stream，事件类型：
  event: token   逐字文本增量
  event: sources 检索来源（JSON，结束前一次）
  event: error   错误信息
  event: done    流结束
前端用 fetch + ReadableStream 或 EventSource 消费。
"""

import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatAgentRequest
from api.deps import get_agent_service
from services.agent_service import AgentService
from services.agent_registry import AgentRegistry
from services.qa_cache import qa_cache
from config import build_agent_prompt

router = APIRouter(prefix="/api/v1", tags=["chat"])

_registry = AgentRegistry()


def _sse(event: str, data: str) -> str:
    # 多 data 行承载换行，避免 token 中的 \n 破坏 SSE 帧结构
    data = (data or "").replace("\r", "")
    body = "".join(f"data: {ln}\n" for ln in data.split("\n"))
    return f"event: {event}\n{body}\n"


def _chunk_text(text: str, size: int = 4):
    """缓存命中时把答案切成短段，模拟逐字流式输出（保留前端打字机体验）。"""
    for i in range(0, len(text), size):
        yield text[i:i + size]


def _try_cache(question: str, scope: str):
    """两级缓存命中：精确匹配(零成本) → 语义相似匹配(算一次 embedding)。

    返回 (answer, sources, hit_type, emb) 或 None：
      - 命中：hit_type 为 "exact"/"semantic"，emb 为本次算出的向量（语义命中时有值）
      - 未命中：hit_type=None，emb 已算好（供生成后写回，避免重复 embedding）
      - embedding 调用异常：返回 None（降级为无缓存路径）
    """
    cached = qa_cache.get(question, scope)  # 精确
    if cached is not None:
        return (*cached, None)
    try:
        emb = qa_cache.embed(question)
    except Exception:
        return None
    cached = qa_cache.get(question, scope, emb=emb)  # 语义
    if cached is not None:
        return (*cached, emb)
    return (None, None, None, emb)


def _cached_response(answer: str, sources: list):
    """缓存命中(精确/语义)的统一伪流式返回。"""

    def event_gen():
        try:
            if sources:
                yield _sse("sources", json.dumps(sources, ensure_ascii=False))
            for seg in _chunk_text(answer):
                yield _sse("token", seg)
        except Exception as e:  # 兜底，避免流中断导致前端挂起
            yield _sse("error", f"缓存读取异常: {e}")
        finally:
            yield _sse("done", "")

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/stream")
def chat_stream_agent(
    payload: ChatAgentRequest,
    agent: AgentService = Depends(get_agent_service),
):
    """按智能体（agent_id）问答：后端解析出知识库 + 人设 + 语言，再跑 RAG。

    客户端（前台）零选择，只把「已发布智能体」的 agent_id 传上来即可。
    """
    if not payload.question.strip():
        return StreamingResponse(
            iter([_sse("error", "问题不能为空")]), media_type="text/event-stream"
        )

    rec = _registry.get_agent(payload.agent_id)
    if not rec:
        return StreamingResponse(
            iter([_sse("error", f"智能体 '{payload.agent_id}' 不存在，请确认后台已发布")]),
            media_type="text/event-stream",
        )

    kb_name = rec.get("kb_name", "")
    if not kb_name:
        return StreamingResponse(
            iter([_sse("error", "该智能体未绑定知识库，请先在后台设置 kb_name")]),
            media_type="text/event-stream",
        )

    # 组合人设 + 语言指令（默认 auto：与用户提问同语言）
    system_prompt = build_agent_prompt(
        rec.get("system_prompt"), rec.get("language_mode", "auto")
    )

    # —— 问答缓存：精确 + 语义相似两级命中，热门/换问法都直接返回 ——
    scope = f"agent:{payload.agent_id}"
    hit = None
    emb = None
    res = _try_cache(payload.question, scope)
    if res is not None:
        _ans, _src, _htype, emb = res
        if _htype is not None:
            hit = (_ans, _src)
    if hit is not None:
        _answer, _sources = hit
        return _cached_response(_answer, _sources)

    def event_gen():
        try:
            full = []
            srcs = []
            for chunk in agent.stream_answer(
                kb_name, payload.question, system_prompt=system_prompt, top_k=payload.top_k
            ):
                if isinstance(chunk, dict):
                    if "error" in chunk:
                        yield _sse("error", chunk["error"])
                        return
                    srcs = chunk.get("sources", [])
                    yield _sse("sources", json.dumps(srcs, ensure_ascii=False))
                else:
                    full.append(chunk)
                    yield _sse("token", chunk)
            # 正常结束才写缓存（异常时不缓存不完整答案）
            qa_cache.put(payload.question, scope, "".join(full), srcs, emb=emb)
        except Exception as e:  # 兜底，避免流中断导致前端挂起
            yield _sse("error", f"服务异常: {e}")
        finally:
            yield _sse("done", "")

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/stream/kb")
def chat_stream_kb(
    payload: ChatRequest,
    agent: AgentService = Depends(get_agent_service),
):
    """按知识库名直接问答（管理/调试用，无需智能体）。"""
    if not payload or not payload.question.strip():
        return StreamingResponse(
            iter([_sse("error", "问题不能为空")]), media_type="text/event-stream"
        )

    # —— 问答缓存：精确 + 语义相似两级命中，热门/换问法都直接返回 ——
    scope = f"kb:{payload.kb_name}"
    hit = None
    emb = None
    res = _try_cache(payload.question, scope)
    if res is not None:
        _ans, _src, _htype, emb = res
        if _htype is not None:
            hit = (_ans, _src)
    if hit is not None:
        _answer, _sources = hit
        return _cached_response(_answer, _sources)

    def event_gen():
        try:
            full = []
            srcs = []
            for chunk in agent.stream_answer(payload.kb_name, payload.question, top_k=payload.top_k):
                if isinstance(chunk, dict):
                    if "error" in chunk:
                        yield _sse("error", chunk["error"])
                        return
                    srcs = chunk.get("sources", [])
                    yield _sse("sources", json.dumps(srcs, ensure_ascii=False))
                else:
                    full.append(chunk)
                    yield _sse("token", chunk)
            # 正常结束才写缓存（异常时不缓存不完整答案）
            qa_cache.put(payload.question, scope, "".join(full), srcs, emb=emb)
        except Exception as e:  # 兜底，避免流中断导致前端挂起
            yield _sse("error", f"服务异常: {e}")
        finally:
            yield _sse("done", "")

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
