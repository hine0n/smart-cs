"""问答缓存管理端点：查看命中统计、手动清空。

供运营/调试使用：观察哪些问题已成「热门」被缓存、必要时手动失效。
路由前缀 /api/v1，与现有资源组织一致。
"""
from typing import Optional

from fastapi import APIRouter

from api.schemas import Message
from services.qa_cache import qa_cache

router = APIRouter(prefix="/api/v1", tags=["cache"])


@router.get("/cache/stats")
def cache_stats():
    """查看缓存条目数、容量、TTL 与热门问题 Top 列表。"""
    return qa_cache.stats()


@router.post("/cache/clear", response_model=Message)
def cache_clear(scope: Optional[str] = None):
    """清空缓存。scope 省略清全部；指定如 'kb:xxx' / 'agent:yyy' 仅清该维度。"""
    qa_cache.clear(scope=scope)
    suffix = f"（维度: {scope}）" if scope else "（全部）"
    return Message(detail=f"缓存已清空{suffix}")
