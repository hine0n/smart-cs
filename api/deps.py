"""依赖注入：HTTP 边界到业务层的桥接。

集中放置「从请求中提取上下文」与「获取 service 单例」的逻辑，
router 不直接 new 业务对象，便于：
- 未来加认证/鉴权（新增 get_principal）
- 未来换实现（替换 service 工厂）
"""

from fastapi import Header, Depends, HTTPException
from typing import Optional

from config import REQUIRE_AUTH, API_KEY
from services.kb_service import KBService
from services.agent_service import AgentService

# service 单例（进程级）。单 worker 下安全；未来多实例可改为连接池/工厂。
_kb_service = KBService()
_agent_service = AgentService(_kb_service)


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """API Key 鉴权（仅当配置了 API_KEY 时强制校验 X-API-Key）。

    生产环境在 .env 设置 API_KEY 后即开启；未配置则保持开发态兼容。
    """
    if not REQUIRE_AUTH:
        return
    import hmac
    # 使用恒定时间比较，避免时序侧信道泄露 Key 是否匹配
    if not hmac.compare_digest(x_api_key or "", API_KEY or ""):
        raise HTTPException(status_code=401, detail="未授权：缺少或错误的 API Key")


def get_kb_service() -> KBService:
    return _kb_service


def get_agent_service() -> AgentService:
    return _agent_service
