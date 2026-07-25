"""智能体（Agent）路由：增删改查，供管理后台使用。

端点（全局智能体注册表）：
  GET    /api/v1/agents             列表（?published=true 仅返回已发布）
  GET    /api/v1/agents/{agent_id} 单个
  POST   /api/v1/agents            创建
  PUT    /api/v1/agents/{agent_id} 更新
  DELETE /api/v1/agents/{agent_id} 删除

客户端通过 GET /api/v1/agents?published=true 拿到要加载的那个 agent。
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException

from api.schemas import AgentCreate, AgentUpdate, AgentOut, Message
from services.agent_registry import AgentRegistry

router = APIRouter(prefix="/api/v1", tags=["agents"])

_registry = AgentRegistry()


def _to_out(a: dict) -> AgentOut:
    return AgentOut(**a)


@router.get("/agents", response_model=List[AgentOut])
def list_agents(published: Optional[bool] = None):
    return [_to_out(a) for a in _registry.list_agents(published)]


@router.get("/agents/{agent_id}", response_model=AgentOut)
def get_agent(agent_id: str):
    a = _registry.get_agent(agent_id)
    if not a:
        raise HTTPException(status_code=404, detail="智能体不存在")
    return _to_out(a)


@router.post("/agents", response_model=AgentOut, status_code=201)
def create_agent(payload: AgentCreate):
    try:
        return _to_out(_registry.create_agent(payload.model_dump()))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/agents/{agent_id}", response_model=AgentOut)
def update_agent(agent_id: str, payload: AgentUpdate):
    rec = _registry.update_agent(agent_id, payload.model_dump(exclude_unset=True))
    if not rec:
        raise HTTPException(status_code=404, detail="智能体不存在")
    return _to_out(rec)


@router.delete("/agents/{agent_id}", response_model=Message)
def delete_agent(agent_id: str):
    if _registry.delete_agent(agent_id):
        return Message(detail="已删除智能体")
    raise HTTPException(status_code=404, detail="智能体不存在")
