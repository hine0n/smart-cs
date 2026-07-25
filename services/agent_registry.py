"""Agent 注册表：管理「智能体」与知识库 / 人设的绑定关系。

持久化在 data/agents.json（单文件，进程级内存 + 文件落地，单 worker 安全；
后续要水平扩展可换 SQLite / 数据库，仅替换本模块实现）。

职责：
- 管理后台在此增删改查智能体（含 kb_name、system_prompt、language_mode、published）
- 客户端读取被「发布到前台」的 agent，自动加载，无需任何选择器
- 保证同一时刻最多一个 agent 处于 published 状态
"""

import json
import threading
import uuid
from typing import Dict, List, Optional

from config import PROJECT_ROOT

AGENTS_FILE = PROJECT_ROOT / "data" / "agents.json"
_lock = threading.Lock()


def _load() -> Dict:
    if AGENTS_FILE.exists():
        try:
            return json.loads(AGENTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"agents": {}}
    return {"agents": {}}


def _save(data: Dict) -> None:
    AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    AGENTS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class AgentRegistry:
    """智能体注册表（JSON 文件落地）。"""

    def list_agents(self, published: Optional[bool] = None) -> List[Dict]:
        agents = list(_load().get("agents", {}).values())
        if published is not None:
            agents = [a for a in agents if a.get("published") == published]
        return agents

    def get_agent(self, agent_id: str) -> Optional[Dict]:
        return _load().get("agents", {}).get(agent_id)

    def get_published(self) -> Optional[Dict]:
        published = [a for a in self.list_agents() if a.get("published")]
        return published[0] if published else None

    def create_agent(self, payload: Dict) -> Dict:
        with _lock:
            data = _load()
            agents = data.setdefault("agents", {})
            agent_id = payload.get("agent_id") or _gen_id(payload.get("name", "agent"))
            if agent_id in agents:
                raise ValueError(f"agent_id 已存在: {agent_id}")
            rec = {
                "agent_id": agent_id,
                "name": payload.get("name", agent_id),
                "kb_name": payload.get("kb_name", ""),
                "description": payload.get("description", ""),
                "system_prompt": payload.get("system_prompt", ""),
                "language_mode": payload.get("language_mode", "auto"),
                "published": bool(payload.get("published", False)),
            }
            if rec["published"]:
                self._unpublish_others(agents, agent_id)
            agents[agent_id] = rec
            _save(data)
            return rec

    def update_agent(self, agent_id: str, payload: Dict) -> Optional[Dict]:
        with _lock:
            data = _load()
            agents = data.setdefault("agents", {})
            if agent_id not in agents:
                return None
            rec = agents[agent_id]
            for k in (
                "name",
                "kb_name",
                "description",
                "system_prompt",
                "language_mode",
            ):
                if k in payload and payload[k] is not None:
                    rec[k] = payload[k]
            if "published" in payload and payload["published"] is not None:
                rec["published"] = bool(payload["published"])
                if rec["published"]:
                    self._unpublish_others(agents, agent_id)
            agents[agent_id] = rec
            _save(data)
            return rec

    def delete_agent(self, agent_id: str) -> bool:
        with _lock:
            data = _load()
            agents = data.setdefault("agents", {})
            if agent_id in agents:
                del agents[agent_id]
                _save(data)
                return True
            return False

    @staticmethod
    def _unpublish_others(agents: Dict, keep_id: str) -> None:
        for aid, a in agents.items():
            if aid != keep_id:
                a["published"] = False


def _gen_id(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in (name or "agent"))
    return f"agt_{slug}_{uuid.uuid4().hex[:6]}"
