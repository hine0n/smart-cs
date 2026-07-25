"""共享 API 客户端：供「管理后台」与「客户端」两个 Streamlit 前端复用。

职责：
- 统一封装对 FastAPI 后端的 HTTP 调用（单一应用，无命名空间前缀）。
- 解析 SSE 流（token / sources / error / done 事件）。
- 清理答案里可能泄露的 JSON / 思考 / 内部数据残尾，保证「只输出答案」。

后端地址通过环境变量 API_BASE_URL 覆盖（默认 http://localhost:8000）。
"""

import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from typing import Iterator, List, Tuple
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter

from config import LOG_DIR

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 120  # 秒（LLM 流式首 token 可能需要较长时间，60s 不够用）
# 显式配置连接池：pool_connections=10（主机连接池大小）、pool_maxsize=20（单主机最大并发连接）
# 解决 Streamlit rerun 导致的 HTTPConnectionPool 连接堆积 / Read timed out
_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=0,  # 我们自己处理重试/错误，不让 urllib3 静默重试掩盖问题
)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
API_KEY = os.getenv("API_KEY", "").strip()  # 与后端 .env 的 API_KEY 一致；为空则不携带

_frontend_loggers = {}


def get_frontend_logger(role: str) -> logging.Logger:
    """按角色(admin / customer)写不同日志文件：logs/admin.log、logs/customer.log。"""
    role = role or "frontend"
    if role in _frontend_loggers:
        return _frontend_loggers[role]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"knowledge_mesh.frontend.{role}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = RotatingFileHandler(
            LOG_DIR / f"{role}.log",
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    _frontend_loggers[role] = logger
    return logger


class ApiClient:
    """FastAPI 后端的轻量客户端（单一应用，无 app_id 命名空间）。

    role 用于日志分级：admin=管理后台操作日志，customer=客户端操作日志。
    """

    def __init__(self, base_url: str = API_BASE_URL, role: str = "frontend"):
        self.base_url = base_url.rstrip("/")
        self.role = role

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    def _auth_headers(self) -> dict:
        """若配置了 API_KEY，则在请求头携带 X-API-Key 用于后端鉴权。"""
        if API_KEY:
            return {"X-API-Key": API_KEY}
        return {}

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        logger = get_frontend_logger(self.role)
        headers = self._auth_headers()
        headers.update(kwargs.pop("headers", {}) or {})
        try:
            resp = _session.request(
                method, self._url(path), timeout=REQUEST_TIMEOUT, headers=headers, **kwargs
            )
            logger.info(f"{method} {path} status={resp.status_code}")
            return resp
        except requests.RequestException as e:
            logger.error(f"{method} {path} FAILED {e}")
            raise RuntimeError(f"无法连接后端 ({self.base_url})：{e}")

    # ---------- 知识库 ----------

    def list_kbs(self) -> List[dict]:
        r = self.request("GET", "/knowledge-bases")
        r.raise_for_status()
        return r.json()

    def create_kb(self, name: str) -> requests.Response:
        return self.request("POST", "/knowledge-bases", json={"name": name})

    def delete_kb(self, name: str) -> requests.Response:
        return self.request("DELETE", f"/knowledge-bases/{quote(name)}")

    def upload_files(self, kb: str, files) -> requests.Response:
        return self.request("POST", f"/knowledge-bases/{quote(kb)}/documents", files=files)

    def import_text(self, kb: str, text: str) -> requests.Response:
        return self.request("POST", f"/knowledge-bases/{quote(kb)}/text", json={"text": text})

    def list_documents(self, kb: str) -> List[str]:
        r = self.request("GET", f"/knowledge-bases/{quote(kb)}/documents")
        return r.json() if r.ok else []

    def delete_document(self, kb: str, source: str) -> requests.Response:
        return self.request(
            "DELETE", f"/knowledge-bases/{quote(kb)}/documents", json={"source": source}
        )

    # ---------- 智能体（Agent，全局接口） ----------

    def list_agents(self, published: bool = None) -> List[dict]:
        url = self._url("/agents")
        params = {}
        if published is not None:
            params["published"] = str(published).lower()
        try:
            r = _session.get(url, params=params, timeout=REQUEST_TIMEOUT, headers=self._auth_headers())
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return []

    def get_agent(self, agent_id: str):
        try:
            r = _session.get(self._url(f"/agents/{agent_id}"), timeout=REQUEST_TIMEOUT, headers=self._auth_headers())
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return None

    def create_agent(self, data: dict) -> requests.Response:
        return _session.post(
            self._url("/agents"), json=data, timeout=REQUEST_TIMEOUT,
            headers=self._auth_headers(),
        )

    def update_agent(self, agent_id: str, data: dict) -> requests.Response:
        return _session.put(
            self._url(f"/agents/{agent_id}"), json=data, timeout=REQUEST_TIMEOUT,
            headers=self._auth_headers(),
        )

    def delete_agent(self, agent_id: str) -> requests.Response:
        return _session.delete(
            self._url(f"/agents/{agent_id}"), timeout=REQUEST_TIMEOUT,
            headers=self._auth_headers(),
        )

    # ---------- 问答缓存管理 ----------

    def cache_stats(self) -> dict:
        """查看缓存统计：条目数、容量、TTL、语义阈值、热门问题 Top 列表。"""
        try:
            r = _session.get(
                self._url("/cache/stats"), timeout=REQUEST_TIMEOUT,
                headers=self._auth_headers(),
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return {}

    def clear_cache(self, scope: str = None) -> bool:
        """清空缓存。scope 省略清全部；指定如 'kb:xxx' / 'agent:yyy' 仅清该维度。"""
        params = {}
        if scope:
            params["scope"] = scope
        try:
            r = _session.post(
                self._url("/cache/clear"), params=params,
                timeout=REQUEST_TIMEOUT, headers=self._auth_headers(),
            )
            return r.ok
        except requests.RequestException:
            return False

    # ---------- 问答（SSE 流式） ----------

    def chat_stream_kb(self, kb: str, question: str) -> requests.Response:
        """按知识库名直接问答（管理/调试用），返回原始 streaming Response。"""
        resp = self.request(
            "POST", "/chat/stream/kb",
            json={"question": question, "kb_name": kb},
            stream=True,
        )
        resp.raise_for_status()
        return resp

    def chat_stream_agent(self, agent_id: str, question: str) -> requests.Response:
        """按智能体问答（前台零选择，只传已发布 agent 的 ID）。"""
        logger = get_frontend_logger(self.role)
        # 不记录用户问题明文，避免 PII/隐私落入日志
        logger.info(f"chat agent={agent_id} question_len={len(question)}")
        resp = _session.post(
            self._url("/chat/stream"),
            json={"agent_id": agent_id, "question": question},
            stream=True,
            timeout=REQUEST_TIMEOUT,
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp

    @staticmethod
    def iter_sse(resp: requests.Response) -> Iterator[Tuple[str, str]]:
        """解析 SSE 流，产出 (event, data) 对。"""
        event, data_lines = None, []
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            if raw == "":  # 事件帧结束
                if event is not None:
                    yield event, "\n".join(data_lines)
                event, data_lines = None, []
                continue
            if raw.startswith("event:"):
                event = raw[len("event:"):].strip()
            elif raw.startswith("data:"):
                data_lines.append(raw[len("data:"):].strip())
        if event is not None:
            yield event, "\n".join(data_lines)


# ---------- 答案清理：确保「只输出答案」，不泄露 JSON / 思考 ----------

# 完整 JSON 数组（含 id/score 等 reranker/工具内部字段）
_JSON_BLOCK = re.compile(
    r'\[\s*\{[^\[\]]*?"(?:id|score|source|content)"\s*:[^\[\]]*?\}(?:\s*,\s*\{[^\[\]]*?\})*\s*\]',
    flags=re.DOTALL,
)
# 单个 JSON 对象片段
_JSON_OBJ = re.compile(r'\{\s*"(?:id|score|source|content)"\s*:[^{}]*?\}', flags=re.DOTALL)
# ReAct 思考前缀（部分模型会吐 Thought/Action/Observation）
_THOUGHT = re.compile(
    r'^\s*(?:Thought|Action(?:\s*Input)?|Observation|思考|动作|观察)\s*[:：].*?$',
    flags=re.MULTILINE,
)


def clean_answer(text: str) -> str:
    """对完整答案做彻底清理（用于最终定稿显示 / 存历史）。"""
    if not text:
        return ""
    text = _JSON_BLOCK.sub("", text)
    text = _JSON_OBJ.sub("", text)
    text = _THOUGHT.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def clean_stream_frame(text: str) -> str:
    """流式渲染中途清理：先做完整清理，再剥离尚未闭合的 JSON 残尾，
    避免打字机过程中短暂闪现 [{"id":... 这类未写完的内部数据。"""
    text = clean_answer(text)
    last_open = text.rfind("[")
    if last_open != -1:
        tail = text[last_open:]
        # 仅当尾部是「[ 后紧跟 { 或 "」的 JSON 残尾且未闭合时才剥离，
        # 避免误伤 [文本]、列表序号、[数字] 等正常写法。
        if re.match(r'^\[\s*[\{"]', tail) and tail.count("[") > tail.count("]"):
            text = text[:last_open].rstrip()
    # 同理处理未闭合的 { 对象残尾
    last_brace = text.rfind("{")
    if last_brace != -1:
        tail = text[last_brace:]
        if re.match(r'^\{\s*"', tail) and tail.count("{") > tail.count("}"):
            text = text[:last_brace].rstrip()
    return text
