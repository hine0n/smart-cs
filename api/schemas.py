"""API 请求/响应数据模型（Pydantic）。

仅描述 HTTP 边界上的数据结构，业务对象仍由 services/ 层持有，
保持边界清晰、便于后续加版本号 / 校验 / 治理字段。
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ---------- 知识库 ----------

class KBCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="知识库名称（同一 app 内唯一）")
    description: Optional[str] = Field(None, max_length=512, description="可选描述，后续治理层可用")


class KBSummary(BaseModel):
    name: str
    document_count: int = 0
    chunk_count: int = 0
    description: Optional[str] = None


class TextImport(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000, description="直接粘贴的文本正文")


class DocumentDelete(BaseModel):
    source: str = Field(..., description="要删除的来源文件名")


# ---------- 问答 ----------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="用户问题")
    kb_name: str = Field(..., max_length=64, description="目标知识库名")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="检索 Top N（1-20），覆盖默认配置")


class ChatAgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="用户问题")
    agent_id: str = Field(..., description="智能体 ID（后台发布到前台的那个）")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="检索 Top N（1-20），覆盖默认配置")


# ---------- 智能体（Agent） ----------

class AgentCreate(BaseModel):
    name: str = Field(..., description="智能体显示名")
    kb_name: str = Field("", description="该智能体服务的具体知识库名")
    description: Optional[str] = Field("", description="简介")
    system_prompt: Optional[str] = Field("", description="人设/语气提示词，留空用默认")
    language_mode: Literal["auto", "zh", "en", "ja"] = Field("auto", description="auto=与用户同语言 / zh / en / ja")
    published: bool = Field(False, description="是否发布到前台客户端")


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    kb_name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    language_mode: Optional[Literal["auto", "zh", "en", "ja"]] = None
    published: Optional[bool] = None


class AgentOut(BaseModel):
    agent_id: str
    name: str
    kb_name: str
    description: Optional[str] = ""
    system_prompt: Optional[str] = ""
    language_mode: str
    published: bool


class SourceItem(BaseModel):
    source: str
    snippet: str = ""


# ---------- 通用 ----------

class Message(BaseModel):
    detail: str
