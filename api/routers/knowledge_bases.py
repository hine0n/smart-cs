"""知识库管理路由：CRUD + 文档导入/删除。

资源层级：/api/v1/knowledge-bases/...
单一应用模型：知识库名全局唯一，无需任何命名空间前缀。
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Path
from typing import List

from api.schemas import (
    KBCreate, KBSummary, TextImport, DocumentDelete, Message,
)
from api.deps import get_kb_service
from services.kb_service import KBService
from services.qa_cache import qa_cache
from services.agent_registry import AgentRegistry

router = APIRouter(prefix="/api/v1", tags=["knowledge-bases"])

# 知识库内容变更时清缓存，避免返回过期答案
_registry = AgentRegistry()


def _invalidate_kb_cache(kb_name: str):
    """知识库内容变更：清 kb 维度缓存，并清绑定该 kb 的所有 agent 维度缓存。"""
    qa_cache.clear(scope=f"kb:{kb_name}")
    for a in _registry.list_agents():
        if a.get("kb_name") == kb_name:
            qa_cache.clear(scope=f"agent:{a.get('agent_id')}")

# 上传防护：单文件上限 50MB、单次最多 20 个文件，防止全量读入内存导致 OOM（DoS）
_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_FILE_COUNT = 20


@router.get("/knowledge-bases", response_model=List[KBSummary])
def list_knowledge_bases(kb: KBService = Depends(get_kb_service)):
    """列出全部知识库。"""
    try:
        return kb.list_kbs()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/knowledge-bases", response_model=Message, status_code=201)
def create_knowledge_base(
    payload: KBCreate = None,
    kb: KBService = Depends(get_kb_service),
):
    """创建知识库。同名返回 409。"""
    if not payload or not payload.name.strip():
        raise HTTPException(status_code=400, detail="知识库名称不能为空")
    try:
        ok = kb.create_kb(payload.name.strip(), payload.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=409, detail=f"知识库 '{payload.name}' 已存在")
    return Message(detail=f"知识库 '{payload.name}' 创建成功")


@router.delete("/knowledge-bases/{kb_name}", response_model=Message)
def delete_knowledge_base(
    kb_name: str = Path(..., description="知识库名"),
    kb: KBService = Depends(get_kb_service),
):
    try:
        ok = kb.delete_kb(kb_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail=f"知识库 '{kb_name}' 不存在")
    _invalidate_kb_cache(kb_name)
    return Message(detail=f"知识库 '{kb_name}' 已删除")


@router.post("/knowledge-bases/{kb_name}/documents", response_model=Message)
async def upload_documents(
    kb_name: str = Path(..., description="知识库名"),
    files: List[UploadFile] = File(..., description="上传的文件"),
    kb: KBService = Depends(get_kb_service),
):
    """批量上传文件并导入知识库。"""
    from fastapi import HTTPException

    if len(files) > _MAX_FILE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"单次上传文件数过多（最多 {_MAX_FILE_COUNT} 个）",
        )

    total_chunks = 0
    for f in files:
        content = await f.read()
        if len(content) > _MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件 {f.filename or '未知'} 超过单文件大小上限（{_MAX_FILE_BYTES // (1024 * 1024)}MB）",
            )
        try:
            n = kb.add_file(kb_name, content, f.filename or "unknown")
            total_chunks += n
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    _invalidate_kb_cache(kb_name)
    return Message(detail=f"已导入 {len(files)} 个文件，共 {total_chunks} 个片段")


@router.post("/knowledge-bases/{kb_name}/text", response_model=Message)
def import_text(
    kb_name: str = Path(..., description="知识库名"),
    payload: TextImport = None,
    kb: KBService = Depends(get_kb_service),
):
    """直接粘贴文本导入知识库。"""
    if not payload or not payload.text.strip():
        raise HTTPException(status_code=400, detail="文本内容不能为空")
    n = kb.add_text(kb_name, payload.text, source="手动输入")
    _invalidate_kb_cache(kb_name)
    return Message(detail=f"已导入文本，共 {n} 个片段")


@router.get("/knowledge-bases/{kb_name}/documents", response_model=List[str])
def list_documents(
    kb_name: str = Path(..., description="知识库名"),
    kb: KBService = Depends(get_kb_service),
):
    """列出知识库内已导入的文档来源（去重）。"""
    return kb.list_documents(kb_name)


@router.delete("/knowledge-bases/{kb_name}/documents", response_model=Message)
def delete_document(
    kb_name: str = Path(..., description="知识库名"),
    payload: DocumentDelete = None,
    kb: KBService = Depends(get_kb_service),
):
    """按来源文件名删除文档。"""
    if not payload or not payload.source.strip():
        raise HTTPException(status_code=400, detail="缺少 source 字段")
    try:
        n = kb.remove_document(kb_name, payload.source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _invalidate_kb_cache(kb_name)
    return Message(detail=f"已删除 {n} 个片段")
