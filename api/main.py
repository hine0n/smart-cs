"""FastAPI 应用入口：知识中台后端服务。

启动（单 worker，避开 Chroma 本地 SQLite 并发锁）：
    uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1

说明：
- 单一应用模型：所有资源以 /api/v1/... 组织，后台前台一对一
- 智能体（Agent）：/api/v1/agents 全局注册表，前台自动加载已发布者
- CORS 放开以便多个前端/业务系统调用（生产环境应限定来源）
- 后续治理层（认证/限流/审计）在 deps.py 与 middleware 处扩展，不影响业务路由
- 日志：每次请求写入 logs/backend.log（见 logging_config）
"""

import sys
import time

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

from api.routers import knowledge_bases, chat, agents, cache
from api.deps import verify_api_key
from api.logging_config import get_backend_logger
from config import CORS_ORIGINS, REQUIRE_AUTH

app = FastAPI(
    title="智能客服知识中台 API",
    version="0.1.0",
    description="可拓展的知识库 / RAG 服务：单一应用、流式问答、智能体管理，预留治理层扩展点",
    dependencies=[Depends(verify_api_key)],
)

_origins = (
    [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
    if CORS_ORIGINS and CORS_ORIGINS != "*"
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(knowledge_bases.router)
app.include_router(chat.router)
app.include_router(agents.router)
app.include_router(cache.router)


# ---- 安全启动自检（不阻断启动，仅打印告警到 stderr） ----
if not REQUIRE_AUTH:
    print(
        "\n\033[93m[SECURITY WARNING] API_KEY 未配置：所有接口当前「无需鉴权」。"
        "任何能访问本服务的人都能读取/删除知识库、上传数据、调用 LLM。"
        "生产部署务必在 .env 设置 API_KEY，并在前端前置认证。\033[0m\n",
        file=sys.stderr,
    )
if CORS_ORIGINS == "*":
    print(
        "\033[93m[SECURITY NOTICE] CORS 允许所有来源 (CORS_ORIGINS=*)，"
        "生产环境请限定可信域名。\033[0m",
        file=sys.stderr,
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每次请求：方法、路径、状态码、耗时。"""
    logger = get_backend_logger()
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(
            f"{request.method} {request.url.path} ERROR {e}",
            exc_info=True,
        )
        raise
    duration = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} {duration:.3f}s"
    )
    return response


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "service": "knowledge-mesh"}
