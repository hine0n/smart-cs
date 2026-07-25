# smart-cs · 智能客服知识中台

基于 **LangChain + LangGraph + ChromaDB** 的 RAG 智能客服系统。后端用 FastAPI 提供知识库与流式问答能力，前端拆成「**管理后台**（喂数据）」和「**客户端**（问答）」两个独立应用，二者共享同一后端。

核心特性：

- **RAG 检索增强**：用户提问 → 向量检索 Top-K → Rerank 精排 → LangGraph Agent 带着 `search_knowledge_base` 工具生成答案。
- **流式回答（SSE）**：客户端逐字输出，体验接近真人在线客服。
- **智能体(Agent)模型**：后台把「知识库 + 人设 + 语言」封装成 Agent 并「发布到前台」，客户端**零选择**——启动时自动加载已发布的那个 Agent。
- **多知识库**：基于 ChromaDB 的本地向量库，按知识库隔离，支持增量导入与同名文件替换式更新（带失败回滚）。
- **三模式 Reranker**：默认 `none`（关闭重排序，零额外调用、最快），可选 `llm`（轻量模型打分，零额外依赖）/ `cross-encoder`（本地模型，更精准）。
- **安全与治理底座**：可选 `X-API-Key` 鉴权、CORS 白名单、全链路请求/操作日志，预留认证、限流、审计扩展点。
- **多格式文档**：支持 `PDF / TXT / DOCX / Markdown / CSV / Excel`。
- **问答缓存**：热门问题（精确或语义相似）命中即直接返回，跳过检索与 LLM 调用，显著降低延迟与成本；知识库内容变更时自动失效对应维度缓存。

---

## 架构

```text
                       ┌──────────────────────────────┐
                       │      FastAPI 后端 (api/)       │
   管理后台 ──HTTP──▶   │  知识库 CRUD + SSE 流式问答 +   │  ◀──HTTP── 客户端
  admin_app.py         │  智能体全局注册表 /agents       │          customer_app.py
  (建库/喂数据/管Agent)  │  单一应用（后台前台一对一）     │          (只提问看答案)
                       └───────────────┬──────────────┘
                                       │
                              services/ + src/
                   (KBService / AgentService / AgentRegistry + 向量库/Rerank/Agent)
```

- **管理后台** `admin_app.py`（`src/admin_ui.py`）：面向运营 / 管理员，负责建库、上传文档、维护知识库，并管理**智能体（Agent）**——把知识库 + 人设绑定成一个 Agent，可发布到前台。**不含问答**。
- **客户端** `customer_app.py`（`src/customer_ui.py`）：面向终端客户，只做问答——输入问题 → 流式输出答案，**只呈现答案本身，不泄露 JSON / 评分 / 思考过程**。前台服务的是一个**固定的已发布智能体**，用户无需也无法做任何选择。
- **后端** `api/`：两前端共享的 REST + SSE 服务；`/api/v1/agents` 为全局智能体注册表。系统为**单一应用模型**（后台前台一对一），不区分多租户 / 命名空间。

### 智能体模型（前台零选择）

```text
管理后台建 Agent ──绑定──▶ kb_name + 人设(system_prompt) + 语言(language_mode)
        │ 勾选「发布到前台」(同一时刻最多一个 published)
        ▼
客户端启动 ──GET /api/v1/agents?published=true──▶ 自动加载那个已发布 Agent
        │ 只把 agent_id 传上去问答，界面无任何下拉/选择
```

- 后台可建多个 Agent（不同人设 / 语言 / 知识库），但同一时刻**最多一个**处于「已发布」。
- 客户端永远只服务那一个已发布 Agent；要换前台客服，回后台改发布即可，不动代码。

---

## 技术栈

| 层 | 选型 |
|----|------|
| LLM / Embedding | 阿里云百炼（DashScope），兼容 OpenAI 接口：`qwen` 系列对话模型 + `text-embedding-v3`（默认 1024 维） |
| 编排 | LangChain + LangGraph（`create_react_agent`） |
| 向量库 | ChromaDB（本地持久化，`langchain-chroma`） |
| 后端 | FastAPI + uvicorn（单 worker，避开 Chroma 本地 SQLite 锁） |
| 前端 | Streamlit（两个独立应用） |
| Reranker | 三种模式：`none`（默认关闭重排序）/ `llm`（轻量模型打分）/ Cross-Encoder（`BAAI/bge-reranker-v2-m3`） |

---

## 快速开始

### 1. 准备 Python 虚拟环境并安装依赖

```bash
cd smart-cs
python -m venv venv
venv/Scripts/pip install -r requirements.txt     # Windows
# 或 venv/bin/pip install -r requirements.txt     # Linux/macOS
```

### 2. 配置环境变量

复制模板并填入你的百炼 API Key：

```bash
cp .env.example .env
```

编辑 `.env`，至少填好 `DASHSCOPE_API_KEY`（见下方「配置说明」）。

### 3. 启动服务（二选一）

**方式 A：一键脚本**（推荐，Windows 请在 Git Bash 中运行）

```bash
./start.sh            # 启动 后端 + 管理后台 + 客户端
./start.sh status     # 查看运行状态
./start.sh stop       # 停止全部
./start.sh restart    # 重启
```

**方式 B：手动启动**

```bash
# 后端（必须先启动，单 worker）
venv/Scripts/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1

# 管理后台（另开终端）
venv/Scripts/streamlit run admin_app.py --server.port 8501

# 客户端（另开终端）
venv/Scripts/streamlit run customer_app.py --server.port 8502
```

启动后访问：

- 管理后台：`http://127.0.0.1:8501`
- 客户端：`http://127.0.0.1:8502`
- 后端 API 文档（Swagger）：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

> **注意**：后端必须**单 worker**（`--workers 1`）启动，否则多个 worker 会争抢本地 Chroma 的 SQLite 锁。

---

## 配置说明

所有配置集中在项目根的 `.env`（由 `config.py` 读取），前端通过 `API_BASE_URL` 连接后端。常用变量如下：

| 变量 | 作用 | 默认 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 百炼 API Key（**必填**），从 https://bailian.console.aliyun.com 获取 | 无 |
| `DASHSCOPE_BASE_URL` | 百炼兼容 OpenAI 的接口地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL` | 对话模型（如 `qwen-turbo` / `qwen-plus` / `qwen-max` / `qwen3-xxx`） | `qwen-turbo` |
| `LLM_TEMPERATURE` | 生成温度 | `0.3` |
| `EMBEDDING_MODEL` | 向量化模型 | `text-embedding-v3` |
| `CHUNK_SIZE` | 文本切分块大小 | `500` |
| `CHUNK_OVERLAP` | 切分块重叠 | `50` |
| `RETRIEVAL_K` | 初始向量检索返回数量（Top-K） | `5` |
| `RERANK_MODE` | 重排序模式：`none`（默认，关闭重排序，零额外调用）/ `llm`（轻量模型打分，零依赖）/ `cross-encoder`（更精准，需装 sentence-transformers） | `none` |
| `RERANK_TOP_K` | 送入重排序的文档数 | `5` |
| `RERANK_FINAL_K` | 重排序后保留的文档数 | `3` |
| `RERANK_LLM_MODEL` | `llm` 模式下给文档打分的轻量模型（提速，不影响回答质量） | `qwen-turbo` |
| `RERANK_MODEL` | `cross-encoder` 模式使用的模型 | `BAAI/bge-reranker-v2-m3` |
| `QA_CACHE_ENABLED` | 是否启用问答缓存（命中热门问题直接返回，跳过检索 + LLM） | `true` |
| `QA_CACHE_MAX_ENTRIES` | 缓存条目 LRU 上限，超出淘汰最久未更新 | `1000` |
| `QA_CACHE_TTL` | 缓存有效期（秒），`0` = 永不过期 | `0` |
| `QA_CACHE_SEMANTIC_THRESHOLD` | 语义相似命中阈值（0~1），越高越严格 | `0.92` |
| `API_KEY` | 后端 API Key；**设置后所有接口强制校验 `X-API-Key` 请求头**，留空则不鉴权（开发态） | 空 |
| `CORS_ORIGINS` | 允许跨域的前端域名，逗号分隔；默认 `*` 放开所有（生产环境建议限定） | `*` |
| `API_BASE_URL` | 前端连接的后端地址 | `http://localhost:8000` |

如需启用 `cross-encoder` 重排序，取消 `requirements.txt` 末尾 `sentence-transformers` 行的注释后重新安装。

---

## 使用流程

1. 管理员打开**管理后台**，新建知识库并导入文档（上传文件或粘贴文本）。
2. 在「智能体管理」中新建一个 Agent，绑定上面的知识库名，设好人设（`system_prompt`）与回答语言（`language_mode`），勾选「发布到前台」。
3. 客户打开**客户端**，无需任何选择，直接提问即可获得基于该知识库的流式回答（默认「用什么语言问，用什么语言答」）。
4. 要换前台的客服 / 知识库？回管理后台改「发布」的那个 Agent 即可，客户端代码不动。

---

## 核心工作流（RAG Pipeline）

```text
用户问题
   │
   ▼
[向量检索]  ChromaDB similarity search → Top K（默认 5）
   │
   ▼
[Rerank]    按相关性精排 → 保留 Top K（默认 3）
   │
   ▼
[LangGraph Agent]  带 search_knowledge_base 工具，
                   依据检索结果生成最终答案（流式输出）
```

- **文档加载与切分**（`src/document_loader.py`）：`RecursiveCharacterTextSplitter` 按中英文标点层级切分；Excel 多 sheet 逐行转自然语言文本便于检索。
- **向量库**（`src/vector_store.py`）：基于 ChromaDB 的多知识库管理，按 `kb_<名称>` 落地；删除采用「软删除 + 重试」规避 Windows 文件锁，必要时落入 `vectordb/_deleted/` 暂存。
- **Reranker**（`src/reranker.py`）：`llm` 模式用轻模型对每个片段打 1–10 分；`cross-encoder` 模式本地推理。打分异常时自动回退到检索原始顺序。
- **Agent**（`src/agent.py`）：LangGraph ReAct Agent，流式输出时主动剥离工具调用 / 思考 / JSON 评分残留，保证「只输出答案」。

---

## API 概览

所有业务接口前缀 `/api/v1`，完整定义见 `http://<host>:<port>/docs`（Swagger）。

### 知识库（`knowledge-bases`）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/knowledge-bases` | 列出全部知识库（含文档数 / 片段数） |
| POST | `/api/v1/knowledge-bases` | 创建知识库（同名返回 409） |
| DELETE | `/api/v1/knowledge-bases/{kb_name}` | 删除知识库 |
| POST | `/api/v1/knowledge-bases/{kb_name}/documents` | 批量上传文件并导入（替换式，带回滚） |
| POST | `/api/v1/knowledge-bases/{kb_name}/text` | 直接粘贴文本导入 |
| GET | `/api/v1/knowledge-bases/{kb_name}/documents` | 列出已导入文档来源（去重） |
| DELETE | `/api/v1/knowledge-bases/{kb_name}/documents` | 按来源文件名删除文档 |

### 问答（SSE 流式，`chat`）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat/stream` | 按 `agent_id` 问答（**客户端用**），后端解析知识库 + 人设 + 语言 |
| POST | `/api/v1/chat/stream/kb` | 按 `kb_name` 直接问答（管理 / 调试用，无需智能体） |

SSE 事件类型：`token`（逐字文本增量）、`sources`（检索来源）、`error`（错误）、`done`（结束）。

### 智能体（`agents`）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/agents?published=true` | 列表；`published=true` 仅返回已发布者（客户端加载用） |
| GET | `/api/v1/agents/{agent_id}` | 单个智能体 |
| POST | `/api/v1/agents` | 创建智能体 |
| PUT | `/api/v1/agents/{agent_id}` | 更新智能体（含发布状态） |
| DELETE | `/api/v1/agents/{agent_id}` | 删除智能体 |

### 问答缓存（cache）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/cache/stats` | 查看缓存条目数、容量、TTL、语义阈值与热门问题 Top 列表 |
| POST | `/api/v1/cache/clear` | 清空缓存；`scope` 省略清全部，指定如 `kb:xxx` / `agent:yyy` 仅清该维度 |

> 调用示例（用 `curl` 体验 SSE 流式问答）：
> ```bash
> curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
>   -H "Content-Type: application/json" \
>   -d '{"question":"你们的退货政策是怎样的？","agent_id":"<已发布 agent_id>"}'
> ```

---

## 目录结构

```text
smart-cs/
├── api/                      # FastAPI 后端（REST + SSE）
│   ├── main.py              # 应用入口：CORS、API Key 鉴权依赖、请求日志、/health
│   ├── schemas.py           # Pydantic 请求 / 响应模型
│   ├── deps.py              # 依赖注入：X-API-Key 校验 + service 单例
│   ├── namespace.py         # 知识库名合法性校验与底层存储 key 拼接
│   ├── logging_config.py    # 后端日志配置
│   └── routers/
│       ├── knowledge_bases.py  # 知识库 CRUD + 文档导入 / 删除
│       ├── chat.py             # SSE 流式问答（按 Agent / 按知识库）
│       └── agents.py           # 智能体全局注册表 CRUD
├── services/                # 业务 service 层（封装 src/ 与注册表）
│   ├── kb_service.py        # 知识库服务（替换式导入 + 失败回滚）
│   ├── agent_service.py     # 问答编排（组装 Agent + Rerank）
│   └── agent_registry.py    # 智能体注册表（data/agents.json 持久化）
├── src/                     # 核心 RAG 模块（可被后端复用）
│   ├── agent.py             # CustomerAgent：检索 → Rerank → LangGraph 回答
│   ├── document_loader.py   # 文档加载与切分（PDF/TXT/DOCX/MD/CSV/XLSX）
│   ├── reranker.py          # 重排序（none / llm / cross-encoder 三模式）
│   ├── vector_store.py      # ChromaDB 多知识库管理
│   ├── api_client.py        # 共享 API 客户端 + SSE 解析 + 答案清理（前端用）
│   ├── admin_ui.py          # 管理后台界面
│   └── customer_ui.py       # 客户端问答界面
├── admin_app.py             # 管理后台入口（Streamlit）
├── customer_app.py          # 客户端入口（Streamlit）
├── config.py                # 全局配置（从 .env 读取）
├── requirements.txt
├── .env.example             # 环境变量模板
├── .gitignore
├── start.sh                 # 一键启动 / 停止 / 重启 / 状态
├── data/                    # 智能体注册表 agents.json + 上传暂存
├── vectordb/                # ChromaDB 持久化（按 kb_* 命名）
├── logs/                    # 运行日志（backend / admin / customer）
└── tests/                   # 测试
```

---

## 日志（本地）

所有运行记录写入 `smart-cs/logs/`（自动滚动，单文件 ≤5MB，保留 5 份）：

| 文件 | 内容 |
|------|------|
| `backend.log` | 后端 API 收到的**每一次请求**：方法、路径、状态码、耗时、异常堆栈 |
| `admin.log` | 管理后台（Streamlit）发起的**每一次操作**：建库、上传、建 / 改 / 删 Agent 等 |
| `customer.log` | 客户端（Streamlit）的**每一次提问**：agent_id + 问题文本 |

日志不记录文档正文，仅记录必要的操作与问题文本，便于排查与审计。

---

