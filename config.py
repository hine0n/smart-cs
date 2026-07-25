"""智能客服系统配置"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 加载项目根目录的 .env（双保险：即使不用 `streamlit run` 启动也能读到配置）
load_dotenv(PROJECT_ROOT / ".env")

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

# 向量数据库目录
VECTORDB_DIR = PROJECT_ROOT / "vectordb"

# 日志目录（前后端日志统一落此处）
LOG_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for d in [DATA_DIR, UPLOAD_DIR, VECTORDB_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# LLM 配置（阿里云百炼 / DashScope，兼容 OpenAI 接口）
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 安全与治理（API Key 鉴权 + CORS）
API_KEY = os.getenv("API_KEY", "").strip()
REQUIRE_AUTH = bool(API_KEY)          # 仅配置了 API_KEY 才强制校验 X-API-Key
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")  # 逗号分隔的可信前端域名，默认放开

# Embedding 配置（百炼 text-embedding-v3，默认 1024 维）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# 文本分割配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 检索配置
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))     # 初始检索 Top N

# Reranker 配置
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))   # 送入重排序的文档数
RERANK_FINAL_K = int(os.getenv("RERANK_FINAL_K", "3")) # 重排序后保留的文档数
RERANK_MODE = os.getenv("RERANK_MODE", "none")        # "none" | "llm" | "cross-encoder"（默认 none；cross-encoder 需联网下载 BAAI/bge-reranker-v2-m3，离线环境会卡死，务必先确认可访问 huggingface 再启用）
RERANK_LLM_MODEL = os.getenv("RERANK_LLM_MODEL", "qwen-turbo")  # rerank 打分用的轻模型（提速，不影响回答质量）
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")  # cross-encoder 模型

# 问答缓存（热门问题命中缓存直接返回，跳过检索+LLM）
QA_CACHE_ENABLED = os.getenv("QA_CACHE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
QA_CACHE_MAX_ENTRIES = int(os.getenv("QA_CACHE_MAX_ENTRIES", "1000"))  # LRU 上限，超出淘汰最久未更新
QA_CACHE_TTL = int(os.getenv("QA_CACHE_TTL", "0"))  # 缓存有效期（秒），0=永不过期
QA_CACHE_SEMANTIC_THRESHOLD = float(os.getenv("QA_CACHE_SEMANTIC_THRESHOLD", "0.92"))  # 语义相似命中阈值(0~1)，越高越严格

# 支持的文件类型
SUPPORTED_FILE_TYPES = {
    "pdf": "PDF 文档",
    "txt": "纯文本",
    "docx": "Word 文档",
    "md": "Markdown",
    "csv": "CSV 表格",
    "xlsx": "Excel 表格",
}

# Agent 系统提示词
AGENT_SYSTEM_PROMPT = """你是一个专业的智能客服助手。你拥有一个知识库搜索工具，可以查询已导入的文档内容。

请按以下流程工作：
1. 收到用户问题后，先使用 search_knowledge_base 工具搜索相关知识
2. 根据搜索结果回答用户问题
3. 如果搜索结果不充分，可以尝试换一种方式重新搜索

回答规则：
- 只根据搜索到的上下文内容回答，不要编造信息
- 如果知识库中没有相关信息，诚实告知用户，建议联系人工客服
- 你是在线客服，用口语化、专业的方式回答客户问题即可
- 绝对不要在回答中输出 JSON、分数、ID、评分理由或内部数据"""


# 语言模式：决定「用什么语言问就用什么语言答」
LANGUAGE_INSTRUCTIONS = {
    "auto": "请始终使用与用户提问相同的语言回答（例如用户用英语提问就用英语回答）。",
    "zh": "请始终使用中文回答。",
    "en": "Please always answer in English.",
    "ja": "必ず日本語で回答してください。",
}


def build_agent_prompt(custom_prompt: str = None, language_mode: str = "auto") -> str:
    """拼接最终系统提示词。

    - custom_prompt 为空时回落到全局 AGENT_SYSTEM_PROMPT（默认人设）
    - 末尾追加语言指令（默认 auto：与用户提问同语言，即「正常翻译」）
    """
    base = (custom_prompt or "").strip() or AGENT_SYSTEM_PROMPT
    instr = LANGUAGE_INSTRUCTIONS.get((language_mode or "auto"), LANGUAGE_INSTRUCTIONS["auto"])
    return f"{base}\n\n{instr}"

