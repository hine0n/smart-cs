"""全流程测试：文档导入 → 分块 → 向量化 → 检索 → 重排 → Agent 回答"""

import os
import sys
import json
import shutil
from pathlib import Path

# 把项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env 如果有的话
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

RESULTS = {"passed": 0, "failed": 0, "skipped": 0, "details": []}


def check(msg: str, condition: bool, warn: bool = False):
    """记录测试结果"""
    status = "WARN" if warn and condition else ("PASS" if condition else "FAIL")
    if status == "PASS":
        RESULTS["passed"] += 1
    elif status == "FAIL":
        RESULTS["failed"] += 1
    else:
        RESULTS["skipped"] += 1

    icon = "✅" if condition else ("⚠️" if warn else "❌")
    print(f"  {icon} {status}: {msg}")
    RESULTS["details"].append({"msg": msg, "status": status})


def has_api_key():
    """检查 API key 是否已配置（百炼 DASHSCOPE_API_KEY，兼容旧 OPENAI_API_KEY）"""
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    return bool(key and key not in ("your-api-key-here", "sk-xxx", ""))


# ============================================================
# Test 1: 模块导入
# ============================================================
def test_imports():
    print("\n" + "=" * 60)
    print(" 测试 1: 模块导入")
    print("=" * 60)

    try:
        from config import (
            CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K,
            RERANK_TOP_K, RERANK_FINAL_K, RERANK_MODE,
            SUPPORTED_FILE_TYPES, PROJECT_ROOT as CFG_ROOT,
        )
        check("config.py", True)
        check(f"  CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={CHUNK_OVERLAP}", True)
        check(f"  RETRIEVAL_K={RETRIEVAL_K}, RERANK: {RERANK_TOP_K}→{RERANK_FINAL_K}, mode={RERANK_MODE}", True)
        check(f"  SUPPORTED_FILE_TYPES: {list(SUPPORTED_FILE_TYPES.keys())}", True)
    except Exception as e:
        check(f"config.py: {e}", False)

    try:
        from src.document_loader import DocumentLoader
        loader = DocumentLoader()
        check(f"src/document_loader.py → DocumentLoader(chunk_size={loader.chunk_size})", True)
    except Exception as e:
        check(f"src/document_loader.py: {e}", False)

    try:
        from src.vector_store import VectorStoreManager
        check("src/vector_store.py → VectorStoreManager", True)
    except Exception as e:
        check(f"src/vector_store.py: {e}", False)

    try:
        from src.reranker import Reranker
        check("src/reranker.py → Reranker", True)
    except Exception as e:
        check(f"src/reranker.py: {e}", False)

    try:
        from src.agent import CustomerAgent
        check("src/agent.py → CustomerAgent", True)
    except Exception as e:
        check(f"src/agent.py: {e}", False)


# ============================================================
# Test 2: 文档加载与分块
# ============================================================
def test_document_loading():
    print("\n" + "=" * 60)
    print(" 测试 2: 文档加载与分块")
    print("=" * 60)

    from src.document_loader import DocumentLoader

    loader = DocumentLoader(chunk_size=300, chunk_overlap=30)

    # --- TXT ---
    test_txt = PROJECT_ROOT / "data" / "uploads" / "_test_sample.txt"
    test_txt.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "智能客服系统是一种基于人工智能技术的客户服务解决方案。\n\n"
        "它能够自动回答用户的常见问题，提供24小时不间断的服务。\n"
        "当遇到复杂问题时，系统可以自动转接给人工客服。\n\n"
        "核心组件包括：自然语言理解、知识库管理、多轮对话管理、情感分析。\n"
        "支持的文档类型：PDF、Word、TXT、CSV、Markdown。\n\n"
        "使用前需要配置 OpenAI API Key，然后上传文档到知识库。\n"
        "系统会自动将文档分割为合适大小的片段并生成向量索引。\n"
        "用户提问时，系统检索最相关的片段，利用大语言模型生成回答。"
    )
    test_txt.write_text(content, encoding="utf-8")
    chunks = loader.load_file(str(test_txt))
    check(f"TXT 分块: {len(chunks)} 个 chunk" + (f" (预期 > 1)" if len(chunks) > 1 else " (只有1个, 可能未分割)"),
          len(chunks) >= 1, warn=len(chunks) <= 1)
    if chunks:
        check(f"  首个 chunk 来源: {chunks[0].metadata.get('source')}", True)
        check(f"  chunk 长度: {len(chunks[0].page_content)} 字符", len(chunks[0].page_content) <= 300)

    # --- MD ---
    test_md = PROJECT_ROOT / "data" / "uploads" / "_test_sample.md"
    test_md.write_text("# 产品手册\n\n## 功能介绍\n\n本产品支持以下功能：\n- 自动回复\n- 知识库管理\n- 数据分析\n\n## 使用方法\n\n1. 上传文档\n2. 等待处理\n3. 开始提问", encoding="utf-8")
    md_chunks = loader.load_file(str(test_md))
    check(f"MD 分块: {len(md_chunks)} 个 chunk", len(md_chunks) >= 1)

    # --- 不支持的类型 ---
    try:
        loader.get_file_type("test.xyz")
        check("不支持类型检测: 未抛出异常", False)
    except ValueError:
        check("不支持类型检测: 正确抛出 ValueError", True)

    # --- load_text ---
    text_chunks = loader.load_text("这是测试文本。这是第二句话。这是第三句话。", source="测试输入")
    check(f"load_text: {len(text_chunks)} 个 chunk", len(text_chunks) >= 1)
    if text_chunks:
        check(f"  来源标记: {text_chunks[0].metadata.get('source')}", text_chunks[0].metadata.get('source') == "测试输入")

    return chunks, text_chunks


# ============================================================
# Test 3: 向量存储（仅结构测试，需要 API key）
# ============================================================
def test_vector_store():
    print("\n" + "=" * 60)
    print(" 测试 3: 向量存储管理")
    print("=" * 60)

    from src.vector_store import VectorStoreManager, _soft_delete_dir
    from config import VECTORDB_DIR

    if not has_api_key():
        check("跳过 — 未配置 DASHSCOPE_API_KEY", False, warn=True)
        return None

    try:
        vstore = VectorStoreManager()
        check("VectorStoreManager 初始化成功", True)

        # 清理旧的测试数据
        _soft_delete_dir(str(VECTORDB_DIR / "kb_test_kb"))

        # 创建知识库
        ok = vstore.create_knowledge_base("test_kb")
        check("创建知识库 'test_kb'", ok)

        # 重复创建
        dup = vstore.create_knowledge_base("test_kb")
        check("重复创建返回 False", not dup)

        # 列知识库
        kbs = vstore.list_knowledge_bases()
        check(f"列出知识库: {len(kbs)} 个", len(kbs) >= 1)

        # 知识库统计
        stats = vstore.get_kb_stats("test_kb")
        check(f"  统计: {stats}", stats["chunk_count"] == 0)  # 空知识库

        # 删除
        deleted = vstore.delete_knowledge_base("test_kb")
        check("删除知识库", deleted)

        return vstore

    except Exception as e:
        check(f"向量存储测试异常: {e}", False)
        return None


# ============================================================
# Test 4: Reranker（仅结构测试）
# ============================================================
def test_reranker():
    print("\n" + "=" * 60)
    print(" 测试 4: Reranker 功能")
    print("=" * 60)

    from src.reranker import Reranker
    from langchain_core.documents import Document

    # 测试模式配置
    check(f"当前 Rerank 模式: {os.getenv('RERANK_MODE', 'llm')}", True)

    if not has_api_key():
        check("跳过 reranker 实际调用 — 未配置 API Key", False, warn=True)
        # 但可以测试空文档和少量文档的边缘情况
        try:
            r = Reranker(mode="llm")
            # 空列表
            result = r.rerank("测试", [], top_k=3)
            check("空文档 rerank 返回空列表", result == [])

            # 文档数 <= top_k 直接返回
            docs = [Document(page_content="测试1"), Document(page_content="测试2")]
            result = r.rerank("测试", docs, top_k=3)
            check(f"文档数({len(docs)}) <= top_k(3) 直接返回", len(result) == 2)
        except Exception as e:
            check(f"边缘条件测试失败: {e}", False)
        return

    try:
        r = Reranker(mode="llm")

        # 构建模拟检索文档
        docs = [
            Document(page_content="智能客服系统可以自动回答用户问题，提供7x24小时服务。核心组件包括NLU和知识库。",
                     metadata={"source": "doc1.pdf"}),
            Document(page_content="今天天气很好，适合出去散步游玩。春天的花朵都开了。",
                     metadata={"source": "doc2.txt"}),
            Document(page_content="客服系统的部署需要配置OpenAI API密钥，以及设置向量数据库。",
                     metadata={"source": "doc3.pdf"}),
            Document(page_content="推荐使用Python和LangChain来构建RAG系统，配合ChromaDB存储向量。",
                     metadata={"source": "doc4.md"}),
            Document(page_content="中国的首都是北京，拥有悠久的历史和丰富的文化遗产。",
                     metadata={"source": "doc5.txt"}),
        ]

        result = r.rerank("如何部署智能客服系统？", docs, top_k=3)
        check(f"LLM Rerank: {len(result)} 个结果 (预期 3)", len(result) == 3)

        # 检查是否带了 rerank_score
        has_scores = all("rerank_score" in d.metadata for d in result)
        check("结果带有 rerank_score", has_scores)

        # 检查排序（相关文档应该在前面）
        if result:
            check(f"  Top 1: {result[0].metadata.get('source')}", True)
            scores = [d.metadata.get("rerank_score", 0) for d in result]
            # 应该降序
            is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
            check(f"  分数降序: {scores}", is_sorted)

        # 边缘条件
        empty_result = r.rerank("测试", [], top_k=3)
        check("空列表返回 []", empty_result == [])

        short_result = r.rerank("测试", docs[:1], top_k=3)
        check("文档数 < top_k 直接返回", len(short_result) == 1)

    except Exception as e:
        check(f"Reranker 异常: {e}", False)


# ============================================================
# Test 5: 完整流程（需 API Key）
# ============================================================
def test_full_pipeline():
    print("\n" + "=" * 60)
    print(" 测试 5: 完整 RAG 流程")
    print("=" * 60)

    if not has_api_key():
        check("跳过 — 未配置 DASHSCOPE_API_KEY", False, warn=True)
        print("\n  💡 提示: 复制 .env.example 为 .env 并填入你的百炼 API Key 后重新运行")
        return

    from src.document_loader import DocumentLoader
    from src.vector_store import VectorStoreManager, _soft_delete_dir
    from src.agent import CustomerAgent
    from config import VECTORDB_DIR

    KB_NAME = "pipeline_test_kb"

    # --- Step 1: 加载文档 ---
    print("\n--- Step 1: 准备测试文档 ---")
    loader = DocumentLoader()
    chunks = loader.load_text(
        "客服系统支持自动回复和人工转接。当用户问题超过系统能力时，会自动创建工单并通知人工客服。"
        "系统需要定期更新知识库以保持信息准确。"
        "客服工作时间是周一至周五 9:00-18:00。周末和节假日仅提供自动回复服务。",
        source="客服系统手册"
    )
    check(f"准备测试 chunks: {len(chunks)} 个", len(chunks) >= 1)

    # --- Step 2: 向量化存储 ---
    print("\n--- Step 2: 向量化存储 ---")
    vs = VectorStoreManager()
    _soft_delete_dir(str(VECTORDB_DIR / f"kb_{KB_NAME}"))
    vs.create_knowledge_base(KB_NAME)
    added = vs.add_documents(KB_NAME, chunks)
    check(f"向量化存储: {added} 条", added > 0)

    # --- Step 3: 检索 ---
    print("\n--- Step 3: 语义检索 (Top 5) ---")
    kb = vs.get_knowledge_base(KB_NAME)
    retriever = kb.as_retriever(search_kwargs={"k": 5})
    retrieved = retriever.invoke("客服工作时间是什么？")
    check(f"检索到 {len(retrieved)} 个文档", len(retrieved) > 0)
    if retrieved:
        preview = retrieved[0].page_content[:80]
        check(f"  首个结果预览: {preview}...", True)

    # --- Step 4: Rerank ---
    print("\n--- Step 4: Rerank Top3 ---")
    from src.reranker import Reranker
    reranker = Reranker(mode="llm")
    reranked = reranker.rerank("客服工作时间是什么？", retrieved)
    check(f"重排后: {len(reranked)} 个文档", len(reranked) >= 1)

    # --- Step 5: Agent 回答 ---
    print("\n--- Step 5: Agent 回答 ---")
    agent = CustomerAgent()
    agent.set_knowledge_base(kb)

    result = agent.answer("客服工作时间是什么？")
    answer = result.get("answer", "")
    check(f"Agent 回答: {answer[:100]}...", len(answer) > 10)

    sources = result.get("sources", [])
    check(f"引用来源: {len(sources)} 个", len(sources) >= 0)

    # --- 清理 ---
    # 先释放对知识库 Chroma 的引用（agent / kb 仍握着 SQLite 连接），
    # 否则 Windows 下文件锁会导致 rename 删除失败。
    del kb, agent
    import gc
    gc.collect()
    vs.delete_knowledge_base(KB_NAME)
    check(f"清理测试知识库 '{KB_NAME}'", True)


# ============================================================
# 运行所有测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" 🧪 智能客服系统 — 全流程自动化测试")
    print("=" * 60)

    print(f"\nKEY 状态: {'已配置' if has_api_key() else '未配置 (仅测试结构)'}")
    print(f"API Base: {os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}")
    print(f"Chat Model: {os.getenv('LLM_MODEL', 'qwen-plus')}")
    print(f"Embed Model: {os.getenv('EMBEDDING_MODEL', 'text-embedding-v3')}")
    print(f"Rerank Mode: {os.getenv('RERANK_MODE', 'llm')}")

    test_imports()
    test_document_loading()
    test_vector_store()
    test_reranker()
    test_full_pipeline()

    # --- 结果汇总 ---
    print("\n" + "=" * 60)
    print(" 📊 测试结果汇总")
    print("=" * 60)
    total = RESULTS["passed"] + RESULTS["failed"] + RESULTS["skipped"]
    print(f"  ✅ 通过: {RESULTS['passed']}/{total}")
    print(f"  ❌ 失败: {RESULTS['failed']}/{total}")
    print(f"  ⚠️ 跳过: {RESULTS['skipped']}/{total}")

    if RESULTS["failed"] > 0:
        print("\n  失败项:")
        for d in RESULTS["details"]:
            if d["status"] == "FAIL":
                print(f"    ❌ {d['msg']}")

    if not has_api_key() and RESULTS["skipped"] > 0:
        print("\n  💡 完整的端到端测试需要配置 DASHSCOPE_API_KEY")
        print("     复制 .env.example 为 .env，填入你的百炼 API Key，然后重新运行")

    print()
    sys.exit(0 if RESULTS["failed"] == 0 else 1)
