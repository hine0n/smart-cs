"""问答缓存：热门问题命中缓存直接返回，跳过检索与 LLM 调用。

设计要点：
- key = 维度(agent_id|kb_name) + 归一化问题，按智能体/知识库隔离，避免跨库串答案
- 两级命中：
  1) 精确匹配（问题归一化后相等）——零额外调用，秒回
  2) 语义相似匹配（text-embedding-v3 向量余弦相似度 >= 阈值）——换问法也能命中
- embedding 调用复用 OpenAIEmbeddings（指向阿里云 DashScope），每次问答最多 1 次
- 持久化：SQLite（data/qa_cache.db），单文件、零额外依赖（标准库 sqlite3）、
  并发安全（每次操作新建连接 + RLock 串行化），并兼容从旧 qa_cache.json 一键迁移
- LRU 上限淘汰 + 命中计数 + 可选 TTL(秒)

对外接口与旧版 JSON 实现完全一致，调用方（chat.py / cache.py / knowledge_bases.py）无需改动。
"""
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from unicodedata import normalize

import numpy as np
from langchain_openai import OpenAIEmbeddings

from config import (
    PROJECT_ROOT,
    QA_CACHE_ENABLED,
    QA_CACHE_MAX_ENTRIES,
    QA_CACHE_TTL,
    QA_CACHE_SEMANTIC_THRESHOLD,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
)

_DB_PATH = PROJECT_ROOT / "data" / "qa_cache.db"
_JSON_PATH = PROJECT_ROOT / "data" / "qa_cache.json"
_WS_RE = re.compile(r"\s+", re.UNICODE)

# embedding 客户端惰性单例（避免启动时即建立连接）
_embeddings = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            check_embedding_ctx_length=False,
        )
    return _embeddings


def _serialize_emb(emb) -> bytes | None:
    """向量 → BLOB（float32 紧凑二进制）。"""
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32).tobytes()


def _deserialize_emb(blob) -> np.ndarray | None:
    """BLOB → 一维 float32 向量。"""
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


class QACache:
    def __init__(self):
        # 可重入锁：避免任何嵌套持锁导致死锁（语义命中路径会二次进入）
        self._lock = threading.RLock()
        if QA_CACHE_ENABLED:
            needs_migrate = (not _DB_PATH.exists()) and _JSON_PATH.exists()
            self._init_db()
            if needs_migrate:
                self._migrate_from_json()

    # ---------- 底层 DB 访问 ----------
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(_DB_PATH))

    def _init_db(self):
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS qa_cache (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope      TEXT NOT NULL,
                    norm       TEXT NOT NULL,
                    answer     TEXT NOT NULL DEFAULT '',
                    sources    TEXT NOT NULL DEFAULT '[]',
                    embedding  BLOB,
                    hits       INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0,
                    UNIQUE(scope, norm)
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_cache_scope ON qa_cache(scope)"
            )
            conn.commit()

    def _migrate_from_json(self):
        """把旧版 qa_cache.json 迁移进 SQLite，并备份原文件为 *.migrated。"""
        try:
            data = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict) or not data:
            return
        rows = []
        for key, v in data.items():
            if ":::" not in key:
                continue
            scope, norm = key.split(":::", 1)
            now = time.time()
            rows.append(
                (
                    scope,
                    norm,
                    v.get("answer", ""),
                    json.dumps(v.get("sources", []), ensure_ascii=False),
                    _serialize_emb(v.get("embedding")),
                    int(v.get("hits", 0)),
                    float(v.get("created_at", 0)) or now,
                    float(v.get("updated_at", 0)) or now,
                )
            )
        if not rows:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO qa_cache
                   (scope, norm, answer, sources, embedding, hits, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
        # 迁移成功后备份旧 JSON，避免重复迁移，同时保留回退可能
        try:
            _JSON_PATH.rename(_JSON_PATH.with_name("qa_cache.json.migrated"))
        except Exception:
            pass

    # ---------- 公开接口 ----------
    @staticmethod
    def normalize(question: str) -> str:
        """问题归一化：NFKC 全角转半角 + 去首尾空白 + 压缩内部空白。"""
        q = normalize("NFKC", question or "")
        q = q.strip()
        q = _WS_RE.sub(" ", q)
        return q

    def embed(self, question: str) -> list:
        """用 embedding 模型算问题向量（一次远程调用）。语义命中前必调。"""
        norm = self.normalize(question)
        return _get_embeddings().embed_documents([norm])[0]

    def get(self, question: str, scope: str, emb=None):
        """两级命中。

        返回 (answer, sources, hit_type) 或 None。
        hit_type: "exact"(精确) | "semantic"(语义相似)。
        emb 为 None 时只做精确匹配；提供 emb 时额外做语义相似匹配。
        """
        if not QA_CACHE_ENABLED:
            return None

        norm = self.normalize(question)

        # 1) 精确命中（零成本）
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT answer, sources, hits, updated_at FROM qa_cache "
                "WHERE scope=? AND norm=?",
                (scope, norm),
            ).fetchone()
            if row is not None:
                answer, sources_json, _hits, updated_at = row
                now = time.time()
                if QA_CACHE_TTL and now - updated_at > QA_CACHE_TTL:
                    conn.execute(
                        "DELETE FROM qa_cache WHERE scope=? AND norm=?",
                        (scope, norm),
                    )
                    conn.commit()
                    return None
                conn.execute(
                    "UPDATE qa_cache SET hits=hits+1, updated_at=? "
                    "WHERE scope=? AND norm=?",
                    (now, scope, norm),
                )
                conn.commit()
                sources = json.loads(sources_json) if sources_json else []
                return (answer, sources, "exact")

        # 2) 语义相似命中（需 emb）
        if emb is not None:
            return self._semantic_get(scope, emb)
        return None

    def _semantic_get(self, scope: str, emb):
        """对该 scope 下所有带 embedding 的缓存条目做余弦相似度，取最高且达阈值者。"""
        q = np.asarray(emb, dtype=np.float32)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT norm, embedding, answer, sources, hits, updated_at "
                "FROM qa_cache WHERE scope=? AND embedding IS NOT NULL",
                (scope,),
            ).fetchall()
        if not rows:
            return None
        try:
            vecs = np.stack([_deserialize_emb(r[1]) for r in rows])  # (n, dim)
            qn = float(np.linalg.norm(q)) or 1e-9
            norms = np.linalg.norm(vecs, axis=1)
            sims = vecs.dot(q) / (norms * qn + 1e-9)
            best = int(np.argmax(sims))
            if float(sims[best]) >= QA_CACHE_SEMANTIC_THRESHOLD:
                norm, _emb, answer, sources_json, _hits, _updated = rows[best]
                now = time.time()
                with self._lock, self._connect() as conn:
                    conn.execute(
                        "UPDATE qa_cache SET hits=hits+1, updated_at=? "
                        "WHERE scope=? AND norm=?",
                        (now, scope, norm),
                    )
                    conn.commit()
                sources = json.loads(sources_json) if sources_json else []
                return (answer, sources, "semantic")
        except Exception:
            return None
        return None

    def put(self, question: str, scope: str, answer: str, sources: list, emb=None):
        """写入/更新缓存；emb 提供时一并持久化向量，供未来语义命中。"""
        if not QA_CACHE_ENABLED:
            return
        norm = self.normalize(question)
        now = time.time()
        answer = answer or ""
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        emb_blob = _serialize_emb(emb)

        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO qa_cache
                       (scope, norm, answer, sources, embedding, hits, created_at, updated_at)
                   VALUES (?,?,?,?,?, 1, ?, ?)
                   ON CONFLICT(scope, norm) DO UPDATE SET
                       answer      = excluded.answer,
                       sources     = excluded.sources,
                       embedding   = COALESCE(excluded.embedding, qa_cache.embedding),
                       hits        = qa_cache.hits + 1,
                       created_at  = COALESCE(qa_cache.created_at, excluded.created_at),
                       updated_at  = excluded.updated_at
                """,
                (scope, norm, answer, sources_json, emb_blob, now, now),
            )

            # LRU 上限淘汰：超出则删最久未更新者
            cur = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
            overflow = cur - QA_CACHE_MAX_ENTRIES
            if overflow > 0:
                conn.execute(
                    "DELETE FROM qa_cache WHERE id IN ("
                    "SELECT id FROM qa_cache ORDER BY updated_at ASC LIMIT ?)",
                    (overflow,),
                )
            conn.commit()

    def clear(self, scope: str = None):
        """清缓存。scope 省略清全部；指定则清该维度(如 'kb:xxx' / 'agent:yyy')。"""
        with self._lock, self._connect() as conn:
            if scope is None:
                conn.execute("DELETE FROM qa_cache")
            else:
                conn.execute("DELETE FROM qa_cache WHERE scope=?", (scope,))
            conn.commit()

    def stats(self):
        with self._lock, self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
            rows = conn.execute(
                "SELECT scope, norm, hits, LENGTH(answer) "
                "FROM qa_cache ORDER BY hits DESC LIMIT 20"
            ).fetchall()
        top = [
            {
                "scope_and_key": f"{scope}:::{norm}",
                "hits": hits,
                "answer_len": ans_len or 0,
            }
            for scope, norm, hits, ans_len in rows
        ]
        # 磁盘容量：缓存数据库文件大小（enable 但未初始化时文件可能不存在 → 0）
        db_size = 0
        try:
            db_size = _DB_PATH.stat().st_size
        except OSError:
            db_size = 0
        return {
            "enabled": QA_CACHE_ENABLED,
            "entries": total,
            "max_entries": QA_CACHE_MAX_ENTRIES,
            "db_size_bytes": db_size,
            "ttl_seconds": QA_CACHE_TTL,
            "semantic_threshold": QA_CACHE_SEMANTIC_THRESHOLD,
            "top": top,
        }


qa_cache = QACache()
