"""文档加载与处理模块"""

import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_FILE_TYPES


def _excel_cell_text(v) -> str | None:
    """把 Excel 单元格值规整为字符串。

    - None / NaN / 空字符串 / 'nan' / 'none' / 'nat' 视为空，返回 None
    - 其余原样保留为字符串（保留前导零、电话、编码等原始文本，不被类型推断篡改）
    """
    import math

    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "nat"):
        return None
    return s


class DocumentLoader:
    """通用文档加载器，支持 PDF/TXT/DOCX/MD/CSV/XLSX（Excel 解析为结构化 JSON）"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def get_file_type(self, file_path: str) -> str:
        """获取文件类型"""
        ext = Path(file_path).suffix.lower().lstrip(".")
        if ext not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"不支持的文件类型: .{ext}，支持: {list(SUPPORTED_FILE_TYPES.keys())}")
        return ext

    def load_file(self, file_path: str, source_name: str = None) -> List[Document]:
        """加载单个文件并分割

        Args:
            file_path: 文件在磁盘上的路径
            source_name: 用于标注来源的文件名。为 None 时取 file_path 的文件名
                         （用于上传场景：临时文件路径无意义，应传入用户原始文件名）
        """
        ext = self.get_file_type(file_path)

        # Excel：多 sheet 表格，逐行解析为「保留行列关系」的结构化 JSON（专用解析，不改动其他类型逻辑）
        if ext == "xlsx":
            return self._load_excel(file_path, source_name)

        loaders = {
            "pdf": PyPDFLoader,
            "txt": lambda p: TextLoader(p, encoding="utf-8"),
            "docx": Docx2txtLoader,
            "md": lambda p: TextLoader(p, encoding="utf-8"),
            "csv": CSVLoader,
        }

        loader = loaders[ext](file_path)
        docs = loader.load()

        if not docs:
            return []

        # 分割文档
        chunks = self.text_splitter.split_documents(docs)

        # 为每个 chunk 添加来源信息
        filename = source_name or Path(file_path).name
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["file_type"] = ext

        return chunks

    def _load_excel(self, file_path: str, source_name: str = None) -> List[Document]:
        """加载 Excel 文件（.xlsx），逐行解析为「保留行列关系」的结构化 JSON。

        设计要点（直接修复「拿不到数据」的根因）：
        1. 用 `header=None` 读取，不做表头自动探测 → 即便表头不在第 1 行、
           存在空白列 / 合并单元格导致列名变 `Unnamed`，也不会整表被丢弃。
        2. 用 `dtype=str` 读取，所有单元格原样保留为文本 → 数字、电话号码、
           带前导零的编码（如 `00123`）、日期等不会被类型推断篡改或丢精度。
        3. 首行作为表头；空列名 / `Unnamed` 退化为 `列N`，保证列索引连续不丢。
        4. 每行生成一条 JSON 记录 `{"sheet","row","data":{列名:值,...}}`，
           作为原子 chunk，**不再二次切分**，彻底保留列→值的行列关系，
           也避免切断 JSON 破坏结构。
        5. 使用 `with` 上下文管理 ExcelFile，读取后及时释放文件句柄，
           避免 Windows 下临时文件删除被锁（影响上传场景清理）。
        """
        import json
        import pandas as pd

        filename = source_name or Path(file_path).name
        docs = []
        try:
            with pd.ExcelFile(file_path, engine="openpyxl") as xls:
                for sheet_name in xls.sheet_names:
                    # 以字符串 + 无表头方式读取整张表，杜绝丢列 / 改值
                    raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=str)
                    if raw is None or raw.empty:
                        continue
                    rows = raw.values.tolist()
                    if not rows:
                        continue

                    # 首行作为表头；空 / Unnamed 退化为 列N
                    header = []
                    for i, c in enumerate(rows[0]):
                        h = _excel_cell_text(c)
                        if not h or h.startswith("Unnamed"):
                            h = f"列{i + 1}"
                        header.append(h)

                    # 其余行逐行转为结构化 JSON（保留 列 -> 值 的行列关系）
                    for ridx, row in enumerate(rows[1:], start=1):
                        record = {}
                        for ci, cell in enumerate(row):
                            col_name = header[ci] if ci < len(header) else f"列{ci + 1}"
                            v = _excel_cell_text(cell)
                            if v is None:
                                continue
                            record[col_name] = v
                        if not record:
                            continue
                        obj = {"sheet": sheet_name, "row": ridx, "data": record}
                        text = json.dumps(obj, ensure_ascii=False)
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "file_type": "xlsx",
                                    "sheet": sheet_name,
                                    "row": ridx,
                                },
                            )
                        )
        except Exception as e:
            raise ValueError(f"读取 Excel 失败（请确认文件为合法的 .xlsx）: {e}")

        if not docs:
            return []

        # 每行已是原子记录，直接作为 chunk，不再二次切分（避免切断 JSON / 破坏行列关系）
        for d in docs:
            d.metadata["source"] = filename
            d.metadata["file_type"] = "xlsx"
        return docs

    def load_uploaded_file(self, file_content: bytes, filename: str) -> List[Document]:
        """加载上传的文件（从内存中）"""
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"不支持的文件类型: .{ext}")

        # 写入临时文件
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{ext}", mode="wb"
        ) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            # 用用户上传的原始文件名作 source，而不是临时文件路径名
            return self.load_file(tmp_path, source_name=filename)
        finally:
            os.unlink(tmp_path)

    def load_text(self, text: str, source: str = "手动输入") -> List[Document]:
        """加载纯文本"""
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self.text_splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata["source"] = source
        return chunks
