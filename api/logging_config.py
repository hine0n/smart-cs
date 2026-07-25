"""后端日志配置。

所有 API 请求写入 logs/backend.log（滚动，单文件最大 5MB，保留 5 份备份），
同时输出到控制台，便于排查。覆盖「后台」全部请求记录。
"""

import logging
from logging.handlers import RotatingFileHandler

from config import LOG_DIR

_logger = None


def get_backend_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("knowledge_mesh.backend")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = RotatingFileHandler(
            LOG_DIR / "backend.log",
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    _logger = logger
    return logger
