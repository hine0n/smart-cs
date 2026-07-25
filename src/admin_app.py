"""知识库管理后台 - 入口。

运行：
    venv/Scripts/streamlit run admin_app.py --server.port 8501

面向运营 / 管理员：建库、喂数据、维护知识库。不含问答。
需先启动 FastAPI 后端（见 README）。
"""

from src.admin_ui import AdminUI


def main():
    AdminUI().render()


if __name__ == "__main__":
    main()
