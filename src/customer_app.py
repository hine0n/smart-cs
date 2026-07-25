"""客户端问答 - 入口。

运行：
    venv/Scripts/streamlit run customer_app.py --server.port 8502

面向终端客户：只问问题、看流式答案，无管理功能。
需先启动 FastAPI 后端（见 README）。
"""

from src.customer_ui import CustomerUI


def main():
    CustomerUI().render()


if __name__ == "__main__":
    main()
