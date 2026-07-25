"""客户端问答前端（Streamlit）：面向终端客户。

只做一件事：客户提问 → 助手流式回答。
- 前台是「一个固定的智能体」：自动加载后台已发布（published）的 agent，客户无需、也无法选择任何东西。
- 打字机式流式输出。
- 只输出答案本身：显示级清理，绝不泄露 JSON / 评分 / 思考过程。
- 无任何管理入口（喂数据请用管理后台 admin_app.py）。
"""

import json

import requests
import streamlit as st

from src.api_client import ApiClient, clean_answer, clean_stream_frame

class CustomerUI:
    """终端客户问答界面（单一固定智能体）。"""

    def _init_session(self):
        defaults = {
            "chat_messages": [],
            "chat_agent": None,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    def _client(self) -> ApiClient:
        # 角色固定为 customer，操作日志写入 logs/customer.log
        return ApiClient(role="customer")

    # ---------- 自动加载已发布的智能体（客户无感） ----------

    def _resolve_agent(self):
        try:
            agents = self._client().list_agents(published=True)
        except Exception:
            agents = []
        # 同一时刻最多一个已发布；取第一个
        st.session_state.chat_agent = agents[0] if agents else None

    # ---------- 页头 ----------

    def _render_header(self, agent: dict):
        name = agent.get("name", "智能客服助手") if agent else "智能客服助手"
        st.markdown(f"💬 **{name}**")
        st.caption("在线为您服务 · 有任何问题都可以直接问我")

    # ---------- 参考来源卡片 ----------

    def _render_sources(self, sources):
        if not sources:
            return
        st.markdown("**参考来源**")
        for s in sources:
            src = s.get("source", "")
            snippet = s.get("content", "") or ""
            if snippet and len(snippet) > 120:
                snippet = snippet[:120] + "…"
            st.markdown(f"📄 **{src}**  \n{snippet or '—'}")

    # ---------- 历史消息（用户右、客服左） ----------

    def _render_history(self):
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                c_left, c_right = st.columns([35, 65])
                with c_right:
                    with st.chat_message("user", avatar="🙂"):
                        st.markdown(msg["content"])
            else:
                c_left, c_right = st.columns([65, 35])
                with c_left:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])
                        if msg.get("sources"):
                            with st.expander("参考来源", expanded=False):
                                self._render_sources(msg["sources"])

    # ---------- 一次问答（流式） ----------

    def _handle_question(self, agent: dict, question: str):
        st.session_state.chat_messages.append({"role": "user", "content": question})
        # 用户消息靠右（气泡占 65% 宽，更舒适）
        c_left, c_right = st.columns([35, 65])
        with c_right:
            with st.chat_message("user", avatar="🙂"):
                st.markdown(question)

        agent_id = agent["agent_id"]
        # 客服回答靠左（气泡占 65% 宽）
        c_left2, c_right2 = st.columns([65, 35])
        answer = ""
        with c_left2:
            with st.chat_message("assistant", avatar="🤖"):
                bubble = st.empty()
                # 首 token 到达前的「正在输入」动画
                bubble.markdown("_正在输入…_")
                raw = ""           # 后端原始 token 累积
                sources = []
                errored = False

                try:
                    resp = self._client().chat_stream_agent(agent_id, question)
                    try:
                        for event, data in ApiClient.iter_sse(resp):
                            if event == "token":
                                raw += data
                                # 显示级清理：剥离未闭合 JSON 残尾，只显示纯答案
                                bubble.markdown(
                                    clean_stream_frame(raw) + " ▌", unsafe_allow_html=True
                                )
                            elif event == "sources":
                                try:
                                    sources = json.loads(data)
                                except Exception:
                                    sources = []
                            elif event == "error":
                                bubble.error(data)
                                errored = True
                                break
                            elif event == "done":
                                break
                    finally:
                        resp.close()  # 确保流式 response 被释放，避免连接池堆积导致后续 Read timed out
                except requests.exceptions.Timeout:
                    # 超时：只给友好提示，绝不暴露 http://... / Read timed out 等技术细节
                    bubble.warning("⏳ 抱歉，回复超时了，请稍后重试，或换个更简洁的问法。")
                    return
                except requests.exceptions.ConnectionError:
                    bubble.error("⚠️ 与客服服务的连接中断，请重新提问。")
                    return
                except Exception:
                    # 兜底：任何底层异常都不向终端用户暴露技术堆栈
                    bubble.error("⚠️ 抱歉，暂时无法获取回复，请稍后重试。")
                    return

                if errored:
                    return

                # 定稿：完整清理后展示
                answer = clean_answer(raw)
                if not answer:
                    answer = "抱歉，我暂时没能找到相关信息，建议您联系人工客服。"
                bubble.markdown(answer, unsafe_allow_html=True)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    # ---------- 入口 ----------

    def render(self):
        st.set_page_config(page_title="智能客服助手", page_icon="💬", layout="wide")
        # 输入框与消息气泡等宽：限制 chat input 宽度与消息区一致
        st.markdown(
            "<style>"
            ".block-container { max-width: 760px; margin: 0 auto; padding-top: 1.2rem; }"
            "[data-testid='stChatInput'] { width: 100%; max-width: 65% !important; margin: 0 auto; }"
            "[data-testid='stChatInput'] textarea { width: 100% !important; }"
            "</style>",
            unsafe_allow_html=True,
        )
        self._init_session()

        self._resolve_agent()
        agent = st.session_state.chat_agent
        self._render_header(agent)

        if not agent:
            st.warning("⚠️ 客服系统暂未发布，请联系管理员在后台发布智能体。")
            if st.button("🔄 重试连接"):
                st.rerun()
            return

        self._render_history()

        question = st.chat_input("请输入您的问题...")
        if question:
            self._handle_question(agent, question)
