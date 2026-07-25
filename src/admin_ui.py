"""管理后台前端（Streamlit）：专门用于「喂数据」。

面向运营/管理员，功能仅限知识库与文档管理：
- 知识库：卡片式选择 / 创建 / 删除
- 文档：上传文件 / 查看列表 / 删除

单一应用模型（后台前台一对一），无多租户 / 命名空间切换。
不含任何问答功能。问答请使用「客户端」（customer_app.py）。
"""

import streamlit as st

from config import SUPPORTED_FILE_TYPES
from src.api_client import ApiClient


class AdminUI:
    """知识库管理后台。"""

    def _init_session(self):
        defaults = {
            "admin_current_kb": None,
            "admin_kbs": [],
            "admin_agents": [],
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    def _client(self) -> ApiClient:
        return ApiClient(role="admin")

    # ---------- 顶部：品牌 + 刷新 ----------

    def _render_header(self):
        title_col, refresh_col = st.columns([0.82, 0.18])
        with title_col:
            st.title("🗂️ 知识库管理后台")
        with refresh_col:
            # 让刷新按钮与标题基线对齐，避免单独占满一行
            st.markdown("<div style='padding-top: 1.6rem;'></div>", unsafe_allow_html=True)
            if st.button("🔄 刷新", use_container_width=True):
                st.rerun()
        st.caption("建库 · 导入文档 · 配置智能体")

        # 拉知识库列表
        try:
            st.session_state.admin_kbs = self._client().list_kbs()
        except Exception as e:
            st.error(f"加载知识库失败：{e}")
            st.session_state.admin_kbs = []

    # ---------- 知识库选择（卡片式） ----------

    def _render_kb_selector(self):
        kbs = st.session_state.admin_kbs
        st.subheader("选择知识库")
        if not kbs:
            st.info("当前还没有知识库，请在下方创建。")
            return
        cols = st.columns(min(len(kbs), 4))
        current = st.session_state.admin_current_kb
        for i, kb in enumerate(kbs):
            with cols[i % 4]:
                selected = kb["name"] == current
                if st.button(
                    kb["name"],
                    key=f"kbcard_{kb['name']}",
                    type="primary" if selected else "secondary",
                    use_container_width=True,
                ):
                    if not selected:
                        st.session_state.admin_current_kb = kb["name"]
                        st.rerun()

    # ---------- 建库 / 删库 ----------

    def _render_kb_ops(self):
        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("新建知识库")
            new_kb = st.text_input("知识库名称", key="admin_new_kb", placeholder="如：faq")
            if st.button("创建", type="primary", use_container_width=True):
                if new_kb.strip():
                    r = self._client().create_kb(new_kb.strip())
                    if r.status_code == 201:
                        st.success(f"知识库 '{new_kb}' 创建成功！")
                        st.session_state.admin_current_kb = new_kb.strip()
                        st.rerun()
                    else:
                        detail = ""
                        try:
                            detail = r.json().get("detail", "")
                        except Exception:
                            pass
                        st.error(detail or "创建失败")
                else:
                    st.warning("请输入知识库名称")

        with c2:
            st.subheader("删除知识库")
            kb = st.session_state.admin_current_kb
            if kb:
                st.caption(f"将删除当前知识库：**{kb}**（含全部文档，不可恢复）")
                with st.popover("删除当前知识库", type="secondary", use_container_width=True):
                    st.warning(f"确认要删除知识库 **{kb}** 吗？\n\n此操作不可恢复！")
                    if st.button("确认删除", type="primary", key="confirm_del_kb"):
                        r = self._client().delete_kb(kb)
                        if r.ok:
                            st.success(f"已删除知识库 '{kb}'")
                            st.session_state.admin_current_kb = None
                            st.rerun()
                        else:
                            st.error("删除失败")
            else:
                st.caption("请先选择一个知识库。")

    # ---------- 喂数据：上传 ----------

    def _render_import(self):
        kb = st.session_state.admin_current_kb
        if not kb:
            return
        st.divider()
        st.subheader(f"向「{kb}」喂数据")
        st.caption("支持 PDF / TXT / DOCX / MD / CSV / XLSX（可多选；重复上传同名文件将覆盖更新）")

        uploaded = st.file_uploader(
            "上传文件",
            type=list(SUPPORTED_FILE_TYPES.keys()),
            accept_multiple_files=True,
            key="admin_uploader",
        )
        if uploaded and st.button("导入文件", type="primary", use_container_width=True):
            progress = st.progress(0, text="正在导入…")
            with st.spinner("正在解析并向量化，请稍候…"):
                files = [
                    ("files", (f.name, f.getvalue(), "application/octet-stream"))
                    for f in uploaded
                ]
                try:
                    r = self._client().upload_files(kb, files)
                    r.raise_for_status()
                    progress.progress(1.0, text="导入完成")
                    st.success(r.json().get("detail", "导入完成"))
                    st.rerun()
                except Exception as e:
                    progress.empty()
                    st.error(f"导入失败：{e}")

    # ---------- 文档列表 / 统计 ----------

    def _render_documents(self):
        kb = st.session_state.admin_current_kb
        if not kb:
            return
        st.divider()

        # 统计
        for info in st.session_state.admin_kbs:
            if info["name"] == kb:
                m1, m2 = st.columns(2)
                m1.metric("文档数", info.get("document_count", 0))
                m2.metric("片段数", info.get("chunk_count", 0))
                break

        st.subheader("已导入文档")
        try:
            sources = self._client().list_documents(kb)
        except Exception:
            sources = []

        if not sources:
            st.caption("暂无文档，请在上方导入。")
            return

        for src in sources:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"📄 `{src}`")
            with col2:
                with st.popover("删除", type="secondary"):
                    st.warning(f"确认删除 **{src}** ？")
                    if st.button("确认删除", key=f"confirm_del_doc_{src}", type="primary"):
                        try:
                            r = self._client().delete_document(kb, src)
                            if r.ok:
                                st.success(f"已删除 {src}")
                                st.rerun()
                            else:
                                st.error("删除失败")
                        except Exception as e:
                            st.error(str(e))

    def render(self):
        st.set_page_config(page_title="知识库管理后台", page_icon="🗂️", layout="wide")
        # 后台内容较多：整体适度限宽；并对输入框/下拉/多行文本做最大宽度约束，
        # 避免超宽屏下控件被无限拉伸，短输入（如知识库名）也能聚焦。
        st.markdown(
            """
            <style>
            .block-container { max-width: 1100px; margin: 0 auto; }
            .stTextInput { max-width: 520px; }
            .stTextArea { max-width: 640px; }
            .stSelectbox { max-width: 360px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        self._init_session()
        self._render_header()
        self._render_kb_selector()
        self._render_kb_ops()
        self._render_import()
        self._render_documents()
        self._render_agents()
        self._render_cache()

    # ---------- 问答缓存管理 ----------

    def _render_cache(self):
        st.divider()
        st.subheader("问答缓存管理")
        st.caption(
            "热门问题命中缓存会直接返回答案，跳过检索与 LLM 调用，显著降低延迟与成本。"
            "可在此查看缓存规模、热门问题与命中次数，必要时手动清理。"
        )

        try:
            stats = self._client().cache_stats()
        except Exception as e:
            st.error(f"加载缓存统计失败：{e}")
            return

        if not stats:
            st.warning("无法获取缓存统计（后端可能未运行或缓存未启用）。")
            return

        # ---------- 指标 ----------
        enabled = stats.get("enabled", False)
        entries = stats.get("entries", 0)
        max_entries = stats.get("max_entries", 0)
        db_bytes = stats.get("db_size_bytes", 0)
        ttl = stats.get("ttl_seconds", 0) or 0
        threshold = stats.get("semantic_threshold", 0)

        if db_bytes >= 1024 * 1024:
            size_str = f"{db_bytes / 1024 / 1024:.2f} MB"
        elif db_bytes >= 1024:
            size_str = f"{db_bytes / 1024:.1f} KB"
        else:
            size_str = f"{db_bytes} B"

        if ttl and ttl > 0:
            # 转成可读：天/时/分/秒
            d, rem = divmod(int(ttl), 86400)
            h, rem = divmod(rem, 3600)
            m, s = divmod(rem, 60)
            parts = []
            if d:
                parts.append(f"{d}天")
            if h:
                parts.append(f"{h}时")
            if m:
                parts.append(f"{m}分")
            if s and not parts:
                parts.append(f"{s}秒")
            ttl_str = "".join(parts) if parts else f"{ttl}秒"
        else:
            ttl_str = "永久（不失效）"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("缓存条目数", f"{entries} / {max_entries}")
        m2.metric("磁盘占用", size_str)
        m3.metric("TTL 过期", ttl_str)
        m4.metric("启用状态", "✅ 已启用" if enabled else "⛔ 已关闭")

        st.caption(
            f"语义命中阈值：余弦相似度 ≥ {threshold}（低于阈值则走精确或重新生成答案）"
        )

        # ---------- 热门问题 Top 列表 ----------
        st.markdown("#### 🔥 热门问题 Top")
        top = stats.get("top", []) or []
        if not top:
            st.info("暂无缓存数据。问答命中后热门问题将出现在此。")
        else:
            rows = []
            for i, item in enumerate(top, start=1):
                key = item.get("scope_and_key", "")
                scope, _, norm = key.partition(":::")
                dim_type, _, dim_name = scope.partition(":")
                dim_label = {"agent": "智能体", "kb": "知识库"}.get(dim_type, dim_type)
                rows.append({
                    "排名": i,
                    "问题": norm,
                    "维度": f"{dim_label}:{dim_name}",
                    "命中次数": item.get("hits", 0),
                    "答案长度(字)": item.get("answer_len", 0),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        # ---------- 清理 ----------
        st.markdown("#### 🧹 清理缓存")
        c1, c2 = st.columns(2)
        with c1:
            with st.popover("🗑️ 清空全部缓存", type="secondary", use_container_width=True):
                st.warning("确认清空**全部**缓存？所有维度的热门问题将被清除（不可恢复）。")
                if st.button("确认清空全部", type="primary", key="confirm_clear_all_cache"):
                    if self._client().clear_cache():
                        st.success("已全部清空")
                        st.rerun()
                    else:
                        st.error("清空失败")
        with c2:
            kb = st.session_state.admin_current_kb
            if kb:
                if st.button(
                    f"清空当前知识库「{kb}」缓存",
                    type="secondary", use_container_width=True,
                ):
                    if self._client().clear_cache(scope=f"kb:{kb}"):
                        st.success(f"已清空知识库「{kb}」的缓存")
                        st.rerun()
                    else:
                        st.error("清空失败")
            else:
                st.caption("选择一个知识库后可一键清空该维度缓存。")

        # 自定义维度
        scope_input = st.text_input(
            "按维度清理（可选）",
            placeholder="如：kb:faq 或 agent:agt_xxx",
            key="cache_scope_input",
        )
        if st.button("清空指定维度", use_container_width=True):
            s = scope_input.strip()
            if s:
                if self._client().clear_cache(scope=s):
                    st.success(f"已清空维度「{s}」")
                    st.rerun()
                else:
                    st.error("清空失败")
            else:
                st.warning("请输入维度，如 kb:faq")

    # ---------- 智能体（Agent）管理 ----------

    def _render_agents(self):
        st.divider()
        st.subheader("智能体管理 (Agent)")
        st.caption(
            "把「知识库 + 人设」绑定成一个智能体，并打开发布；"
            "前台客户端会自动加载已发布的那个，无需任何选择。"
        )

        try:
            agents = self._client().list_agents()
        except Exception:
            agents = []
        st.session_state.admin_agents = agents

        # 卡片网格概览
        if agents:
            cols = st.columns(min(len(agents), 2))
            for i, a in enumerate(agents):
                with cols[i % 2]:
                    badge_text = "🟢 已发布" if a.get("published") else "⚪ 未发布"
                    st.markdown(
                        f"**{a['name']}**  \n"
                        f"kb `{a.get('kb_name')}` · 语言 {a.get('language_mode')}  \n"
                        f"{badge_text}"
                    )
                    st.divider()
        else:
            st.info("尚无智能体，请在下方创建。")

        # 新建
        with st.expander("➕ 新建智能体"):
            with st.form("agent_create_form", clear_on_submit=True):
                name = st.text_input("智能体名称 *", placeholder="如：售后客服")
                kb_name = st.text_input("服务知识库 kb_name *", placeholder="如：faq")
                description = st.text_input("简介")
                system_prompt = st.text_area("人设提示词（留空用默认）", height=200)
                language_mode = st.selectbox(
                    "回答语言", ["auto", "zh", "en", "ja"], index=0,
                    help="auto = 与用户提问相同的语言（正常翻译）",
                )
                published = st.checkbox("发布到前台（客户端自动加载）")
                if st.form_submit_button("创建智能体", type="primary"):
                    if name.strip() and kb_name.strip():
                        r = self._client().create_agent({
                            "name": name.strip(),
                            "kb_name": kb_name.strip(),
                            "description": description,
                            "system_prompt": system_prompt,
                            "language_mode": language_mode,
                            "published": published,
                        })
                        if r.ok:
                            st.success(f"智能体 '{name}' 创建成功")
                            st.rerun()
                        else:
                            detail = ""
                            try:
                                detail = r.json().get("detail", "")
                            except Exception:
                                pass
                            st.error(f"创建失败：{detail or r.status_code}")
                    else:
                        st.warning("名称、kb_name 均为必填")

        # 逐个编辑 / 删除
        for a in agents:
            aid = a["agent_id"]
            with st.expander(f"✏️ 编辑：{a['name']}"):
                with st.form(f"agent_edit_{aid}"):
                    e_name = st.text_input("智能体名称", value=a["name"], key=f"e_name_{aid}")
                    e_kb = st.text_input("kb_name", value=a.get("kb_name", ""), key=f"e_kb_{aid}")
                    e_desc = st.text_input("简介", value=a.get("description", ""), key=f"e_desc_{aid}")
                    e_sp = st.text_area("人设提示词", value=a.get("system_prompt", ""), height=200, key=f"e_sp_{aid}")
                    lang_options = ["auto", "zh", "en", "ja"]
                    e_lang = st.selectbox(
                        "回答语言", lang_options,
                        index=lang_options.index(a.get("language_mode", "auto")),
                        key=f"e_lang_{aid}",
                    )
                    e_pub = st.checkbox("发布到前台", value=a.get("published", False), key=f"e_pub_{aid}")
                    if st.form_submit_button("保存", type="primary"):
                        r = self._client().update_agent(aid, {
                            "name": e_name,
                            "kb_name": e_kb,
                            "description": e_desc,
                            "system_prompt": e_sp,
                            "language_mode": e_lang,
                            "published": e_pub,
                        })
                        if r.ok:
                            st.success("已保存")
                            st.rerun()
                        else:
                            st.error("保存失败")

                # 删除放在 form 外部（form 内只允许 form_submit_button）
                with st.popover("删除", use_container_width=False):
                    st.warning(f"确认删除智能体 **{a['name']}** ？此操作不可恢复。")
                    if st.button("确认删除", key=f"del_agent_{aid}", type="primary"):
                        r = self._client().delete_agent(aid)
                        if r.ok:
                            st.success("已删除")
                            st.rerun()
                        else:
                            st.error("删除失败")
