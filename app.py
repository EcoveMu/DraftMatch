
import streamlit as st
from pathlib import Path
import io, fitz, re, difflib, unicodedata, base64, tempfile, os
from comparison_algorithm import compare_pdf_first   # 新核心
from text_extraction import extract_text_from_word, extract_text_from_pdf_with_page_info

# ----------------------- UI 基本 -----------------------
st.set_page_config("期刊比對系統", "📊", layout="wide")
st.markdown("<h1 style='text-align:center;'>期刊比對系統</h1>", unsafe_allow_html=True)

MAX_PAGES = 20
if "selected_pages" not in st.session_state:
    st.session_state.selected_pages = None
if "pdf_info" not in st.session_state:
    st.session_state.pdf_info = {}

# ----------------------- SideBar 設定 -------------------
with st.sidebar:
    st.header("⚙️ 比對設定")
    mode = st.radio("比對模式", ["Hybrid", "Exact", "Semantic", "AI"], horizontal=True)
    threshold = st.slider("相似度閾值", 0.0, 1.0, 0.6, 0.05)

# ----------------------- 上傳文件 -----------------------
col1, col2 = st.columns(2)
with col1:
    word_file = st.file_uploader("上傳 Word (.docx)", type=["docx"])
with col2:
    pdf_file  = st.file_uploader("上傳 PDF (.pdf)", type=["pdf"])

# Utility ‑––––––––––––––––––––––––––––––––––––––––––––––
def get_pdf_page_count(uploaded):
    uploaded.seek(0)
    return fitz.open(stream=uploaded.read(), filetype="pdf").page_count

def build_sub_pdf(uploaded, pages:list[int])->io.BytesIO:
    uploaded.seek(0)
    src = fitz.open(stream=uploaded.read(), filetype="pdf")
    new = fitz.open()
    for p in pages:
        if 1 <= p <= src.page_count:
            new.insert_pdf(src, from_page=p-1, to_page=p-1)
    buf = io.BytesIO(new.tobytes())
    buf.seek(0)
    return buf

# ----------------------- PDF 頁面選擇 UI ---------------
if pdf_file:
    total_pages = get_pdf_page_count(pdf_file)
    st.session_state.pdf_info = {"total": total_pages}

    if total_pages > MAX_PAGES and st.session_state.selected_pages is None:
        st.warning(f"⚠️  目前上傳的 PDF 共 **{total_pages}** 頁。系統一次最多比對 {MAX_PAGES} 頁。")
        mode_pick = st.radio("選擇頁面方式", ["連續頁區間", "指定頁碼"])
        if mode_pick == "連續頁區間":
            c1, c2 = st.columns(2)
            start = c1.number_input("起始頁", 1, total_pages, 1, 1)
            end   = c2.number_input("結束頁", 1, total_pages, min(MAX_PAGES, total_pages), 1)
            end   = min(end, start+MAX_PAGES-1)
            pages = list(range(start, end+1))
        else:
            pages = st.multiselect("指定頁碼 (最多 20 頁)",
                                   list(range(1, total_pages+1)),
                                   max_selections=MAX_PAGES)
        if st.button("✅ 確定頁面"):
            if pages and 1 <= len(pages) <= MAX_PAGES:
                st.session_state.selected_pages = sorted(set(pages))
                st.success(f"已選擇頁面：{st.session_state.selected_pages}")
            else:
                st.error("請選擇 1‑20 頁。")

# 清除選頁
def reset_pages():
    st.session_state.selected_pages = None
    st.experimental_rerun()

# ----------------------- 開始比對 -----------------------
st.divider()
if st.button("🚀 開始比對"):
    if not word_file or not pdf_file:
        st.error("請同時上傳 Word 與 PDF。")
        st.stop()

    # 產生要比對的 PDF BytesIO
    if st.session_state.selected_pages:
        sub_pdf = build_sub_pdf(pdf_file, st.session_state.selected_pages)
    else:
        # 若頁數 <= MAX_PAGES 直接全檔比對
        if st.session_state.pdf_info.get("total", 0) > MAX_PAGES:
            st.error("請先選擇欲比對的 PDF 頁面。")
            st.stop()
        pdf_file.seek(0)
        sub_pdf = io.BytesIO(pdf_file.read())
        sub_pdf.seek(0)

    # ––– 抽取文字
    word_data = extract_text_from_word(word_file)
    pdf_paragraphs = extract_text_from_pdf_with_page_info(sub_pdf)
    pdf_data = {"paragraphs": pdf_paragraphs, "tables": []}

    # ––– 執行比對
    result = compare_pdf_first(
        word_data, pdf_data,
        comparison_mode=mode.lower(),
        similarity_threshold=threshold,
        ignore_options=dict(ignore_space=True, ignore_punctuation=True,
                            ignore_case=True, ignore_newline=True)
    )

    st.success(f"完成！共匹配 {result['statistics']['matched']} 段 / "
               f"PDF {result['statistics']['total_pdf']} 段。")

    # ---- 匹配結果表格 ----
    import pandas as pd, html
    df = pd.DataFrame([{
        "PDF頁": m["pdf_page"], "PDF段": m["pdf_index"],
        "Word段": m["word_index"], "相似度": f"{m['similarity']:.2f}"
    } for m in result["matches"]])
    st.dataframe(df, use_container_width=True)

    # ---- 展開差異 ----
    for m in result["matches"]:
        with st.expander(f"📄 PDF p{m['pdf_page']} 段 {m['pdf_index']} vs Word 段 {m['word_index']}  ({m['similarity']:.2f})"):
            st.markdown("**Word：**")
            st.markdown(m["word_text"], unsafe_allow_html=True)
            st.markdown("**PDF：**")
            st.markdown(m["pdf_text"], unsafe_allow_html=True)
            st.markdown("**差異：**", unsafe_allow_html=True)
            st.markdown(m["diff_html"], unsafe_allow_html=True)

    # ---- 重新選擇頁面 ----
    if st.button("🔄 重新選擇 PDF 頁面比對"):
        reset_pages()
