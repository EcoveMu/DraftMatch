
import streamlit as st
import os, tempfile, re, io
import fitz  # PyMuPDF
import docx, difflib
from pathlib import Path
import pandas as pd
from collections import defaultdict

from text_extraction import extract_text_from_word, extract_text_from_pdf_with_page_info
from comparison_algorithm import compare_pdf_first
from custom_ai import CustomAI
from qwen_ocr import QwenOCR

st.set_page_config(page_title="期刊比對系統", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown('<link rel="stylesheet" href="custom.css">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">期刊比對系統</h1>', unsafe_allow_html=True)
st.markdown('本系統用於比對原始Word文件與美編後PDF文件的內容差異，幫助校對人員快速找出不一致之處。')

# ------------- Session Defaults -----------------
DEFAULTS = dict(
    comparison_mode="hybrid",
    similarity_threshold=0.5,
    use_ocr=False,
    ocr_engine="qwen_builtin",
    ocr_api_key="",
    use_ai=False,
    ai_provider="deepseek_builtin",
    ai_api_key="",
    ignore_whitespace=True,
    ignore_punctuation=True,
    ignore_case=True,
    ignore_linebreaks=True,
    selected_pages=None,
    total_pages=None,
)
for k,v in DEFAULTS.items():
    st.session_state.setdefault(k,v)

# ------------- Side Bar -----------------
with st.sidebar:
    st.header("⚙️ 比對設定")

    # ---- 比對模式 ----
    mode_labels = ["混合比對（Hybrid）","精確比對（Exact）","語意比對（Semantic）","AI 比對"]
    mode_map = {
        "混合比對（Hybrid）":"hybrid",
        "精確比對（Exact）":"exact",
        "語意比對（Semantic）":"semantic",
        "AI 比對":"ai",
    }
    current_label = next(k for k,v in mode_map.items() if v == st.session_state.comparison_mode)
    selected_label = st.radio("比對模式", mode_labels, index=mode_labels.index(current_label))
    st.session_state.comparison_mode = mode_map[selected_label]

    st.session_state.similarity_threshold = st.slider("相似度閾值", 0.0, 1.0, st.session_state.similarity_threshold, 0.05)

    st.divider(); st.subheader("🔍 OCR 設定")
    st.session_state.use_ocr = st.checkbox("啟用 OCR", value=st.session_state.use_ocr)
    if st.session_state.use_ocr:
        ocr_choice = st.radio("OCR 引擎", ["Qwen（內建）","EasyOCR（內建）","Tesseract（內建）","自定義 OCR API"], horizontal=True)
        ocr_map = {
            "Qwen（內建）":"qwen_builtin",
            "EasyOCR（內建）":"easyocr",
            "Tesseract（內建）":"tesseract",
            "自定義 OCR API":"ocr_custom"
        }
        st.session_state.ocr_engine = ocr_map[ocr_choice]
        if st.session_state.ocr_engine in {"ocr_custom","qwen_api"}:
            st.session_state.ocr_api_key = st.text_input("OCR API Key", type="password", value=st.session_state.ocr_api_key)

    st.divider(); st.subheader("🤖 生成式 AI 設定")
    st.session_state.use_ai = st.checkbox("使用生成式 AI", value=st.session_state.use_ai)
    if st.session_state.use_ai:
        ai_choice = st.selectbox("AI 來源 / 模型", ["DeepSeek（內建）","Qwen2.5（內建）","OpenAI","Anthropic","自定義 AI API"],
                                 index=["deepseek_builtin","qwen_builtin","openai","anthropic","ai_custom"].index(st.session_state.ai_provider))
        st.session_state.ai_provider = {
            "DeepSeek（內建）":"deepseek_builtin","Qwen2.5（內建）":"qwen_builtin",
            "OpenAI":"openai","Anthropic":"anthropic","自定義 AI API":"ai_custom"
        }[ai_choice]
        if st.session_state.ai_provider in {"openai","anthropic","ai_custom"}:
            st.session_state.ai_api_key = st.text_input("AI API Key", type="password", value=st.session_state.ai_api_key)

    st.divider(); st.subheader("🧹 忽略規則")
    st.session_state.ignore_whitespace = st.checkbox("忽略空格", value=st.session_state.ignore_whitespace)
    st.session_state.ignore_punctuation = st.checkbox("忽略標點符號", value=st.session_state.ignore_punctuation)
    st.session_state.ignore_case = st.checkbox("忽略大小寫", value=st.session_state.ignore_case)
    st.session_state.ignore_linebreaks = st.checkbox("忽略斷行", value=st.session_state.ignore_linebreaks)

    st.divider(); st.subheader("ℹ️ 系統資訊")
    st.info("系統一次最多比對 20 頁 PDF，可多次選擇頁面進行逐批比對。")

# ------------- Upload Section -----------------
st.header("📁 文件上傳")
col1,col2 = st.columns(2)
with col1:
    st.subheader("原始Word文件")
    word_file = st.file_uploader("上傳原始 Word 文稿", type=["docx"], key="word_uploader")
    if word_file: st.success(f"已上傳: {word_file.name}")
with col2:
    st.subheader("美編後PDF文件")
    pdf_file = st.file_uploader("上傳美編後 PDF 文件", type=["pdf"], key="pdf_uploader")
    if pdf_file: st.success(f"已上傳: {pdf_file.name}")

MAX_PAGES = 20
def get_pdf_page_count(uploaded):
    pos=uploaded.tell(); uploaded.seek(0); count=len(fitz.open("pdf", uploaded.read())); uploaded.seek(pos); return count


def select_pdf_pages(pdf_file):
    if "selected_pages" not in st.session_state or st.session_state.selected_pages is None:
        total = get_pdf_page_count(pdf_file)
        st.session_state.total_pages = total
        if total > MAX_PAGES:
            st.warning(f"PDF 共 {total} 頁，系統一次最多比對 {MAX_PAGES} 頁，請選擇需比對頁面。")
            mode = st.radio("頁面選擇方式", ["連續區間","指定頁碼"])
            if mode == "連續區間":
                s, e = st.columns(2)
                start = s.number_input("起始頁", 1, total, 1, 1)
                end = e.number_input("結束頁", start, min(start + MAX_PAGES - 1, total), start + MAX_PAGES - 1, 1)
                pages = list(range(int(start), int(end) + 1))
            else:
                pages = st.multiselect("選擇頁碼", list(range(1, total + 1)))

            if st.button("✅ 確定頁面") and pages and 1 <= len(pages) <= MAX_PAGES:
                st.session_state.selected_pages = sorted(set(pages))
                st.success(f"已選擇頁面: {st.session_state.selected_pages}")

    if st.session_state.selected_pages:
        st.info(f"將比對頁面: {st.session_state.selected_pages}")
def build_sub_pdf(uploaded, pages):
    pos=uploaded.tell(); uploaded.seek(0); data=uploaded.read(); uploaded.seek(pos)
    src=fitz.open(stream=data, filetype="pdf")
    new=fitz.open()
    for p in pages:
        if 1<=p<=src.page_count:
            new.insert_pdf(src, from_page=p-1, to_page=p-1)
    buf=io.BytesIO(new.tobytes()); buf.seek(0); return buf

def pdf_page_image(pdf_bytes, page, zoom=1.0):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pix = doc.load_page(page-1).get_pixmap(matrix=fitz.Matrix(zoom,zoom))
        return pix.tobytes("png")

# Page selection UI
if pdf_file: select_pdf_pages(pdf_file)

# ------------- Comparison -----------------
st.markdown("---")
if st.button("🚀 開始比對", use_container_width=True):
    if not word_file or not pdf_file:
        st.error("請同時上傳 Word 與 PDF 文件")
        st.stop()
    # Prepare PDF bytes
    total_pages = get_pdf_page_count(pdf_file)
    if total_pages>MAX_PAGES and not st.session_state.selected_pages:
        st.error("請先選擇要比對的 PDF 頁面")
        st.stop()
    pages = st.session_state.selected_pages if st.session_state.selected_pages else list(range(1,total_pages+1))
    sub_pdf = build_sub_pdf(pdf_file, pages) if st.session_state.selected_pages else io.BytesIO(pdf_file.read())
    sub_pdf.seek(0)
    # Extract text
    word_data = extract_text_from_word(word_file)
    pdf_paragraphs = extract_text_from_pdf_with_page_info(sub_pdf)
    pdf_data = {"paragraphs": pdf_paragraphs, "tables":[]}
    # AI / OCR instance (stub)
    ai_instance = CustomAI() if st.session_state.use_ai else None
    # Compare
    st.info("比對中，請稍候...")
    res = compare_pdf_first(
        word_data, pdf_data,
        comparison_mode=st.session_state.comparison_mode,
        similarity_threshold=st.session_state.similarity_threshold,
        ignore_options=dict(
            ignore_space=st.session_state.ignore_whitespace,
            ignore_punctuation=st.session_state.ignore_punctuation,
            ignore_case=st.session_state.ignore_case,
            ignore_newline=st.session_state.ignore_linebreaks
        ),
        ai_instance=ai_instance
    )
    st.success(f"完成！匹配 {res['statistics']['matched']} 段 / PDF 段 {res['statistics']['total_pdf']}")

    # Group by page
    page_matches = defaultdict(list)
    for m in res['matches']:
        page_matches[m['pdf_page']].append(m)

    pdf_bytes = sub_pdf.getvalue()
    for p in pages:
        st.subheader(f"📄 PDF 頁 {p}")
        st.image(pdf_page_image(pdf_bytes,p,0.8), use_column_width=True)
        if p in page_matches:
            df = pd.DataFrame([{"Word段":m['word_index'],"相似度":f"{m['similarity']:.2f}"} for m in page_matches[p]])
            st.dataframe(df, use_container_width=True)
            for m in page_matches[p]:
                with st.expander(f"Word 段 {m['word_index']} (相似度 {m['similarity']:.2f})"):
                    st.markdown("**Word：**")
                    st.write(m['word_text'])
                    st.markdown("**PDF 片段：**")
                    st.write(m['pdf_text'])
                    st.markdown("**差異：**", unsafe_allow_html=True)
                    st.markdown(m['diff_html'], unsafe_allow_html=True)
        else:
            st.info("此頁無匹配段落")

    if st.button("🔄 重新選擇 PDF 頁面"):
        st.session_state.selected_pages = None
        st.experimental_rerun()