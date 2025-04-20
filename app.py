import streamlit as st
import io, re, tempfile, os
from collections import defaultdict
from pathlib import Path

import fitz                     # PyMuPDF
import pandas as pd
import docx, difflib

from text_extraction import (
    extract_text_from_word,
    extract_text_from_pdf_with_page_info,
)
from comparison_algorithm import compare_pdf_first
from custom_ai import CustomAI

###############################################################################
# --------------------------- 基本設定 ---------------------------------------
###############################################################################

st.set_page_config(
    page_title="期刊比對系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown('<link rel="stylesheet" href="custom.css">', unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>期刊比對系統</h1>", unsafe_allow_html=True)
st.markdown("本系統用於比對原始 Word 與美編後 PDF 的內容差異，協助校對人員快速找出不一致之處。")

###############################################################################
# --------------------------- SessionState 預設值 ----------------------------
###############################################################################

DEFAULTS = dict(
    comparison_mode="hybrid",
    similarity_threshold=0.6,
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
    uploaded_word=None,
    uploaded_pdf=None,
)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

###############################################################################
# ------------------------------ Sidebar -------------------------------------
###############################################################################

with st.sidebar:
    st.header("⚙️ 比對設定")

    mode_labels = {
        "混合比對（Hybrid）": "hybrid",
        "精確比對（Exact）": "exact",
        "語意比對（Semantic）": "semantic",
        "AI 比對": "ai",
    }
    # 依現有狀態找出中文標籤
    current_label = next(k for k, v in mode_labels.items() if v == st.session_state.comparison_mode)
    new_label = st.radio("比對模式", list(mode_labels.keys()), index=list(mode_labels.keys()).index(current_label))
    st.session_state.comparison_mode = mode_labels[new_label]

    st.session_state.similarity_threshold = st.slider(
        "相似度閾值", 0.0, 1.0, st.session_state.similarity_threshold, 0.05
    )

    # ------------------ OCR ------------------
    st.divider(); st.subheader("🔍 OCR 設定")
    st.session_state.use_ocr = st.checkbox("啟用 OCR", value=st.session_state.use_ocr)
    if st.session_state.use_ocr:
        ocr_labels = {
            "Qwen（內建）": "qwen_builtin",
            "EasyOCR（內建）": "easyocr",
            "Tesseract（內建）": "tesseract",
            "自定義 OCR API": "ocr_custom",
        }
        current = next(k for k, v in ocr_labels.items() if v == st.session_state.ocr_engine)
        ocr_label = st.radio("OCR 引擎", list(ocr_labels.keys()), horizontal=True, index=list(ocr_labels.keys()).index(current))
        st.session_state.ocr_engine = ocr_labels[ocr_label]
        if st.session_state.ocr_engine == "ocr_custom":
            st.session_state.ocr_api_key = st.text_input("OCR API Key", type="password", value=st.session_state.ocr_api_key)

    # ------------------ AI ------------------
    st.divider(); st.subheader("🤖 生成式 AI 設定")
    st.session_state.use_ai = st.checkbox("使用生成式 AI", value=st.session_state.use_ai)
    if st.session_state.use_ai:
        ai_labels = {
            "DeepSeek（內建）": "deepseek_builtin",
            "Qwen2.5（內建）": "qwen_builtin",
            "OpenAI": "openai",
            "Anthropic": "anthropic",
            "自定義 AI API": "ai_custom",
        }
        current_ai = next(k for k, v in ai_labels.items() if v == st.session_state.ai_provider)
        ai_label = st.selectbox("AI 來源 / 模型", list(ai_labels.keys()), index=list(ai_labels.keys()).index(current_ai))
        st.session_state.ai_provider = ai_labels[ai_label]
        if st.session_state.ai_provider in {"openai", "anthropic", "ai_custom"}:
            st.session_state.ai_api_key = st.text_input("AI API Key", type="password", value=st.session_state.ai_api_key)

    # ------------------ 忽略規則 ------------------
    st.divider(); st.subheader("🧹 忽略規則")
    st.session_state.ignore_whitespace = st.checkbox("忽略空格", value=st.session_state.ignore_whitespace)
    st.session_state.ignore_punctuation = st.checkbox("忽略標點符號", value=st.session_state.ignore_punctuation)
    st.session_state.ignore_case = st.checkbox("忽略大小寫", value=st.session_state.ignore_case)
    st.session_state.ignore_linebreaks = st.checkbox("忽略斷行", value=st.session_state.ignore_linebreaks)

    st.divider(); st.subheader("ℹ️ 系統資訊")
    st.info("系統一次最多比對 20 頁 PDF，可多次選擇頁面進行逐批比對。")

###############################################################################
# ----------------------------- 上傳檔案區 -----------------------------------
###############################################################################

st.header("📁 文件上傳")
col1, col2 = st.columns(2)

with col1:
    st.subheader("原始 Word 文件")
    word_file = st.file_uploader("上傳原始 Word 文稿", type=["docx"], key="word_uploader")
    if word_file:
        st.session_state.uploaded_word = word_file
        st.success(f"已上傳: {word_file.name}")

with col2:
    st.subheader("美編後 PDF 文件")
    pdf_file = st.file_uploader("上傳美編後 PDF 文件", type=["pdf"], key="pdf_uploader")
    if pdf_file:
        st.session_state.uploaded_pdf = pdf_file
        st.success(f"已上傳: {pdf_file.name}")

###############################################################################
# ------------------------ PDF 頁面選擇 & 工具函式 ---------------------------
###############################################################################

MAX_PAGES = 20

def get_pdf_page_count(uploaded):
    """快速計算頁數（不改變檔案指標）。"""
    pos = uploaded.tell()
    uploaded.seek(0)
    n_pages = len(fitz.open("pdf", uploaded.read()))
    uploaded.seek(pos)
    return n_pages


def select_pdf_pages(pdf_file):
    """顯示頁面選擇 UI，並把結果寫入 session_state.selected_pages"""
    if st.session_state.selected_pages is not None:
        if st.button("🔄 重新選擇 PDF 頁面", key="select_page_reset_btn"):
            st.session_state.selected_pages = None
            st.stop()

    total = get_pdf_page_count(pdf_file)
    st.session_state.total_pages = total

    if total <= MAX_PAGES:
        st.info(f"PDF 共 {total} 頁，將全數比對。")
        pages = list(range(1, total + 1))
        if st.button("✅ 確定頁面", key="confirm_all_pages"):
            st.session_state.selected_pages = pages
            st.success(f"已選擇頁面: {pages}")
        return

    st.warning(f"PDF 共 {total} 頁，系統一次最多比對 {MAX_PAGES} 頁，請選擇需比對頁面。")
    mode = st.radio("頁面選擇方式", ["連續區間", "指定頁碼"], key="page_select_mode")

    if mode == "連續區間":
        c1, c2 = st.columns(2)
        start = c1.number_input("起始頁", 1, total, 1, 1, key="start_page")
        end = c2.number_input(
            "結束頁",
            start,
            min(start + MAX_PAGES - 1, total),
            min(start + MAX_PAGES - 1, total),
            1,
            key="end_page",
        )
        pages = list(range(int(start), int(end) + 1))
    else:
        pages = st.multiselect("選擇頁碼", list(range(1, total + 1)), key="manual_pages")

    if st.button("✅ 確定頁面", key="confirm_pages") and pages and len(pages) <= MAX_PAGES:
        st.session_state.selected_pages = sorted(set(map(int, pages)))
        st.success(f"已選擇頁面: {st.session_state.selected_pages}")


def build_sub_pdf(uploaded, pages):
    """依選頁組裝新 PDF，傳回 BytesIO"""
    pos = uploaded.tell()
    uploaded.seek(0)
    data = uploaded.read()
    uploaded.seek(pos)

    src = fitz.open(stream=data, filetype="pdf")
    new = fitz.open()
    for p in pages:
        if 1 <= p <= src.page_count:
            new.insert_pdf(src, from_page=p - 1, to_page=p - 1)
    buf = io.BytesIO(new.tobytes())
    buf.seek(0)
    return buf


def pdf_page_image(pdf_bytes, page, zoom=0.8):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pix = doc.load_page(page - 1).get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")

###############################################################################
# ------------------------------ 主流程 UI -----------------------------------
###############################################################################

# 定義 start_btn_disabled
start_btn_disabled = not (
    st.session_state.uploaded_word and 
    st.session_state.uploaded_pdf and 
    st.session_state.selected_pages
)

# 處理PDF頁面選擇
if st.session_state.uploaded_pdf:
    select_pdf_pages(st.session_state.uploaded_pdf)

st.markdown("---")

# 使用 start_btn_disabled
if st.button("🚀 開始比對", use_container_width=True, disabled=start_btn_disabled, key="start_compare"):
    word_file = st.session_state.uploaded_word
    pdf_file = st.session_state.uploaded_pdf

    total_pages = get_pdf_page_count(pdf_file)
    pages = st.session_state.selected_pages

    # 組裝子 PDF：若PDF頁數過多則建立僅包含選定頁面的子PDF
    sub_pdf = build_sub_pdf(pdf_file, pages) if total_pages > MAX_PAGES else io.BytesIO(pdf_file.read())
    sub_pdf.seek(0)

    # 定義 pdf_bytes
    pdf_bytes = sub_pdf.getvalue()

    # 提取 Word 和 PDF 文本內容
    word_data = extract_text_from_word(word_file)
    pdf_paragraphs = extract_text_from_pdf_with_page_info(sub_pdf)

    # **處理子PDF頁碼**：如果使用了子PDF，將段落中的頁碼轉換回原始PDF的頁碼
    if total_pages > MAX_PAGES:
        for para in pdf_paragraphs:
            para["page"] = pages[para["page"] - 1]
    pdf_data = {"paragraphs": pdf_paragraphs, "tables": []}

    # AI / OCR 實例（如使用AI輔助比對）
    ai_instance = CustomAI() if st.session_state.use_ai else None

    st.info("比對中，請稍候...")
    res = compare_pdf_first(
        word_data,
        pdf_data,
        comparison_mode=st.session_state.comparison_mode,
        similarity_threshold=st.session_state.similarity_threshold,
        ignore_options={
            "ignore_space": st.session_state.ignore_whitespace,
            "ignore_punctuation": st.session_state.ignore_punctuation,
            "ignore_case": st.session_state.ignore_case,
            "ignore_newline": st.session_state.ignore_linebreaks,
        },
        ai_instance=ai_instance,
    )

    st.success(f"完成！匹配 {res['statistics']['matched']} 段 / PDF 段 {res['statistics']['total_pdf']}")
    
    # 比對完成後設置狀態
    st.session_state.comparison_done = True
        
    # 顯示每一頁結果
    for p in st.session_state.selected_pages:
        st.subheader(f"PDF 頁 {p}")
        try:
            st.image(pdf_page_image(pdf_bytes, p), use_container_width=True)
        except Exception as e:
            st.error(f"無法顯示頁面 {p} 圖像：{e}")
        # … match table & expander …
