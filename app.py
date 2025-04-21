import streamlit as st
import io, re, tempfile, os
from collections import defaultdict
from pathlib import Path

import fitz                     # PyMuPDF
import pandas as pd
import docx, difflib

import asyncio
import nest_asyncio

# 簡化異步處理初始化
try:
    nest_asyncio.apply()
except Exception:
    pass

from text_extraction import (
    extract_text_from_word,
    extract_text_from_pdf_with_page_info,
    extract_text_from_pdf_with_pdfplumber,
)
from comparison_algorithm import compare_pdf_first
from custom_ai import CustomAI
from table_compare import compare_tables
from table_viz import display_table_comparison
from sentence_compare import prepare_sentences, compare_sentences, split_into_sentences

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

if "debug_info" not in st.session_state:
    st.session_state.debug_info = False
if "comparison_done" not in st.session_state:
    st.session_state.comparison_done = False

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
    total = get_pdf_page_count(pdf_file)
    st.session_state.total_pages = total

    if total <= MAX_PAGES:
        st.info(f"PDF 共 {total} 頁，將全數比對。")
        pages = list(range(1, total + 1))
        if st.button("✅ 確定頁面", key="confirm_all_pages"):
            st.session_state.selected_pages = pages
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"已選擇頁面: {pages}")
            with col2:
                if st.button("🔄 重新選擇 PDF 頁面", key="reset_pages_btn"):
                    st.session_state.selected_pages = None
                    st.rerun()
        return

    # 大於 MAX_PAGES 的情況
    if st.session_state.selected_pages is None:
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
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"已選擇頁面: {st.session_state.selected_pages}")
            with col2:
                if st.button("🔄 重新選擇 PDF 頁面", key="reset_pages_btn"):
                    st.session_state.selected_pages = None
                    st.rerun()
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"已選擇頁面: {st.session_state.selected_pages}")
        with col2:
            if st.button("🔄 重新選擇 PDF 頁面", key="reset_pages_btn"):
                st.session_state.selected_pages = None
                st.rerun()


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

def display_matches(matches, page_number):
    """顯示句子匹配結果表格。"""
    df = pd.DataFrame({
        'PDF 句子': [m['pdf_text'] for m in matches],
        'Word 對應句子': [m['word_text'] for m in matches],
        'Word 段落編號': [', '.join(map(str, m['word_indices'])) for m in matches],
        'Word 頁碼': [m.get('word_page', "") for m in matches],
        '相似度': [f"{m['similarity']:.2%}" for m in matches]
    })
    st.dataframe(df, use_container_width=True)

def display_text_comparison(matches, page, results):
    """顯示文字比對結果"""
    if matches:
        display_matches(matches, page)
        with st.expander("查看文字差異詳細"):
            for m in matches:
                st.markdown(f"**相似度: {m['similarity']:.2%}**")
                st.markdown(m["diff_html"], unsafe_allow_html=True)
                st.divider()
    else:
        st.info("本頁沒有文字內容差異或匹配結果。")
    
    # 顯示未匹配段落
    page_unmatched = [
        para for para in results["unmatched_pdf"] 
        if para["page"] == page
    ]
    if page_unmatched:
        with st.expander("未匹配的段落"):
            for para in page_unmatched:
                st.markdown(f"- {para['content']}")

def display_comparison_results(results, pdf_bytes, word_data, pdf_data):
    """顯示比對結果"""
    try:
        # 添加結果頁面的控制按鈕
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("重新選擇頁面"):
                st.session_state.comparison_done = False
                st.experimental_rerun()
        with col2:
            st.session_state.debug_info = st.toggle("顯示調試資訊", st.session_state.debug_info)
        with col3:
            if st.button("返回首頁"):
                st.session_state.clear()
                st.experimental_rerun()

        # 如果開啟調試資訊，顯示詳細資訊
        if st.session_state.debug_info:
            with st.expander("調試資訊", expanded=True):
                # 計算總句子數
                word_sentences = sum(
                    len(split_into_sentences(p["content"]))
                    for p in word_data["paragraphs"]
                    if p.get("content")
                )
                pdf_sentences = sum(
                    len(split_into_sentences(p["content"]))
                    for p in pdf_data["paragraphs"]
                    if p.get("content")
                )
                
                st.write("Word 文件統計：", {
                    "段落數": len(word_data["paragraphs"]),
                    "表格數": len(word_data.get("tables", [])),
                    "句子數": word_sentences
                })
                st.write("PDF 文件統計：", {
                    "段落數": len(pdf_data["paragraphs"]),
                    "表格數": len(pdf_data.get("tables", [])),
                    "句子數": pdf_sentences
                })
                st.write("比對結果統計：", {
                    "文字匹配數": len(results.get("matches", [])),
                    "表格匹配數": len(results.get("table_matches", [])),
                    "未匹配段落數": len(results.get("unmatched_pdf", []))
                })

        # 顯示頁面選擇和比對結果
        selected_pages = st.multiselect(
            "選擇要查看的頁面",
            range(1, results.get("total_pages", 1) + 1),
            default=[1]
        )

        for p in selected_pages:
            st.subheader(f"PDF 第 {p} 頁")
            st.image(pdf_page_image(pdf_bytes, p), use_container_width=True)
            
            # 篩選本頁的文字和表格比對結果
            page_text_matches = [
                m for m in results.get("matches", []) 
                if m.get("pdf_page") == p
            ]
            page_table_matches = [
                t for t in results.get("table_matches", []) 
                if t.get("pdf_page") == p
            ]
            
            tab1, tab2 = st.tabs(["文字比對結果", "表格比對結果"])
            
            with tab1:
                if page_text_matches:
                    filtered_matches = [
                        m for m in page_text_matches 
                        if m['similarity'] >= 0.0 and 
                        (not False or m['similarity'] < 1.0)
                    ]
                    display_text_comparison(filtered_matches, p, results)
                else:
                    st.info("本頁沒有文字比對結果。")
            
            with tab2:
                if page_table_matches:
                    filtered_tables = [
                        t for t in page_table_matches
                        if t['similarity'] >= 0.0 and
                        (not False or t['similarity'] < 1.0)
                    ]
                    for table_match in filtered_tables:
                        display_table_comparison(table_match, pdf_data, word_data)
                        st.divider()
                else:
                    st.info("本頁沒有表格比對結果。")
    except Exception as e:
        st.error(f"顯示比對結果時發生錯誤：{str(e)}")
        st.exception(e)

###############################################################################
# ------------------------------ 主流程 UI -----------------------------------
###############################################################################

def main():
    try:
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

        if st.button("🚀 開始比對", use_container_width=True, disabled=start_btn_disabled, key="start_compare"):
            progress_bar = st.progress(0)
            
            def update_progress(progress):
                progress_bar.progress(progress)
            
            with st.spinner("正在進行文件比對..."):
                word_file = st.session_state.uploaded_word
                pdf_file = st.session_state.uploaded_pdf

                total_pages = get_pdf_page_count(pdf_file)
                pages = st.session_state.selected_pages

                # 組裝子 PDF：若PDF頁數過多則建立僅包含選定頁面的子PDF
                sub_pdf = build_sub_pdf(pdf_file, pages) if total_pages > MAX_PAGES else io.BytesIO(pdf_file.read())
                sub_pdf.seek(0)

                pdf_bytes = sub_pdf.getvalue()

                # 提取 Word 和 PDF 文本內容
                word_data = extract_text_from_word(word_file)
                if not isinstance(word_data, dict):
                    word_data = {
                        'paragraphs': word_data if isinstance(word_data, list) else [],
                        'tables': []
                    }

                # 使用 pdfplumber 提取 PDF 內容和表格
                pdf_data = extract_text_from_pdf_with_pdfplumber(sub_pdf)

                # 如果需要處理子PDF頁碼
                if total_pages > MAX_PAGES:
                    for para in pdf_data["paragraphs"]:
                        para["page"] = pages[para["page"] - 1]
                    for table in pdf_data["tables"]:
                        table["page"] = pages[table["page"] - 1]

                # AI / OCR 實例（如使用AI輔助比對）
                ai_instance = CustomAI() if st.session_state.use_ai else None

                st.info("比對中，請稍候...")
                
                # 準備句子列表
                pdf_sentences, word_sentences = prepare_sentences(
                    pdf_data,
                    word_data,
                    ignore_options={
                        "ignore_space": st.session_state.ignore_whitespace,
                        "ignore_punctuation": st.session_state.ignore_punctuation,
                        "ignore_case": st.session_state.ignore_case,
                        "ignore_newline": st.session_state.ignore_linebreaks,
                    }
                )
                
                # 1. 執行句子級比對
                text_results = compare_sentences(
                    word_sentences,
                    pdf_sentences,
                    comparison_mode=st.session_state.comparison_mode,
                    similarity_threshold=st.session_state.similarity_threshold,
                    ignore_options={
                        "ignore_space": st.session_state.ignore_whitespace,
                        "ignore_punctuation": st.session_state.ignore_punctuation,
                        "ignore_case": st.session_state.ignore_case,
                        "ignore_newline": st.session_state.ignore_linebreaks,
                    },
                    ai_instance=ai_instance,
                    progress_callback=update_progress
                )

                # 2. 執行表格比對
                table_results = compare_tables(
                    word_data["tables"], 
                    pdf_data["tables"],
                    similarity_threshold=st.session_state.similarity_threshold
                )

                # 3. 合併結果
                combined_results = {
                    **text_results,
                    "table_matches": table_results,
                    "total_pages": len(pages)
                }

                # 設置比對完成狀態
                st.session_state.comparison_done = True
                # 保存比對結果到 session_state
                st.session_state.comparison_results = combined_results
                st.session_state.word_data = word_data
                st.session_state.pdf_data = pdf_data
                st.session_state.pdf_bytes = pdf_bytes

        # 如果已完成比對，顯示結果
        if st.session_state.comparison_done:
            display_comparison_results(
                st.session_state.comparison_results,
                st.session_state.pdf_bytes,
                st.session_state.word_data,
                st.session_state.pdf_data
            )

    except Exception as e:
        st.error(f"執行時發生錯誤：{str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
