import streamlit as st
import os
import tempfile
import docx
import re
from comparison_algorithm import semantic_matching
import difflib
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import json
import shutil
import sys
from pathlib import Path
import fitz  # PyMuPDF

# 檢查sentence-transformers是否可用
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# 檢查easyocr是否可用
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# 檢查tabula是否可用
try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

# 檢查pdfplumber是否可用
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# 檢查pytesseract是否可用
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# 檢查qwen_ocr模組是否可用
try:
    from qwen_ocr import QwenOCR
    QWEN_OCR_AVAILABLE = True
except ImportError:
    QWEN_OCR_AVAILABLE = False

# 設置頁面配置
st.set_page_config(
    page_title="期刊比對系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS樣式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .highlight-diff {
        background-color: #FFECB3;
        padding: 2px;
        border-radius: 3px;
    }
    .diff-added {
        color: #000000;
        background-color: #C8E6C9;
        padding: 2px;
        border-radius: 3px;
    }
    .diff-removed {
        color: #000000;
        background-color: #FFCDD2;
        padding: 2px;
        border-radius: 3px;
    }
    .result-container {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .similarity-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .similarity-medium {
        color: #F57F17;
        font-weight: bold;
    }
    .similarity-low {
        color: #C62828;
        font-weight: bold;
    }
    .table-warning {
        background-color: #FFF3E0;
        padding: 10px;
        border-left: 4px solid #FF9800;
        margin-bottom: 10px;
    }
    .file-uploader-container {
        border: 1px dashed #BDBDBD;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .multi-file-uploader {
        margin-bottom: 20px;
    }
    .chapter-selector {
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .diff-char-removed {
        color: #000000;
        background-color: #FFCDD2;
        font-weight: bold;
        padding: 1px;
        border-radius: 2px;
    }
    .diff-char-added {
        color: #000000;
        background-color: #C8E6C9;
        font-weight: bold;
        padding: 1px;
        border-radius: 2px;
    }
    .diff-section {
        margin-top: 10px;
        margin-bottom: 10px;
        padding: 10px;
        border: 1px solid #E0E0E0;
        border-radius: 5px;
    }
    .diff-navigation {
        margin-top: 10px;
        margin-bottom: 10px;
        text-align: center;
    }
    .diff-count {
        font-weight: bold;
        margin-left: 10px;
        margin-right: 10px;
    }
    .api-key-input {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .success-message {
        color: #2E7D32;
        background-color: #E8F5E9;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .error-message {
        color: #C62828;
        background-color: #FFEBEE;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .ai-model-section {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    /* 適應深色主題的樣式 */
    @media (prefers-color-scheme: dark) {
        .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown li {
            color: white !important;
        }
        .stText {
            color: white !important;
        }
        .stTextInput > div > div > input {
            color: white !important;
        }
        .stSelectbox > div > div > div {
            color: white !important;
        }
        .stSlider > div > div > div {
            color: white !important;
        }
        .stCheckbox > div > div > label {
            color: white !important;
        }
        .stExpander > div > div > div > div > p {
            color: white !important;
        }
        .stExpander > div > div > div > div > div > p {
            color: white !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 顯示標題
st.markdown('<h1 class="main-header">期刊比對系統</h1>', unsafe_allow_html=True)
st.markdown('本系統用於比對原始Word文件與美編後PDF文件的內容差異，幫助校對人員快速找出不一致之處。')

# --- SessionState 預設值與初始化 ------------------------------------
default_states = {
    "comparison_mode": "hybrid",
    "similarity_threshold": 0.6,
    "use_ocr": False,
    "ocr_engine": "qwen_builtin",
    "ocr_api_key": "",
    "use_ai": False,
    "ai_provider": "deepseek_builtin",
    "ai_api_key": "",
    "ignore_whitespace": True,
    "ignore_punctuation": True,
    "ignore_case": True,
    "ignore_linebreaks": True,
}
for k, v in default_states.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar 設定（繁體中文介面）
with st.sidebar:
    st.header("⚙️ 比對設定")

    mode_labels = {
        "精確比對（Exact）": "exact",
        "語意比對（Semantic）": "semantic",
        "混合比對（Hybrid）": "hybrid",
        "AI 比對": "ai",
    }
    # 取得當前模式對應的中文標籤
    current_mode_label = next(k for k,v in mode_labels.items() if v == st.session_state.comparison_mode)
    selected_label = st.selectbox("比對模式", list(mode_labels.keys()),
                                  index=list(mode_labels.keys()).index(current_mode_label))
    st.session_state.comparison_mode = mode_labels[selected_label]

    st.session_state.similarity_threshold = st.slider(
        "相似度閾值",
        0.0, 1.0,
        st.session_state.similarity_threshold,
        0.05
    )

    st.divider()
    st.subheader("🔍 OCR 設定")

    st.session_state.use_ocr = st.checkbox("啟用 OCR", value=st.session_state.use_ocr)

    if st.session_state.use_ocr:
        ocr_choice = st.radio(
            "OCR 引擎",
            ["Qwen（內建）", "EasyOCR", "Tesseract", "自定義 OCR API"],
            index=["qwen_builtin", "easyocr", "tesseract", "ocr_custom"]
            .index(st.session_state.ocr_engine)
        )
        st.session_state.ocr_engine = {
            "Qwen（內建）": "qwen_builtin",
            "EasyOCR": "easyocr",
            "Tesseract": "tesseract",
            "自定義 OCR API": "ocr_custom",
        }[ocr_choice]

        if st.session_state.ocr_engine == "ocr_custom":
            st.session_state.ocr_api_key = st.text_input(
                "🔑 請輸入 OCR API 金鑰",
                type="password",
                value=st.session_state.ocr_api_key
            )

    st.divider()
    st.subheader("🤖 生成式 AI 設定")

    st.session_state.use_ai = st.checkbox("使用生成式 AI", value=st.session_state.use_ai)

    if st.session_state.use_ai:
        ai_choice = st.selectbox(
            "AI 來源 / 模型",
            ["DeepSeek（內建）", "Qwen2.5（內建）", "Mistral‑7B（內建）",
             "OpenAI", "Anthropic", "Qwen (API)", "自定義 AI API"],
            index=[
                "deepseek_builtin","qwen_builtin","mistral_builtin",
                "openai","anthropic","qwen_api","ai_custom"
            ].index(st.session_state.ai_provider)
        )
        st.session_state.ai_provider = {
            "DeepSeek（內建）": "deepseek_builtin",
            "Qwen2.5（內建）": "qwen_builtin",
            "Mistral‑7B（內建）": "mistral_builtin",
            "OpenAI": "openai",
            "Anthropic": "anthropic",
            "Qwen (API)": "qwen_api",
            "自定義 AI API": "ai_custom",
        }[ai_choice]

        if st.session_state.ai_provider in {"openai","anthropic","qwen_api","ai_custom"}:
            st.session_state.ai_api_key = st.text_input(
                "🔑 生成式 AI API 金鑰",
                type="password",
                value=st.session_state.ai_api_key
            )

    st.divider()
    st.subheader("🧹 忽略規則")
    st.session_state.ignore_whitespace = st.checkbox("忽略空格", value=st.session_state.ignore_whitespace)
    st.session_state.ignore_punctuation = st.checkbox("忽略標點符號", value=st.session_state.ignore_punctuation)
    st.session_state.ignore_case = st.checkbox("忽略大小寫", value=st.session_state.ignore_case)
    st.session_state.ignore_linebreaks = st.checkbox("忽略斷行", value=st.session_state.ignore_linebreaks)

    st.divider()
    st.subheader("ℹ️ 系統資訊")
    st.info("本系統用於比對原始 Word 與 PDF 內容差異，協助校對。")

# 文件上傳區域
st.header("📁 文件上傳")

col1, col2 = st.columns(2)
with col1:
    st.subheader("原始Word文件")
    word_file = st.file_uploader("上傳原始 Word 文稿", type=["docx"])
    
    if word_file:
        st.success(f"已上傳: {word_file.name}")
        
with col2:
    st.subheader("美編後PDF文件")
    pdf_file = st.file_uploader("上傳美編後 PDF 文件", type=["pdf"])
    
    if pdf_file:
        st.success(f"已上傳: {pdf_file.name}")

# 使用示例文件選項
use_example_files = st.checkbox("使用示例文件進行演示", value=False)

# 文本提取和處理函數
def extract_text_from_word(word_file):
    """從Word文件中提取文本"""
    doc = docx.Document(word_file)
    
    paragraphs = []
    tables = []
    
    # 提取段落
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            paragraphs.append({
                "index": i,
                "content": text,
                "type": "paragraph"
            })
    
    # 提取表格
    for i, table in enumerate(doc.tables):
        table_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                row_data.append(cell.text.strip())
            table_data.append(row_data)
        
        if any(any(cell for cell in row) for row in table_data):
            tables.append({
                "index": i,
                "content": table_data,
                "type": "table"
            })
    
    return {
        "paragraphs": paragraphs,
        "tables": tables
    }

def extract_text_from_pdf(pdf_file):
    """從PDF文件中提取文本（使用PyMuPDF）"""
    doc = fitz.open(pdf_file)
    
    paragraphs = []
    page_texts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        page_texts.append(text)
        
        # 簡單地按行分割文本
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                paragraphs.append({
                    "index": len(paragraphs),
                    "content": line,
                    "type": "paragraph",
                    "page": page_num + 1
                })
    
    return {
        "paragraphs": paragraphs,
        "tables": [],  # 簡化版本不提取表格
        "page_texts": page_texts
    }

def enhanced_pdf_extraction(word_path, pdf_path):
    """增強版的文檔提取函數"""
    # 提取Word文檔內容
    if word_path.endswith('.docx'):
        word_data = extract_text_from_word(word_path)
    else:
        # 如果不是.docx文件，嘗試作為文本文件讀取
        try:
            with open(word_path, 'r', encoding='utf-8') as f:
                content = f.read()
                paragraphs = []
                for i, para in enumerate(content.split('\n\n')):
                    para = para.strip()
                    if para:
                        paragraphs.append({
                            "index": i,
                            "content": para,
                            "type": "paragraph"
                        })
                word_data = {
                    "paragraphs": paragraphs,
                    "tables": []
                }
        except Exception as e:
            st.error(f"無法讀取Word文件: {e}")
            return None, None
    
    # 提取PDF文檔內容
    if pdf_path.endswith('.pdf'):
        pdf_data = extract_text_from_pdf(pdf_path)
    else:
        # 如果不是.pdf文件，嘗試作為文本文件讀取
        try:
            with open(pdf_path, 'r', encoding='utf-8') as f:
                content = f.read()
                paragraphs = []
                for i, para in enumerate(content.split('\n\n')):
                    para = para.strip()
                    if para:
                        paragraphs.append({
                            "index": i,
                            "content": para,
                            "type": "paragraph",
                            "page": 1  # 假設只有一頁
                        })
                pdf_data = {
                    "paragraphs": paragraphs,
                    "tables": [],
                    "page_texts": [content]
                }
        except Exception as e:
            st.error(f"無法讀取PDF文件: {e}")
            return None, None
    
    return word_data, pdf_data

def improved_matching_algorithm(word_data, pdf_data, similarity_threshold=0.6):
    """改進的匹配算法"""
    matches = []
    
    # 對每個Word段落，找到最相似的PDF段落
    for word_para in word_data["paragraphs"]:
        best_match = None
        best_similarity = 0
        
        for pdf_para in pdf_data["paragraphs"]:
            # 使用difflib計算相似度
            similarity = difflib.SequenceMatcher(None, word_para["content"], pdf_para["content"]).ratio()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pdf_para
        
        # 如果找到足夠相似的匹配
        if best_match and best_similarity >= similarity_threshold:
            matches.append({
                "doc1_index": word_para["index"],
                "doc1_text": word_para["content"],
                "doc2_index": best_match["index"],
                "doc2_text": best_match["content"],
                "similarity": best_similarity,
                "page_number": best_match.get("page", None)
            })
    
    return matches

# 比對算法
def compare_documents(doc1_data, doc2_data, ignore_options=None, comparison_mode="hybrid",
                      similarity_threshold=0.6, ai_instance=None):
    """比對兩個文檔的內容，支援精確 / 語意 / 混合 / AI 模式，並內建進度條顯示"""
    if ignore_options is None:
        ignore_options = {
            "ignore_whitespace": True,
            "ignore_punctuation": True,
            "ignore_case": True,
            "ignore_linebreaks": True
        }

    # 預處理函式
    def preprocess_text(text):
        if ignore_options.get("ignore_whitespace", False):
            text = re.sub(r'\s+', ' ', text)
        if ignore_options.get("ignore_punctuation", False):
            text = re.sub(r'[^\w\s]', '', text)
        if ignore_options.get("ignore_case", False):
            text = text.lower()
        if ignore_options.get("ignore_linebreaks", False):
            text = text.replace('\n', ' ')
        return text.strip()

    # 整理段落
    for para in doc1_data["paragraphs"]:
        para["processed"] = preprocess_text(para["content"])
    for para in doc2_data["paragraphs"]:
        para["processed"] = preprocess_text(para["content"])

    total = len(doc1_data["paragraphs"])
    progress_bar = st.progress(0.0)

    matches = []
    for idx, src in enumerate(doc1_data["paragraphs"]):
        best_match = None
        best_sim = 0.0

        for tgt in doc2_data["paragraphs"]:
            if comparison_mode == "exact":
                sim = difflib.SequenceMatcher(None, src["processed"], tgt["processed"]).ratio()
            elif comparison_mode == "semantic":
                sim = semantic_matching(src["processed"], tgt["processed"])
            elif comparison_mode == "hybrid":
                exact_sim = difflib.SequenceMatcher(None, src["processed"], tgt["processed"]).ratio()
                if exact_sim >= similarity_threshold:
                    sim = exact_sim
                else:
                    sem_sim = semantic_matching(src["processed"], tgt["processed"])
                    sim = max(exact_sim, sem_sim)
            elif comparison_mode == "ai" and ai_instance:
                sim, _ = ai_instance.semantic_comparison(src["content"], tgt["content"])
            else:
                sim = 0.0

            if sim > best_sim:
                best_sim = sim
                best_match = tgt

        if best_match and best_sim >= similarity_threshold:
            # 生成差異 html
            d = difflib.Differ()
            diff = list(d.compare(src["content"], best_match["content"]))
            diff_html = []
            for i, s in enumerate(diff):
                if s.startswith('  '):
                    diff_html.append(s[2:])
                elif s.startswith('- '):
                    diff_html.append(f'<span class="diff-removed">{s[2:]}</span>')
                elif s.startswith('+ '):
                    diff_html.append(f'<span class="diff-added">{s[2:]}</span>')
            matches.append({
                "doc1_index": src["index"],
                "doc1_text": src["content"],
                "doc2_index": best_match["index"],
                "doc2_text": best_match["content"],
                "similarity": best_sim,
                "page_number": best_match.get("page", None),
                "diff_html": ''.join(diff_html)
            })

        progress_bar.progress((idx + 1) / total)

    return matches


# -------------------- 開始比對按鈕與結果展示 --------------------
st.markdown("---")
if st.button("🚀 開始比對", use_container_width=True):
    if not word_file and not use_example_files:
        st.error("請先上傳 Word 文件，或勾選使用示例文件。")
    elif not pdf_file and not use_example_files:
        st.error("請先上傳 PDF 文件，或勾選使用示例文件。")
    else:
        if use_example_files:
            # 從 example 資料夾讀入內建範例
            sample_dir = Path(__file__).parent / "examples"
            word_path = sample_dir / "sample.docx"
            pdf_path = sample_dir / "sample.pdf"
            if not word_path.exists() or not pdf_path.exists():
                st.error("找不到示例文件，請確認 examples 目錄存在 sample.docx / sample.pdf")
                st.stop()
            word_data = extract_text_from_word(str(word_path))
            pdf_data = extract_text_from_pdf(str(pdf_path))
        else:
            word_data = extract_text_from_word(word_file)
            pdf_data = extract_text_from_pdf(pdf_file)

        st.info("正在執行比對，請稍候...")
        matches = compare_documents(
            word_data, pdf_data,
            ignore_options={
                "ignore_whitespace": st.session_state.ignore_whitespace,
                "ignore_punctuation": st.session_state.ignore_punctuation,
                "ignore_case": st.session_state.ignore_case,
                "ignore_linebreaks": st.session_state.ignore_linebreaks,
            },
            comparison_mode=st.session_state.comparison_mode,
            similarity_threshold=st.session_state.similarity_threshold,
            ai_instance=None
        )
        if not matches:
            st.warning("未找到任何高於相似度閾值的匹配段落。")
        else:
            st.success(f"比對完成，共找到 {len(matches)} 組匹配！")

            # 統計與摘要
            total_paragraphs = len(word_data["paragraphs"])
            matched = len(matches)
            unmatched = total_paragraphs - matched
            match_rate = matched / total_paragraphs * 100 if total_paragraphs else 0.0

            st.subheader("📊 比對結果摘要")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Word段落總數", total_paragraphs)
            col_b.metric("匹配段落數", matched)
            col_c.metric("匹配率", f"{match_rate:.1f}%")

            # 詳細表格
            st.subheader("📄 詳細匹配結果")
            import pandas as pd
            df = pd.DataFrame(matches)
            df_display = df[["doc1_index","doc2_index","similarity","page_number"]]
            st.dataframe(df_display, use_container_width=True)

            # 可展開查看差異
            for m in matches:
                with st.expander(f"段落 {m['doc1_index']} ↔ PDF段落 {m['doc2_index']} (相似度 {m['similarity']:.2f})"):
                    st.markdown(f"**Word：** {m['doc1_text']}", unsafe_allow_html=True)
                    st.markdown(f"**PDF：** {m['doc2_text']}", unsafe_allow_html=True)
                    st.markdown("**差異：**", unsafe_allow_html=True)
                    st.markdown(m["diff_html"], unsafe_allow_html=True)
