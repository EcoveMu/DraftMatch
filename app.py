import streamlit as st
import os
import tempfile
import docx
import re
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

# 初始化會話狀態
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = "hybrid"
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.6
if 'use_ocr' not in st.session_state:
    st.session_state.use_ocr = False
if 'ocr_engine' not in st.session_state:
    st.session_state.ocr_engine = "Qwen"
if 'use_ai' not in st.session_state:
    st.session_state.use_ai = False
if 'ai_key' not in st.session_state:
    st.session_state.ai_key = ""
if 'ignore_whitespace' not in st.session_state:
    st.session_state.ignore_whitespace = True
if 'ignore_punctuation' not in st.session_state:
    st.session_state.ignore_punctuation = True
if 'ignore_case' not in st.session_state:
    st.session_state.ignore_case = True
if 'ignore_linebreaks' not in st.session_state:
    st.session_state.ignore_linebreaks = True

# Sidebar 設定
with st.sidebar:
    st.header("⚙️ 比對設定")

    st.session_state.comparison_mode = st.selectbox(
        "比對模式", 
        ["exact", "semantic", "hybrid", "ai"],
        index=["exact", "semantic", "hybrid", "ai"].index(st.session_state.comparison_mode)
    )
    
    st.session_state.similarity_threshold = st.slider(
        "相似度閾值", 
        0.0, 1.0, 
        st.session_state.similarity_threshold, 
        0.05
    )
    
    st.divider()
    st.subheader("🔍 OCR設置")
    
    st.session_state.use_ocr = st.checkbox(
        "啟用 OCR", 
        value=st.session_state.use_ocr
    )
    
    if st.session_state.use_ocr:
        st.session_state.ocr_engine = st.radio(
            "OCR引擎",
            ["Qwen", "EasyOCR", "Tesseract", "自定義API"],
            index=["Qwen", "EasyOCR", "Tesseract", "自定義API"].index(st.session_state.ocr_engine)
        )
        
        if st.session_state.ocr_engine == "Qwen" or st.session_state.ocr_engine == "自定義API":
            st.session_state.ai_key = st.text_input(
                "🔑 請輸入 OCR API 金鑰", 
                type="password",
                value=st.session_state.ai_key
            )
    
    st.divider()
    st.subheader("🤖 生成式AI設置")
    
    st.session_state.use_ai = st.checkbox(
        "使用生成式 AI", 
        value=st.session_state.use_ai
    )
    
    if st.session_state.use_ai and not st.session_state.ai_key:
        st.session_state.ai_key = st.text_input(
            "🔑 請輸入 AI API 金鑰", 
            type="password"
        )

    st.divider()
    st.subheader("🧹 忽略規則")
    
    st.session_state.ignore_whitespace = st.checkbox(
        "忽略空格", 
        value=st.session_state.ignore_whitespace
    )
    
    st.session_state.ignore_punctuation = st.checkbox(
        "忽略標點符號", 
        value=st.session_state.ignore_punctuation
    )
    
    st.session_state.ignore_case = st.checkbox(
        "忽略大小寫", 
        value=st.session_state.ignore_case
    )
    
    st.session_state.ignore_linebreaks = st.checkbox(
        "忽略斷行", 
        value=st.session_state.ignore_linebreaks
    )

    st.divider()
    st.subheader("ℹ️ 系統資訊")
    st.info("本系統用於比對原始Word文件與美編後PDF文件的內容差異，幫助校對人員快速找出不一致之處。")

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
def compare_documents(doc1_data, doc2_data, ignore_options=None, comparison_mode="hybrid", similarity_threshold=0.6, ai_instance=None):
    """比對兩個文檔的內容"""
    if ignore_options is None:
        ignore_options = {
            "ignore_whitespace": True,
            "ignore_punctuation": True,
            "ignore_case": True,
            "ignore_linebreaks": True
        }
    
    # 預處理文本
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
    
    # 預處理所有段落
    for para in doc1_data["paragraphs"]:
        para["processed_content"] = preprocess_text(para["content"])
    
    for para in doc2_data["paragraphs"]:
        para["processed_content"] = preprocess_text(para["content"])
    
    # 根據比對模式選擇不同的算法
    if comparison_mode == "exact":
        # 精確比對
        matches = []
        for doc1_para in doc1_data["paragraphs"]:
            for doc2_para in doc2_data["paragraphs"]:
                if doc1_para["processed_content"] == doc2_para["processed_content"]:
                    matches.append({
                        "doc1_index": doc1_para["index"],
                        "doc1_text": doc1_para["content"],
                        "doc2_index": doc2_para["index"],
                        "doc2_text": doc2_para["content"],
                        "similarity": 1.0,
                        "page_number": doc2_para.get("page", None)
                    })
                    break
        
        # 對於沒有精確匹配的段落，使用模糊匹配
        for doc1_para in doc1_data["paragraphs"]:
            if not any(match["doc1_index"] == doc1_para["index"] for match in matches):
                best_match = None
                best_similarity = 0
                
                for doc2_para in doc2_data["paragraphs"]:
                    similarity = difflib.SequenceMatcher(None, doc1_para["processed_content"], doc2_para["processed_content"]).ratio()
                    
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_match = doc2_para
                
                if best_match:
                    matches.append({
                        "doc1_index": doc1_para["index"],
                        "doc1_text": doc1_para["content"],
                        "doc2_index": best_match["index"],
                        "doc2_text": best_match["content"],
                        "similarity": best_similarity,
                        "page_number": best_match.get("page", None)
                    })
    
    elif comparison_mode == "semantic":
        # 語意比對
        if SENTENCE_TRANSFORMERS_AVAILABLE and ai_instance and ai_instance.is_available():
            # 使用AI進行語意比對
            matches = ai_instance.match_paragraphs(doc1_data["paragraphs"], doc2_data["paragraphs"])
        else:
            # 如果AI不可用，退回到模糊比對
            matches = improved_matching_algorithm(doc1_data, doc2_data, similarity_threshold)
    
    elif comparison_mode == "hybrid" or comparison_mode == "ai":
        # 混合比對或AI比對
        matches = improved_matching_algorithm(doc1_data, doc2_data, similarity_threshold)
    
    else:
        # 默認使用模糊比對
        matches = improved_matching_algorithm(doc1_data, doc2_data, similarity_threshold)
    
    # 為每個匹配生成差異標記
    for match in matches:
        # 字符級別差異
        d = difflib.Differ()
        diff = list(d.compare(match["doc1_text"], match["doc2_text"]))
        
        # 生成HTML差異顯示
        html_diff = []
        for i, s in enumerate(diff):
            if s.startswith('  '):  # 相同
                html_diff.append(s[2:])
            elif s.startswith('- '):  # 刪除
                if i+1 < len(diff) and diff[i+1].startswith('? '):
                    # 有標記，使用字符級別差異
                    markers = diff[i+1][2:]
                    s = s[2:]
                    html_s = ""
                    for j, c in enumerate(s):
                        if j < len(markers) and markers[j] in '-^':
                            html_s += f'<span class="diff-char-removed">{c}</span>'
                        else:
                            html_s += c
                    html_diff.append(f'<span class="diff-removed">{html_s}</span>')
                else:
                    # 沒有標記，使用行級別差異
                    html_diff.append(f'<span class="diff-removed">{s[2:]}</span>')
            elif s.startswith('+ '):  # 添加
                if i+1 < len(diff) and diff[i+1].startswith('? '):
                    # 有標記，使用字符級別差異
                    markers = diff[i+1][2:]
                    s = s[2:]
                    html_s = ""
                    for j, c in enumerate(s):
                        if j < len(markers) and markers[j] in '+^':
                            html_s += f'<span class="diff-char-added">{c}</span>'
                        else:
                            html_s += c
                    html_diff.append(f'<span class="diff-added">{html_s}</span>')
                else:
                    # 沒有標記，使用行級別差異
                    html_diff.append(f'<span class="diff-added">{s[2:]}</span>')
            elif s.startswith('? '):  # 標記行，已在上面處理
                continue
        
        match["diff_html"] = ''.join(html_diff)
    
    return matches

# 自定義AI類
class CustomAI:
    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key
        self.model_name = model_name
    
    def is_available(self):
        """檢查API是否可用"""
        return self.api_key is not None and len(self.api_key) > 0
    
    def match_paragraphs(self, source_paragraphs, target_paragraphs):
        """使用AI匹配段落"""
        # 簡化版本，實際上只是使用改進的匹配算法
        matches = []
        
        for source_para in source_paragraphs:
            best_match = None
            best_similarity = 0
            
            for target_para in target_paragraphs:
                # 使用difflib計算相似度
                similarity = difflib.SequenceMatcher(None, source_para["content"], target_para["content"]).ratio()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = target_para
            
            # 如果找到足夠相似的匹配
            if best_match and best_similarity >= 0.6:
                matches.append({
                    "doc1_index": source_para["index"],
                    "doc1_text": source_para["content"],
                    "doc2_index": best_match["index"],
                    "doc2_text": best_match["content"],
                    "similarity": best_similarity,
                    "page_number": best_match.get("page", None)
                })
        
        return matches

# 創建一個簡單的QwenOCR類，如果原始模組不可用
if not QWEN_OCR_AVAILABLE:
    class QwenOCR:
        def __init__(self, api_key=None, api_url=None):
            self.api_key = api_key
            self.api_url = api_url
        
        def is_available(self):
            return self.api_key is not None and len(self.api_key) > 0
        
        def extract_text_from_image(self, image_path):
            return "OCR功能需要安裝qwen_ocr模組"

if st.button("開始比對"):
    if (word_file is None or pdf_file is None) and not use_example_files:
        st.warning("請先上傳 Word 與 PDF 檔案，或選擇使用示例文件")
    else:
        st.info("🧠 開始比對中...")

        # 1. 保存上傳檔案至暫存
        if use_example_files:
            # 使用示例文件
            word_path = "比對素材-原稿.docx"  # 假設示例文件存在於當前目錄
            pdf_path = "比對素材-美編後完稿.pdf"  # 假設示例文件存在於當前目錄
            
            if not os.path.exists(word_path) or not os.path.exists(pdf_path):
                st.error("示例文件不存在，請上傳自己的文件")
                st.stop()
        else:
            # 使用上傳的文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_word:
                tmp_word.write(word_file.read())
                word_path = tmp_word.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name

        # 2. 進行文字抽取
        word_data, pdf_data = enhanced_pdf_extraction(word_path, pdf_path)

        # 3. 建立 AI 模型（如啟用）
        ai_instance = None
        if st.session_state.use_ai and st.session_state.ai_key:
            ai_instance = CustomAI(api_key=st.session_state.ai_key, model_name="Qwen")

        # 4. 執行比對演算法
        ignore_options = {
            "ignore_whitespace": st.session_state.ignore_whitespace,
            "ignore_punctuation": st.session_state.ignore_punctuation,
            "ignore_case": st.session_state.ignore_case,
            "ignore_linebreaks": st.session_state.ignore_linebreaks,
        }

        result = compare_documents(
            word_data,
            pdf_data,
            ignore_options=ignore_options,
            comparison_mode=st.session_state.comparison_mode,
            similarity_threshold=st.session_state.similarity_threshold,
            ai_instance=ai_instance
        )

        # 5. 顯示結果
        if result:
            st.success(f"比對完成，共處理 {len(result)} 組段落！")
            
            # 創建一個摘要表格
            summary_data = {
                "總段落數": len(word_data["paragraphs"]),
                "PDF段落數": len(pdf_data["paragraphs"]),
                "匹配段落數": len(result),
                "差異段落數": sum(1 for item in result if item["similarity"] < 1.0)
            }
            
            st.subheader("比對結果摘要")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("總段落數", summary_data["總段落數"])
            col2.metric("PDF段落數", summary_data["PDF段落數"])
            col3.metric("匹配段落數", summary_data["匹配段落數"])
            col4.metric("差異段落數", summary_data["差異段落數"])
            
            st.subheader("段落比對結果")
            
            # 按相似度排序結果（從低到高）
            sorted_result = sorted(result, key=lambda x: x["similarity"])
            
            for i, item in enumerate(sorted_result):
                similarity_class = ""
                if item["similarity"] >= 0.9:
                    similarity_class = "similarity-high"
                elif item["similarity"] >= 0.7:
                    similarity_class = "similarity-medium"
                else:
                    similarity_class = "similarity-low"
                
                with st.expander(f"段落 {i+1}: {item['doc1_text'][:50]}... (相似度: {item['similarity']:.2f})"):
                    st.markdown(f"**原始文本:**")
                    st.markdown(f"{item['doc1_text']}")
                    
                    st.markdown(f"**美編後文本:**")
                    if "page_number" in item and item["page_number"]:
                        st.markdown(f"頁碼: {item['page_number']}")
                    st.markdown(f"{item['doc2_text']}")
                    
                    st.markdown(f"**相似度:** <span class='{similarity_class}'>{item['similarity']:.2f}</span>", unsafe_allow_html=True)
                    
                    if "diff_html" in item and item["diff_html"]:
                        st.markdown("**差異顯示:**")
                        st.markdown(item["diff_html"], unsafe_allow_html=True)
                    
                    # 顯示PDF頁面預覽
                    if "page_number" in item and item["page_number"] and os.path.exists(pdf_path):
                        st.markdown("**PDF頁面預覽:**")
                        try:
                            doc = fitz.open(pdf_path)
                            page_num = item["page_number"] - 1  # 頁碼從1開始，但PyMuPDF從0開始
                            if 0 <= page_num < len(doc):
                                page = doc[page_num]
                                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                                img_bytes = pix.tobytes("png")
                                st.image(img_bytes, caption=f"頁碼: {item['page_number']}")
                            else:
                                st.warning(f"頁碼 {item['page_number']} 超出範圍")
                        except Exception as e:
                            st.error(f"無法顯示PDF頁面: {e}")
        else:
            st.warning("未比對到有效段落，請檢查文件內容是否正確。")
