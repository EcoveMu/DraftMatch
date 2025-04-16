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
from enhanced_extraction import enhanced_pdf_extraction, improved_matching_algorithm
from qwen_api import QwenOCR
from enhanced_extraction import enhanced_pdf_extraction
from comparison_algorithm_example import compare_documents
from custom_ai import CustomAI

# 檢查sentence-transformers是否可用
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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
    
    st.session_state.use_ocr = st.checkbox(
        "啟用 OCR", 
        value=st.session_state.use_ocr
    )
    
    st.session_state.use_ai = st.checkbox(
        "使用生成式 AI", 
        value=st.session_state.use_ai
    )
    
    if st.session_state.use_ai:
        st.session_state.ai_key = st.text_input(
            "🔑 請輸入 AI API 金鑰", 
            type="password",
            value=st.session_state.ai_key
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
        else:
            st.warning("未比對到有效段落，請檢查文件內容是否正確。")


# 檢查Java是否安裝
def is_java_installed():
    try:
        result = os.system("java -version > /dev/null 2>&1")
        return result == 0
    except:
        return False

# 檢查EasyOCR是否可用
def is_easyocr_available():
    try:
        import easyocr
        return True
    except ImportError:
        return False

# 檢查tabula-py是否可用
def is_tabula_available():
    if not is_java_installed():
        return False
    try:
        import tabula
        return True
    except ImportError:
        return False

# 檢查sentence-transformers是否可用
def is_sentence_transformers_available():
    return SENTENCE_TRANSFORMERS_AVAILABLE

# 加載語義模型
@st.cache_resource
def load_semantic_model():
    if is_sentence_transformers_available():
        try:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            return model
        except Exception as e:
            st.error(f"加載語義模型失敗: {e}")
            return None
    return None

# 生成式AI模型類
class GenerativeAI:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key
        self.is_available = self._check_availability()
    
    def _check_availability(self):
        """檢查模型是否可用"""
        if self.model_name in ["BERT多語言模型", "MPNet中文模型", "RoBERTa中文模型"]:
            # 檢查本地模型
            try:
                from transformers import AutoModel, AutoTokenizer
                return True
            except ImportError:
                return False
        elif self.model_name in ["OpenAI API", "Anthropic API", "Gemini API", "Qwen API"]:
            # 檢查API模型
            return self.api_key is not None and len(self.api_key) > 0
        return False
    
    def match_paragraphs(self, source_paragraphs, target_paragraphs):
        """使用生成式AI匹配段落"""
        if not self.is_available:
            return None
        
        if self.model_name == "BERT多語言模型":
            return self._match_with_bert(source_paragraphs, target_paragraphs)
        elif self.model_name == "MPNet中文模型":
            return self._match_with_mpnet(source_paragraphs, target_paragraphs)
        elif self.model_name == "RoBERTa中文模型":
            return self._match_with_roberta(source_paragraphs, target_paragraphs)
        elif self.model_name == "OpenAI API":
            return self._match_with_openai(source_paragraphs, target_paragraphs)
        elif self.model_name == "Anthropic API":
            return self._match_with_anthropic(source_paragraphs, target_paragraphs)
        elif self.model_name == "Gemini API":
            return self._match_with_gemini(source_paragraphs, target_paragraphs)
        elif self.model_name == "Qwen API":
            return self._match_with_qwen(source_paragraphs, target_paragraphs)
        
        return None
