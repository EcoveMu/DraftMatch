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
</style>
""", unsafe_allow_html=True)

# 顯示標題
st.markdown('<h1 class="main-header">期刊比對系統</h1>', unsafe_allow_html=True)
st.markdown('本系統用於比對原始Word文件與美編後PDF文件的內容差異，幫助校對人員快速找出不一致之處。')

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

# 顯示系統狀態
with st.expander("系統狀態", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if is_java_installed():
            st.success("Java已安裝")
        else:
            st.error("Java未安裝，表格提取功能可能受限")
    
    with col2:
        if is_easyocr_available():
            st.success("OpenOCR (EasyOCR) 已安裝")
        else:
            st.warning("OpenOCR未安裝，將使用Tesseract作為替代")
    
    with col3:
        if is_tabula_available():
            st.success("表格提取工具已安裝")
        else:
            st.error("表格提取工具未安裝或無法使用")
    
    with col4:
        if is_sentence_transformers_available():
            st.success("語義模型已安裝")
        else:
            st.warning("語義模型未安裝，語義比對功能將使用簡化版")
    
    if not is_java_installed():
        st.info("安裝Java: 請執行 'sudo apt-get install -y default-jre' 或安裝適合您系統的Java運行環境")
    
    if not is_easyocr_available():
        st.info("安裝OpenOCR: 請執行 'pip install easyocr'")
    
    if not is_sentence_transformers_available():
        st.info("安裝語義模型: 請執行 'pip install sentence-transformers'")
        st.warning("注意：語義模型未安裝，系統將使用簡化版語義比對，精度可能較低")

# 側邊欄 - 比對設置
with st.sidebar:
    st.header("比對設置")
    
    # 比對模式
    comparison_mode = st.radio(
        "比對模式",
        ["精確比對", "語意比對", "混合比對"],
        index=2
    )
    
    # 相似度閾值
    similarity_threshold = st.slider(
        "相似度閾值",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="相似度低於此閾值的段落將被標記為不一致"
    )
    
    # 忽略選項
    st.subheader("忽略選項")
    ignore_space = st.checkbox("忽略空格", value=True)
    ignore_punctuation = st.checkbox("忽略標點符號", value=True)
    ignore_case = st.checkbox("忽略大小寫", value=True)
    ignore_newline = st.checkbox("忽略換行", value=True)
    
    # OCR設置
    st.subheader("OCR設置")
    ocr_engine = st.radio(
        "OCR引擎",
        ["自動選擇", "Tesseract", "OpenOCR (EasyOCR)", "Qwen API"],
        index=0,
        help="選擇用於提取PDF文本的OCR引擎"
    )
    
    # 如果選擇Qwen API，顯示API Key輸入框
    if ocr_engine == "Qwen API":
        qwen_api_key = st.text_input("Qwen API Key", type="password", help="輸入您的Qwen API Key")
        st.info("Qwen API提供高精度OCR和表格識別，特別適合複雜排版的PDF")
    else:
        use_ocr = st.checkbox("使用OCR提取", value=True, help="啟用OCR可以提高文本提取質量，但會增加處理時間")
    
    # 表格處理設置
    st.subheader("表格處理")
    table_handling = st.radio(
        "表格處理方式",
        ["輔助人工比對", "嘗試自動比對", "僅標記表格位置"],
        index=0,
        help="選擇系統如何處理表格比對"
    )
    
    # 高級設置
    with st.expander("高級設置"):
        segment_type = st.radio(
            "分段類型",
            ["段落", "句子"],
            index=0
        )
        
        show_all_content = st.checkbox(
            "顯示所有內容",
            value=False,
            help="勾選後顯示所有段落，否則只顯示不一致的段落"
        )
        
        # 差異顯示設置
        st.subheader("差異顯示設置")
        diff_display_mode = st.radio(
            "差異顯示模式",
            ["字符級別", "詞語級別", "行級別"],
            index=0,
            help="選擇如何顯示差異"
        )
        
        # 顏色設置
        st.subheader("顏色設置")
        diff_removed_color = st.color_picker("刪除內容顏色", "#FFCDD2")
        diff_added_color = st.color_picker("添加內容顏色", "#C8E6C9")

# 主要內容區域 - 文件上傳
st.markdown('<h2 class="sub-header">文件上傳</h2>', unsafe_allow_html=True)

# 選擇單文件或多文件模式
file_mode = st.radio(
    "文件上傳模式",
    ["單一Word文件", "多個Word文件（章節）"],
    index=0,
    help="選擇上傳單一Word文件或多個Word文件（作為不同章節）"
)

if file_mode == "單一Word文件":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始Word文件")
        word_files = [st.file_uploader("上傳原始Word文件", type=["docx"])]
    
    with col2:
        st.subheader("美編後PDF文件")
        pdf_file = st.file_uploader("上傳美編後PDF文件", type=["pdf"])
else:
    st.subheader("原始Word文件（多章節）")
    
    # 創建一個容器來存放多個文件上傳器
    uploaded_files = st.file_uploader("上傳多個Word文件", type=["docx"], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"已上傳 {len(uploaded_files)} 個Word文件")
        
        # 顯示上傳的文件
        for i, file in enumerate(uploaded_files):
            st.text(f"章節 {i+1}: {file.name}")
    
    word_files = uploaded_files if uploaded_files else []
    
    st.subheader("美編後PDF文件")
    pdf_file = st.file_uploader("上傳美編後PDF文件", type=["pdf"])

# 文本提取函數
def extract_docx_text(file):
    """從Word文件提取文本和表格"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp.docx")
    
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())
    
    doc = docx.Document(temp_path)
    
    # 提取段落和表格，記錄它們在文檔中的順序
    content_items = []
    
    for item in doc.element.body:
        if item.tag.endswith('p'):  # 段落
            paragraph = docx.text.paragraph.Paragraph(item, doc)
            if paragraph.text.strip():
                content_items.append({
                    'type': 'paragraph',
                    'content': paragraph.text,
                    'context': {
                        'previous_text': '',  # 將在後處理中填充
                        'next_text': ''       # 將在後處理中填充
                    }
                })
        elif item.tag.endswith('tbl'):  # 表格
            table = docx.table.Table(item, doc)
            table_data = []
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    row_data.append(cell.text)
                table_data.append(row_data)
            
            # 提取表格標題（假設表格前的段落可能是標題）
            table_title = ""
            if len(content_items) > 0 and content_items[-1]['type'] == 'paragraph':
                table_title = content_items[-1]['content']
            
            content_items.append({
                'type': 'table',
                'content': table_data,
                'title': table_title,
                'context': {
                    'previous_text': '',  # 將在後處理中填充
                    'next_text': ''       # 將在後處理中填充
                }
            })
    
    # 填充上下文信息
    for i in range(len(content_items)):
        # 前文
        if i > 0:
            content_items[i]['context']['previous_text'] = (
                content_items[i-1]['content'] if content_items[i-1]['type'] == 'paragraph'
                else "表格: " + content_items[i-1]['title']
            )
        
        # 後文
        if i < len(content_items) - 1:
            content_items[i]['context']['next_text'] = (
                content_items[i+1]['content'] if content_items[i+1]['type'] == 'paragraph'
                else "表格: " + content_items[i+1]['title']
            )
    
    # 分離段落和表格
    paragraphs = [item for item in content_items if item['type'] == 'paragraph']
    tables = [item for item in content_items if item['type'] == 'table']
    
    # 清理臨時目錄
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    return {
        "content_items": content_items,
        "paragraphs": paragraphs,
        "tables": tables,
        "filename": file.name
    }

# 增強的PDF提取函數，支持多種OCR引擎
def advanced_pdf_extraction(pdf_file, ocr_engine="自動選擇", qwen_api_key=None, use_ocr=True):
    """
    增強的PDF文本提取函數，支持多種OCR引擎
    """
    # 創建臨時目錄
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp.pdf")
    
    # 保存上傳的文件
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getvalue())
    
    # 如果選擇Qwen API
    if ocr_engine == "Qwen API" and qwen_api_key:
        try:
            st.info("正在使用Qwen API提取文本，這可能需要一些時間...")
            
            # 初始化Qwen OCR
            qwen_ocr = QwenOCR(qwen_api_key)
            
            # 使用PyMuPDF獲取PDF頁數
            import fitz
            doc = fitz.open(temp_path)
            num_pages = len(doc)
            
            # 存儲所有提取的段落和表格
            all_paragraphs = []
            all_tables = []
            
            # 處理每一頁
            for page_num in range(num_pages):
                st.text(f"正在處理第 {page_num+1}/{num_pages} 頁...")
                
                # 將PDF頁面轉換為圖像
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                img_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                pix.save(img_path)
                
                # 使用Qwen API提取文本
                extracted_text = qwen_ocr.extract_text_from_image(img_path)
                
                # 分割段落
                paragraphs = [p.strip() for p in extracted_text.split('\n\n') if p.strip()]
                for para in paragraphs:
                    all_paragraphs.append({
                        'content': para,
                        'page': page_num + 1,
                        'method': 'qwen_api',
                        'confidence': 0.95
                    })
                
                # 提取表格
                tables = qwen_ocr.extract_tables_from_image(img_path)
                if isinstance(tables, list) and tables:
                    for table in tables:
                        all_tables.append({
                            'content': table,
                            'page': page_num + 1,
                            'method': 'qwen_api',
                            'context': {
                                'previous_text': '',
                                'next_text': ''
                            },
                            'title': ''
                        })
            
            # 清理臨時文件
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
            
            return {
                "paragraphs": all_paragraphs,
                "tables": all_tables
            }
            
        except Exception as e:
            st.error(f"Qwen API提取失敗: {e}")
            st.warning("將使用備用方法提取文本...")
    
    # 使用增強的PDF提取函數
    return enhanced_pdf_extraction(pdf_file, use_ocr=use_ocr)

# 文本預處理函數
def preprocess_text(text, ignore_space=True, ignore_punctuation=True, ignore_case=True, ignore_newline=True):
    """文本預處理"""
    if ignore_space:
        text = re.sub(r'\s+', ' ', text)
    
    if ignore_punctuation:
        text = re.sub(r'[.,;:!?，。；：！？]', '', text)
    
    if ignore_case:
        text = text.lower()
    
    if ignore_newline:
        text = text.replace('\n', ' ')
    
    return text.strip()

# 比對函數
def exact_matching(text1, text2, ignore_space=True, ignore_punctuation=True, ignore_case=True, ignore_newline=True):
    """精確比對"""
    # 文本預處理
    processed_text1 = preprocess_text(text1, ignore_space, ignore_punctuation, ignore_case, ignore_newline)
    processed_text2 = preprocess_text(text2, ignore_space, ignore_punctuation, ignore_case, ignore_newline)
    
    # 計算相似度
    matcher = difflib.SequenceMatcher(None, processed_text1, processed_text2)
    similarity = matcher.ratio()
    
    # 生成差異
    diff = list(difflib.ndiff(text1.splitlines(), text2.splitlines()))
    
    return similarity, diff

def semantic_matching(text1, text2, model=None):
    """語意比對（使用Sentence-BERT或簡化版）"""
    if model is not None and is_sentence_transformers_available():
        try:
            # 使用Sentence-BERT計算語義相似度
            embedding1 = model.encode(text1)
            embedding2 = model.encode(text2)
            
            # 計算餘弦相似度
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            st.warning(f"語義模型計算失敗: {e}，使用簡化版語義比對")
    
    # 簡化版語義比對（詞袋模型）
    words1 = set(preprocess_text(text1, True, True,
(Content truncated due to size limit. Use line ranges to read in chunks)