import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
import time
import json
import io
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
from text_extraction import extract_and_process_documents
from comparison_algorithm import compare_documents, generate_comparison_report, format_diff_html
from generative_ai import QwenAI
from qwen_ocr import QwenOCR

# 設置頁面配置
st.set_page_config(
    page_title="期刊比對系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
def load_css():
    css = """
    <style>
        .diff-removed {
            background-color: #ffcccc;
            text-decoration: line-through;
            color: black;
        }
        .diff-added {
            background-color: #ccffcc;
            color: black;
        }
        .diff-char-removed {
            background-color: #ffcccc;
            text-decoration: line-through;
            display: inline;
            color: black;
        }
        .diff-char-added {
            background-color: #ccffcc;
            display: inline;
            color: black;
        }
        .comparison-result {
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            color: black;
        }
        .similar {
            border-left: 5px solid green;
        }
        .different {
            border-left: 5px solid red;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: black;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e6f0ff;
            border-bottom: 2px solid #4c83ff;
        }
        .highlight {
            background-color: yellow;
            color: black;
        }
        .table-container {
            overflow-x: auto;
        }
        .table-container table {
            width: 100%;
            border-collapse: collapse;
        }
        .table-container th, .table-container td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            color: black;
        }
        .table-container th {
            background-color: #f2f2f2;
        }
        .table-container tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .summary-card {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            color: black;
        }
        .summary-card h3 {
            margin-top: 0;
            color: #333;
        }
        .metric-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .metric-box {
            background-color: white;
            border-radius: 5px;
            padding: 10px;
            margin: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            flex: 1;
            min-width: 120px;
            color: black;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 5px 0;
            color: black;
        }
        .metric-label {
            font-size: 14px;
            color: #333;
        }
        .ai-analysis {
            background-color: #f0f7ff;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            border-left: 5px solid #4c83ff;
            color: black;
        }
        .pdf-preview {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            background-color: white;
        }
        .pdf-preview img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        .pdf-preview-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown li {
            color: black !important;
        }
        .stText {
            color: black !important;
        }
        .stTextInput > div > div > input {
            color: black !important;
        }
        .stSelectbox > div > div > div {
            color: black !important;
        }
        .stSlider > div > div > div {
            color: black !important;
        }
        .stCheckbox > div > div > label {
            color: black !important;
        }
        .stExpander > div > div > div > div > p {
            color: black !important;
        }
        .stExpander > div > div > div > div > div > p {
            color: black !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 初始化會話狀態
def init_session_state():
    if 'word_data' not in st.session_state:
        st.session_state.word_data = None
    if 'pdf_data' not in st.session_state:
        st.session_state.pdf_data = None
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'comparison_report' not in st.session_state:
        st.session_state.comparison_report = None
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = None
    if 'ai_summary_report' not in st.session_state:
        st.session_state.ai_summary_report = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'current_paragraph_index' not in st.session_state:
        st.session_state.current_paragraph_index = 0
    if 'current_table_index' not in st.session_state:
        st.session_state.current_table_index = 0
    if 'show_all_content' not in st.session_state:
        st.session_state.show_all_content = False
    if 'diff_display_mode' not in st.session_state:
        st.session_state.diff_display_mode = "字符級別"
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = "混合比對"
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.8
    if 'ignore_options' not in st.session_state:
        st.session_state.ignore_options = {
            "ignore_space": True,
            "ignore_punctuation": True,
            "ignore_case": True,
            "ignore_newline": True
        }
    if 'use_ocr' not in st.session_state:
        st.session_state.use_ocr = True
    if 'ocr_engine' not in st.session_state:
        st.session_state.ocr_engine = "tesseract"
    if 'use_ai' not in st.session_state:
        st.session_state.use_ai = False
    if 'ai_api_key' not in st.session_state:
        st.session_state.ai_api_key = ""
    if 'ocr_api_key' not in st.session_state:
        st.session_state.ocr_api_key = ""
    if 'custom_ocr_api' not in st.session_state:
        st.session_state.custom_ocr_api = False
    if 'custom_ai_api' not in st.session_state:
        st.session_state.custom_ai_api = False
    if 'pdf_page_images' not in st.session_state:
        st.session_state.pdf_page_images = {}
    if 'highlighted_images' not in st.session_state:
        st.session_state.highlighted_images = {}

# 側邊欄設置
def sidebar_settings():
    with st.sidebar:
        st.title("期刊比對系統")
        
        # 系統設置
        st.header("系統設置")
        
        # 比對設置
        st.subheader("比對設置")
        st.session_state.comparison_mode = st.selectbox(
            "比對模式",
            ["精確比對", "語意比對", "混合比對"],
            index=2
        )
        
        st.session_state.similarity_threshold = st.slider(
            "相似度閾值",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
        
        # 忽略選項
        st.subheader("忽略選項")
        st.session_state.ignore_options["ignore_space"] = st.checkbox("忽略空格", value=True)
        st.session_state.ignore_options["ignore_punctuation"] = st.checkbox("忽略標點符號", value=True)
        st.session_state.ignore_options["ignore_case"] = st.checkbox("忽略大小寫", value=True)
        st.session_state.ignore_options["ignore_newline"] = st.checkbox("忽略換行", value=True)
        
        # OCR設置
        st.subheader("OCR設置")
        st.session_state.use_ocr = st.checkbox("使用OCR提取PDF文本", value=True)
        
        if st.session_state.use_ocr:
            st.session_state.ocr_engine = st.selectbox(
                "OCR引擎",
                ["tesseract", "Qwen", "自定義API"],
                index=0
            )
            
            if st.session_state.ocr_engine == "Qwen" or st.session_state.ocr_engine == "自定義API":
                st.session_state.ocr_api_key = st.text_input("OCR API密鑰", type="password", value=st.session_state.ocr_api_key)
                
                if st.session_state.ocr_engine == "自定義API":
                    st.session_state.custom_ocr_api = st.text_input("自定義OCR API URL", value=st.session_state.custom_ocr_api if st.session_state.custom_ocr_api else "")
        
        # AI設置
        st.subheader("生成式AI設置")
        st.session_state.use_ai = st.checkbox("使用生成式AI增強功能", value=False)
        
        if st.session_state.use_ai:
            ai_model = st.selectbox(
                "AI模型",
                ["Qwen (免費)", "自定義API"],
                index=0
            )
            
            if ai_model == "自定義API":
                st.session_state.custom_ai_api = True
                st.session_state.ai_api_key = st.text_input("AI API密鑰", type="password", value=st.session_state.ai_api_key)
                st.session_state.ai_api_url = st.text_input("AI API URL", value=st.session_state.ai_api_url if 'ai_api_url' in st.session_state else "")
            else:
                st.session_state.custom_ai_api = False
        
        # 顯示設置
        st.subheader("顯示設置")
        st.session_state.diff_display_mode = st.selectbox(
            "差異顯示模式",
            ["字符級別", "詞語級別", "行級別"],
            index=0
        )
        
        st.session_state.show_all_content = st.checkbox("顯示所有內容", value=False)
        
        # 系統資訊
        st.subheader("系統資訊")
        st.info("本系統用於比對原始Word文件與美編後PDF文件的內容差異，幫助校對人員快速找出不一致之處。")

# 文件上傳區域
def file_upload_section():
    st.header("文件上傳")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始Word文件")
        word_file = st.file_uploader("上傳原始Word文件", type=["docx"], key="word_uploader")
        
        if word_file:
            st.success(f"已上傳: {word_file.name}")
    
    with col2:
        st.subheader("美編後PDF文件")
        pdf_file = st.file_uploader("上傳美編後PDF文件", type=["pdf"], key="pdf_uploader")
        
        if pdf_file:
            st.success(f"已上傳: {pdf_file.name}")
    
    if word_file and pdf_file:
        if st.button("開始比對", key="start_comparison"):
            with st.spinner("正在提取文件內容並進行比對..."):
                # 初始化OCR引擎
                ocr = None
                if st.session_state.use_ocr:
                    if st.session_state.ocr_engine == "Qwen":
                        ocr = QwenOCR(st.session_state.ocr_api_key)
                    elif st.session_state.ocr_engine == "自定義API":
                        ocr = QwenOCR(st.session_state.ocr_api_key, st.session_state.custom_ocr_api)
                
                # 提取文件內容
                word_data, pdf_data = extract_and_process_documents(
                    word_file, 
                    pdf_file, 
                    st.session_state.use_ocr, 
                    st.session_state.ocr_engine,
                    ocr
                )
                
                st.session_state.word_data = word_data
                st.session_state.pdf_data = pdf_data
                
                # 初始化AI
                ai = None
                if st.session_state.use_ai:
                    if st.session_state.custom_ai_api:
                        ai = QwenAI(st.session_state.ai_api_key, st.session_state.ai_api_url)
                    else:
                        ai = QwenAI()
                
                # 進行比對
                comparison_results = compare_documents(
                    word_data,
                    pdf_data,
                    st.session_state.ignore_options,
                    st.session_state.comparison_mode,
                    st.session_state.similarity_threshold,
                    ai if st.session_state.comparison_mode == "語意比對" else None
                )
                
                st.session_state.comparison_results = comparison_results
                
                # 生成比對報告
                comparison_report = generate_comparison_report(
                    comparison_results,
                    st.session_state.diff_display_mode,
                    st.session_state.show_all_content
                )
                
                st.session_state.comparison_report = comparison_report
                
                # 使用AI分析比對結果
                if st.session_state.use_ai and ai and ai.is_available():
                    with st.spinner("正在使用AI分析比對結果..."):
                        # 獲取原始文本和編輯後文本的樣本
                        original_sample = "\n".join([p['content'] for p in word_data['paragraphs'][:5]])
                        edited_sample = "\n".join([p['content'] for p in pdf_data['paragraphs'][:5] if 'content' in p])
                        
                        # 分析比對結果
                        ai_analysis = ai.analyze_comparison_results(
                            original_sample,
                            edited_sample,
                            comparison_results
                        )
                        
                        st.session_state.ai_analysis = ai_analysis
                        
                        # 生成摘要報告
                        ai_summary_report = ai.generate_summary_report(comparison_results)
                        st.session_state.ai_summary_report = ai_summary_report
                
                # 提取PDF頁面圖像
                with st.spinner("正在提取PDF頁面圖像..."):
                    # 保存上傳的PDF文件到臨時文件
                    temp_dir = tempfile.mkdtemp()
                    temp_pdf_path = os.path.join(temp_dir, "temp.pdf")
                    
                    with open(temp_pdf_path, "wb") as f:
                        f.write(pdf_file.getvalue())
                    
                    # 打開PDF文件
                    pdf_doc = fitz.open(temp_pdf_path)
                    
                    # 提取每一頁的圖像
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc[page_num]
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        
                        # 將圖像保存到臨時文件
                        img_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                        pix.save(img_path)
                        
                        # 將圖像讀取為PIL圖像
                        img = Image.open(img_path)
                        
                        # 將圖像轉換為bytes
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        # 保存到session_state
                        st.session_state.pdf_page_images[page_num+1] = img_byte_arr
                    
                    # 關閉PDF文件
                    pdf_doc.close()
                    
                    # 如果使用OCR，為不同的段落標記位置
                    if st.session_state.use_ocr and (st.session_state.ocr_engine == "Qwen" or st.session_state.ocr_engine == "自定義API") and ocr:
                     pass  # ← 暫時不執行任何動作

(Content truncated due to size limit. Use line ranges to read in chunks)
