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
from improved_generative_ai import QwenAI
from improved_qwen_ocr import QwenOCR

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
        .diff-warning {
            background-color: #fff3cd;
            color: black;
        }
        .diff-error {
            background-color: #f8d7da;
            color: black;
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
        .table-tab {
            margin-top: 20px;
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
        st.session_state.ocr_engine = "Qwen"  # 默認使用Qwen
    if 'use_ai' not in st.session_state:
        st.session_state.use_ai = True  # 默認啟用AI
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
    if 'use_example_files' not in st.session_state:
        st.session_state.use_example_files = False

# 側邊欄設置
def sidebar_settings():
    with st.sidebar:
        st.title("期刊比對系統")
        
        # 系統設置
        st.header("系統設置")
        
        # 示例文件選項
        st.session_state.use_example_files = st.checkbox("使用示例文件進行演示", value=st.session_state.use_example_files)
        
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
                ["Qwen (免費)", "Qwen (API)", "自定義API"],
                index=0
            )
            
            if st.session_state.ocr_engine == "Qwen (API)" or st.session_state.ocr_engine == "自定義API":
                st.session_state.ocr_api_key = st.text_input("OCR API密鑰", type="password", value=st.session_state.ocr_api_key)
                
                if st.session_state.ocr_engine == "自定義API":
                    st.session_state.custom_ocr_api = st.text_input("自定義OCR API URL", value=st.session_state.custom_ocr_api if st.session_state.custom_ocr_api else "")
        
        # AI設置
        st.subheader("生成式AI設置")
        st.session_state.use_ai = st.checkbox("使用生成式AI增強功能", value=True)
        
        if st.session_state.use_ai:
            ai_model = st.selectbox(
                "AI模型",
                ["Qwen (免費)", "Qwen (API)", "自定義API"],
                index=0
            )
            
            if ai_model == "Qwen (API)":
                st.session_state.custom_ai_api = False
                st.session_state.ai_api_key = st.text_input("AI API密鑰", type="password", value=st.session_state.ai_api_key)
            elif ai_model == "自定義API":
                st.session_state.custom_ai_api = True
                st.session_state.ai_api_key = st.text_input("AI API密鑰", type="password", value=st.session_state.ai_api_key)
                st.session_state.ai_api_url = st.text_input("AI API URL", value=st.session_state.ai_api_url if 'ai_api_url' in st.session_state else "")
            else:
                st.session_state.custom_ai_api = False
                st.session_state.ai_api_key = ""
        
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
        word_file = st.file_uploader("上傳原始Word文件", type=["docx"], key="word_uploader", disabled=st.session_state.use_example_files)
        
        if word_file:
            st.success(f"已上傳: {word_file.name}")
    
    with col2:
        st.subheader("美編後PDF文件")
        pdf_file = st.file_uploader("上傳美編後PDF文件", type=["pdf"], key="pdf_uploader", disabled=st.session_state.use_example_files)
        
        if pdf_file:
            st.success(f"已上傳: {pdf_file.name}")
    
    # 使用示例文件
    if st.session_state.use_example_files:
        word_file_path = "比對素材-原稿.docx"
        pdf_file_path = "比對素材-美編後完稿.pdf"
        
        # 檢查文件是否存在
        if os.path.exists(word_file_path) and os.path.exists(pdf_file_path):
            st.success(f"使用示例文件: {word_file_path} 和 {pdf_file_path}")
            word_file = open(word_file_path, "rb")
            pdf_file = open(pdf_file_path, "rb")
        else:
            st.error("示例文件不存在，請上傳自己的文件或取消勾選「使用示例文件進行演示」選項。")
            word_file = None
            pdf_file = None
    
    if (word_file and pdf_file) or st.session_state.use_example_files:
        if st.button("開始比對", key="start_comparison"):
            with st.spinner("正在提取文件內容並進行比對..."):
                # 初始化OCR引擎
                ocr = None
                if st.session_state.use_ocr:
                    if st.session_state.ocr_engine == "Qwen (API)":
                        ocr = QwenOCR(st.session_state.ocr_api_key)
                    elif st.session_state.ocr_engine == "自定義API":
                        ocr = QwenOCR(st.session_state.ocr_api_key, st.session_state.custom_ocr_api)
                    else:  # Qwen (免費)
                        ocr = QwenOCR()  # 無需API key
                
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
                    elif st.session_state.ocr_engine == "Qwen (API)":
                        ai = QwenAI(st.session_state.ai_api_key)
                    else:
                        ai = QwenAI()  # 無需API key
                
                # 進行比對
                comparison_results = compare_documents(
                    word_data,
                    pdf_data,
                    st.session_state.ignore_options,
                    st.session_state.comparison_mode,
                    st.session_state.similarity_threshold,
                    ai
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
                    if st.session_state.use_ocr and ocr and ocr.is_available():
                        with st.spinner("正在標記PDF中的差異位置..."):
                            # 為每個不同的段落標記位置
                            for result in comparison_results["paragraph_results"]:
                                if not result["is_similar"] and result["matched_page"] != "未找到":
                                    try:
                                        page_num = int(result["matched_page"])
                                        
                                        # 獲取頁面圖像路徑
                                        img_path = os.path.join(temp_dir, f"page_{page_num}.png")
                                        
                                        # 標記文本位置
                                        highlighted_img_path = ocr.highlight_text_in_image(
                                            img_path,
                                            result["matched_text"],
                                            os.path.join(temp_dir, f"highlighted_page_{page_num}_{result['original_index']}.png")
                                        )
                                        
                                        if highlighted_img_path:
                                            # 讀取標記後的圖像
                                            highlighted_img = Image.open(highlighted_img_path)
                                            
                                            # 將圖像轉換為bytes
                                            highlighted_img_byte_arr = io.BytesIO()
                                            highlighted_img.save(highlighted_img_byte_arr, format='PNG')
                                            highlighted_img_byte_arr = highlighted_img_byte_arr.getvalue()
                                            
                                            # 保存到session_state
                                            key = f"{page_num}_{result['original_index']}"
                                            st.session_state.highlighted_images[key] = highlighted_img_byte_arr
                                    except Exception as e:
                                        st.warning(f"標記差異位置時出錯: {str(e)}")
                
                st.session_state.processing_complete = True

# 顯示比對結果
def display_comparison_results():
    if st.session_state.processing_complete and st.session_state.comparison_results and st.session_state.comparison_report:
        st.header("比對結果")
        
        # 顯示摘要信息
        st.subheader("摘要")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "段落相似度",
                f"{st.session_state.comparison_report['summary']['paragraph_similarity_percentage']:.2f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "表格相似度",
                f"{st.session_state.comparison_report['summary']['table_similarity_percentage']:.2f}%",
                delta=None
            )
        
        with col3:
            total_items = st.session_state.comparison_report['summary']['total_paragraphs'] + st.session_state.comparison_report['summary']['total_tables']
            similar_items = st.session_state.comparison_report['summary']['similar_paragraphs'] + st.session_state.comparison_report['summary']['similar_tables']
            
            if total_items > 0:
                overall_similarity = similar_items / total_items * 100
            else:
                overall_similarity = 0
            
            st.metric(
                "整體相似度",
                f"{overall_similarity:.2f}%",
                delta=None
            )
        
        # 顯示詳細統計信息
        with st.expander("詳細統計信息"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**段落統計**")
                st.markdown(f"總段落數: {st.session_state.comparison_report['summary']['total_paragraphs']}")
                st.markdown(f"相似段落數: {st.session_state.comparison_report['summary']['similar_paragraphs']}")
                st.markdown(f"不同段落數: {st.session_state.comparison_report['summary']['different_paragraphs']}")
            
            with col2:
                st.markdown("**表格統計**")
                st.markdown(f"總表格數: {st.session_state.comparison_report['summary']['total_tables']}")
                st.markdown(f"相似表格數: {st.session_state.comparison_report['summary']['similar_tables']}")
                st.markdown(f"不同表格數: {st.session_state.comparison_report['summary']['different_tables']}")
        
        # 顯示AI分析結果
        if st.session_state.use_ai and st.session_state.ai_analysis:
            with st.expander("AI分析", expanded=True):
                st.markdown(st.session_state.ai_analysis)
        
        # 顯示AI摘要報告
        if st.session_state.use_ai and st.session_state.ai_summary_report:
            with st.expander("AI摘要報告", expanded=False):
                st.markdown(st.session_state.ai_summary_report)
        
        # 創建標籤頁
        tab1, tab2 = st.tabs(["段落比對結果", "表格比對結果"])
        
        # 段落比對結果標籤頁
        with tab1:
            # 顯示段落比對結果
            st.subheader("段落比對結果")
            
            # 過濾結果
            if st.session_state.show_all_content:
                paragraph_details = st.session_state.comparison_report["paragraph_details"]
            else:
                paragraph_details = [detail for detail in st.session_state.comparison_report["paragraph_details"] if not detail["is_similar"]]
            
            # 排序結果，將相似度最低的放在前面
            paragraph_details.sort(key=lambda x: x["exact_similarity"])
            
            # 顯示段落比對結果
            for i, detail in enumerate(paragraph_details):
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"**段落 {detail['original_index'] + 1}**")
                        st.markdown(f"相似度: {detail['exact_similarity']:.2f}")
                        st.markdown(f"頁碼: {detail['matched_page']}")
                    
                    with col2:
                        # 顯示原始文本和匹配文本
                        st.markdown("**原始文本:**")
                        st.markdown(detail["original_text"])
                        
                        st.markdown("**美編後文本:**")
                        if detail["matched_text"]:
                            st.markdown(detail["matched_text"])
                        else:
                            st.markdown("未找到匹配文本")
                        
                        # 顯示差異
                        if detail["diff_html"]:
                            st.markdown("**差異:**")
                            st.markdown(detail["diff_html"], unsafe_allow_html=True)
                    
                    # 顯示PDF頁面預覽
                    if detail["matched_page"] != "未找到":
                        try:
                            page_num = int(detail["matched_page"])
                            
                            # 檢查是否有標記後的圖像
                            key = f"{page_num}_{detail['original_index']}"
                            if key in st.session_state.highlighted_images:
                                st.image(
                                    st.session_state.highlighted_images[key],
                                    caption=f"頁面 {page_num} (已標記差異)",
                                    use_column_width=True
                                )
                            # 否則顯示原始頁面
                            elif page_num in st.session_state.pdf_page_images:
                                st.image(
                                    st.session_state.pdf_page_images[page_num],
                                    caption=f"頁面 {page_num}",
                                    use_column_width=True
                                )
                        except:
                            pass
                
                st.markdown("---")
        
        # 表格比對結果標籤頁
        with tab2:
            # 顯示表格比對結果
            st.subheader("表格比對結果")
            
            # 過濾結果
            if "table_details" in st.session_state.comparison_report:
                if st.session_state.show_all_content:
                    table_details = st.session_state.comparison_report["table_details"]
                else:
                    table_details = [detail for detail in st.session_state.comparison_report["table_details"] if not detail["is_similar"]]
                
                # 排序結果，將相似度最低的放在前面
                table_details.sort(key=lambda x: x["similarity"])
                
                # 顯示表格比對結果
                for i, detail in enumerate(table_details):
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            st.markdown(f"**表格 {detail['original_index'] + 1}**")
                            st.markdown(f"相似度: {detail['similarity']:.2f}")
                            st.markdown(f"頁碼: {detail['matched_page']}")
                        
                        with col2:
                            # 顯示原始表格和匹配表格
                            st.markdown("**原始表格:**")
                            if detail["original_table"]:
                                df1 = pd.DataFrame(detail["original_table"])
                                st.dataframe(df1)
                            else:
                                st.markdown("無表格數據")
                            
                            st.markdown("**美編後表格:**")
                            if detail["matched_table"]:
                                df2 = pd.DataFrame(detail["matched_table"])
                                st.dataframe(df2)
                            else:
                                st.markdown("未找到匹配表格")
                            
                            # 顯示差異
                            if detail["diff_html"]:
                                st.markdown("**差異:**")
                                st.markdown(detail["diff_html"], unsafe_allow_html=True)
                        
                        # 顯示PDF頁面預覽
                        if detail["matched_page"] != "未找到":
                            try:
                                page_num = int(detail["matched_page"])
                                if page_num in st.session_state.pdf_page_images:
                                    st.image(
                                        st.session_state.pdf_page_images[page_num],
                                        caption=f"頁面 {page_num}",
                                        use_column_width=True
                                    )
                            except:
                                pass
                    
                    st.markdown("---")
            else:
                st.info("沒有表格比對結果")

# 主函數
def main():
    # 加載CSS
    load_css()
    
    # 初始化會話狀態
    init_session_state()
    
    # 側邊欄設置
    sidebar_settings()
    
    # 文件上傳區域
    file_upload_section()
    
    # 顯示比對結果
    display_comparison_results()

if __name__ == "__main__":
    main()
