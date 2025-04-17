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
import re
import difflib

# 導入本地模組
from text_extraction import extract_and_process_documents

# 簡化版的比對算法，不依賴sentence-transformers
def exact_matching(text1, text2, ignore_space=True, ignore_punctuation=True, ignore_case=True):
    """精確比對兩段文本的相似度"""
    if ignore_space:
        text1 = re.sub(r'\s+', ' ', text1)
        text2 = re.sub(r'\s+', ' ', text2)
    if ignore_punctuation:
        text1 = re.sub(r'[.,;:!?，。；：！？]', '', text1)
        text2 = re.sub(r'[.,;:!?，。；：！？]', '', text2)
    if ignore_case:
        text1 = text1.lower()
        text2 = text2.lower()
    matcher = difflib.SequenceMatcher(None, text1, text2)
    similarity = matcher.ratio()
    diff = list(difflib.ndiff(text1.splitlines(), text2.splitlines()))
    return similarity, diff

def compare_documents(doc1, doc2, ignore_options=None, comparison_mode='exact', similarity_threshold=0.6, ai_instance=None):
    """比對兩個文檔的內容"""
    if ignore_options is None:
        ignore_options = {}
    
    # 初始化結果
    paragraph_results = []
    table_results = []
    
    # 比對段落
    for i, para1 in enumerate(doc1["paragraphs"]):
        best_match = None
        best_similarity = 0
        best_index = -1
        best_page = "未找到"
        
        for j, para2 in enumerate(doc2["paragraphs"]):
            # 使用精確比對
            sim, diff = exact_matching(
                para1['content'], para2['content'],
                ignore_space=ignore_options.get("ignore_space", True),
                ignore_punctuation=ignore_options.get("ignore_punctuation", True),
                ignore_case=ignore_options.get("ignore_case", True),
            )
            
            if sim > best_similarity:
                best_similarity = sim
                best_match = para2
                best_index = j
                best_page = para2.get("page", "未找到")
        
        # 判斷是否相似
        is_similar = best_similarity >= similarity_threshold
        
        # 添加結果
        paragraph_results.append({
            "original_index": i,
            "original_text": para1["content"],
            "matched_index": best_index,
            "matched_text": best_match["content"] if best_match else "",
            "matched_page": best_page,
            "exact_similarity": best_similarity,
            "is_similar": is_similar
        })
    
    # 比對表格
    for i, table1 in enumerate(doc1.get("tables", [])):
        best_match = None
        best_similarity = 0
        best_index = -1
        best_page = "未找到"
        
        for j, table2 in enumerate(doc2.get("tables", [])):
            # 計算表格相似度
            table_similarity = calculate_table_similarity(table1["content"], table2["content"])
            
            if table_similarity > best_similarity:
                best_similarity = table_similarity
                best_match = table2
                best_index = j
                best_page = table2.get("page", "未找到")
        
        # 判斷是否相似
        is_similar = best_similarity >= similarity_threshold
        
        # 添加結果
        table_results.append({
            "original_index": i,
            "original_table": table1["content"],
            "matched_index": best_index,
            "matched_table": best_match["content"] if best_match else None,
            "matched_page": best_page,
            "similarity": best_similarity,
            "is_similar": is_similar
        })
    
    # 計算統計信息
    statistics = {
        "total_paragraphs": len(paragraph_results),
        "similar_paragraphs": sum(1 for r in paragraph_results if r["is_similar"]),
        "different_paragraphs": sum(1 for r in paragraph_results if not r["is_similar"]),
        "total_tables": len(table_results),
        "similar_tables": sum(1 for r in table_results if r["is_similar"]),
        "different_tables": sum(1 for r in table_results if not r["is_similar"])
    }
    
    return {
        "paragraph_results": paragraph_results,
        "table_results": table_results,
        "statistics": statistics
    }

def calculate_table_similarity(table1, table2):
    """計算兩個表格的相似度"""
    # 如果表格為空，返回0
    if not table1 or not table2:
        return 0
    
    # 將表格轉換為文本
    text1 = "\n".join([" ".join(row) for row in table1])
    text2 = "\n".join([" ".join(row) for row in table2])
    
    # 使用精確比對
    similarity, _ = exact_matching(text1, text2)
    
    return similarity

def format_diff_html(diff, mode="字符級別"):
    """將差異格式化為HTML"""
    if not diff:
        return ""
    
    if mode == "字符級別":
        # 字符級別差異
        result = []
        for line in diff:
            if line.startswith('- '):
                result.append(f'<span class="diff-char-removed">{line[2:]}</span>')
            elif line.startswith('+ '):
                result.append(f'<span class="diff-char-added">{line[2:]}</span>')
            elif line.startswith('  '):
                result.append(line[2:])
        return "".join(result)
    
    elif mode == "詞語級別":
        # 詞語級別差異
        result = []
        for line in diff:
            if line.startswith('- '):
                result.append(f'<span class="diff-removed">{line[2:]}</span><br>')
            elif line.startswith('+ '):
                result.append(f'<span class="diff-added">{line[2:]}</span><br>')
            elif line.startswith('  '):
                result.append(f'{line[2:]}<br>')
        return "".join(result)
    
    else:  # 行級別
        # 行級別差異
        result = []
        for line in diff:
            if line.startswith('- '):
                result.append(f'<div class="diff-removed">{line[2:]}</div>')
            elif line.startswith('+ '):
                result.append(f'<div class="diff-added">{line[2:]}</div>')
            elif line.startswith('  '):
                result.append(f'<div>{line[2:]}</div>')
        return "".join(result)

def generate_comparison_report(comparison_results, diff_display_mode="字符級別", show_all_content=False):
    """生成比對報告"""
    # 處理段落比對結果
    paragraph_details = []
    for result in comparison_results["paragraph_results"]:
        # 生成差異HTML
        diff_html = ""
        if result["matched_text"]:
            # 使用精確比對
            _, diff = exact_matching(result["original_text"], result["matched_text"])
            diff_html = format_diff_html(diff, diff_display_mode)
        
        # 添加詳細信息
        paragraph_details.append({
            "original_index": result["original_index"],
            "original_text": result["original_text"],
            "matched_text": result["matched_text"],
            "matched_page": result["matched_page"],
            "exact_similarity": result["exact_similarity"],
            "is_similar": result["is_similar"],
            "diff_html": diff_html
        })
    
    # 處理表格比對結果
    table_details = []
    for result in comparison_results["table_results"]:
        table_details.append({
            "original_index": result["original_index"],
            "original_table": result["original_table"],
            "matched_table": result["matched_table"],
            "matched_page": result["matched_page"],
            "similarity": result["similarity"],
            "is_similar": result["is_similar"]
        })
    
    # 計算摘要信息
    total_paragraphs = comparison_results["statistics"]["total_paragraphs"]
    similar_paragraphs = comparison_results["statistics"]["similar_paragraphs"]
    different_paragraphs = comparison_results["statistics"]["different_paragraphs"]
    
    total_tables = comparison_results["statistics"]["total_tables"]
    similar_tables = comparison_results["statistics"]["similar_tables"]
    different_tables = comparison_results["statistics"]["different_tables"]
    
    # 計算相似度百分比
    paragraph_similarity_percentage = (similar_paragraphs / total_paragraphs * 100) if total_paragraphs > 0 else 100
    table_similarity_percentage = (similar_tables / total_tables * 100) if total_tables > 0 else 100
    
    # 生成摘要
    summary = {
        "total_paragraphs": total_paragraphs,
        "similar_paragraphs": similar_paragraphs,
        "different_paragraphs": different_paragraphs,
        "paragraph_similarity_percentage": paragraph_similarity_percentage,
        "total_tables": total_tables,
        "similar_tables": similar_tables,
        "different_tables": different_tables,
        "table_similarity_percentage": table_similarity_percentage
    }
    
    return {
        "summary": summary,
        "paragraph_details": paragraph_details,
        "table_details": table_details
    }

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
        st.session_state.comparison_mode = "精確比對"
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
        st.session_state.use_ocr = False
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
            ["精確比對"],
            index=0
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
        st.session_state.use_ocr = st.checkbox("使用OCR提取PDF文本", value=False)
        
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
        st.warning("示例文件功能需要上傳您自己的文件。請取消勾選「使用示例文件進行演示」選項，然後上傳您的文件。")
        word_file = None
        pdf_file = None
    
    if word_file and pdf_file:
        if st.button("開始比對", key="start_comparison"):
            with st.spinner("正在提取文件內容並進行比對..."):
                # 提取文件內容
                word_data, pdf_data = extract_and_process_documents(
                    word_file, 
                    pdf_file, 
                    st.session_state.use_ocr, 
                    "None",
                    None
                )
                
                st.session_state.word_data = word_data
                st.session_state.pdf_data = pdf_data
                
                # 進行比對
                comparison_results = compare_documents(
                    word_data,
                    pdf_data,
                    st.session_state.ignore_options,
                    st.session_state.comparison_mode,
                    st.session_state.similarity_threshold,
                    None
                )
                
                st.session_state.comparison_results = comparison_results
                
                # 生成比對報告
                comparison_report = generate_comparison_report(
                    comparison_results,
                    st.session_state.diff_display_mode,
                    st.session_state.show_all_content
                )
                
                st.session_state.comparison_report = comparison_report
                
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
                        try:    
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
                                    st.warning(f"頁碼 {detail['matched_page']} 超出範圍")
                        except Exception as e:
                                st.error(f"無法顯示表格: {e}")
            else:
                st.warning("未比對到有效表格，請檢查文件內容是否包含表格。")
    else:
        if not st.session_state.processing_complete:
            st.info("請上傳文件並點擊「開始比對」按鈕。")
        else:
            st.warning("未比對到有效段落，請檢查文件內容是否正確。")

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
