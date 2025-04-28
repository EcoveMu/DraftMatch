import streamlit as st
import os, io
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from text_preview import TextPreview, QwenOCR
from table_processor import TableProcessor
import tempfile
import time
import numpy as np
import difflib

# 添加錯誤處理
try:
    from comparison_algorithm import compare_documents, display_match_results, merge_word_paragraphs
except ImportError as e:
    st.error(f"導入比對算法時出錯: {str(e)}")
    st.info("請確保已安裝所有必要的依賴項。如需本地運行，請執行 'pip install -r requirements.txt'")
    compare_documents = None
    display_match_results = None
    merge_word_paragraphs = None

def main():
    # 設定頁面
    st.set_page_config(page_title="DraftMatch 文件比對系統", page_icon="📊", layout="wide")
    
    # 頁面標題
    st.title("DraftMatch 文件比對系統")
    st.write("上傳 Word 和 PDF 文件，預覽內容並進行智能比對分析。")
    
    # 初始化對象
    text_preview = TextPreview()
    table_processor = TableProcessor()
    
    # 側邊欄設定
    with st.sidebar:
        st.header("功能說明")
        st.info("1. 上傳 Word 與 PDF 文件\n"
                "2. 系統自動提取文字與表格\n"
                "3. 如無法提取文字，自動使用 OCR\n"
                "4. 選擇「文字比對」或「表格比對」標籤\n"
                "5. 點擊相應按鈕開始比對")
        st.markdown("---")
        st.write("提示: 上傳文件後，可以先查看內容預覽，確認文字提取準確性")
        st.markdown("---")
        st.header("API設置")
        
        # 顯示當前API模式
        use_free_api = text_preview.ocr.should_use_free_api()
        current_mode = "免費API模式" if use_free_api else "官方API模式"
        st.success(f"當前OCR設置: {current_mode}")
        
        api_key = st.text_input("Qwen OCR API密鑰（可選）", type="password", 
                               help="若要使用官方API，請輸入您的API密鑰。留空則使用免費API。")
        
        if api_key:
            # 設置環境變數
            os.environ["QWEN_API_KEY"] = api_key
            # 直接設置到OCR物件
            text_preview.ocr.set_api_key(api_key)
            table_processor.qwen_ocr.set_api_key(api_key)
            st.success("API密鑰已設置，將使用官方API")
        else:
            st.info("未提供API密鑰，將使用免費API")
    
    # 設置側邊欄
    st.sidebar.title("設置")
    
    # 設置相似度閾值滑桿
    similarity_threshold = st.sidebar.slider(
        "相似度閾值", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.05
    )
    
    # 設置搜索方向並添加說明
    st.sidebar.markdown("### 搜尋方向")
    st.sidebar.info("""
    **搜尋方向說明**:
    - **PDF → Word**: 從PDF文件中的每個段落開始，尋找Word文件中最匹配的內容。適合檢查PDF中的內容是否存在於Word草稿中。
    - **Word → PDF**: 從Word文件中的每個段落開始，尋找PDF文件中最匹配的內容。適合檢查Word草稿中的內容是否出現在最終PDF中。
    """)
    search_direction = st.sidebar.radio(
        "選擇搜尋方向",
        ["PDF → Word", "Word → PDF"]
    )
    
    # 檔案上傳區
    with st.expander("上傳文件", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("上傳 Word 草稿文件")
            word_file = st.file_uploader("選擇 Word 文件 (.docx)", type=["docx"])
        
        with col2:
            st.subheader("上傳 PDF 定稿文件")
            pdf_file = st.file_uploader("選擇 PDF 文件 (.pdf)", type=["pdf"])
    
    # 只有當兩個文件都上傳後才處理
    if word_file and pdf_file:
        # 顯示載入中
        with st.spinner("正在提取文件內容..."):
            try:
                # OCR模式提示
                ocr_mode = "免費API" if text_preview.ocr.should_use_free_api() else "官方API"
                st.info(f"使用 {ocr_mode} 進行文件處理...")
                
                # 提取文本和表格內容
                word_content = text_preview.extract_word_content(word_file)
                pdf_content = text_preview.extract_pdf_content(pdf_file)
                
                # 提取表格內容
                word_tables = table_processor.extract_word_tables(word_file)
                pdf_tables = table_processor.extract_pdf_tables(pdf_file)
                
                # 創建標籤頁
                tab1, tab2, tab3 = st.tabs(["內容預覽", "表格預覽", "比對結果"])
                
                with tab1:
                    # 顯示文本內容
                    text_preview.display_content(word_content, pdf_content)
                
                with tab2:
                    # 顯示表格內容
                    table_comparison_results = table_processor.display_tables(word_tables, pdf_tables)
                    if table_comparison_results:
                        st.session_state.table_comparison_results = table_comparison_results
                
                with tab3:
                    st.header("比對結果")
                    
                    if compare_documents is None:
                        st.warning("比對功能不可用，請檢查依賴項是否安裝正確")
                    elif st.button("開始比對文件", type="primary"):
                        with st.spinner("正在進行文件比對..."):
                            try:
                                # 根據搜尋方向設置參數
                                pdf_first = search_direction == "PDF → Word"
                                
                                # 進行文本比對
                                comparison_results = compare_documents(
                                    word_content if not pdf_first else pdf_content,
                                    pdf_content if not pdf_first else word_content,
                                    similarity_threshold=similarity_threshold,
                                    matching_method="hybrid"
                                )
                                st.session_state.comparison_results = comparison_results
                                st.session_state.search_direction = search_direction
                                st.success("比對完成！")
                            except Exception as e:
                                st.error(f"比對過程中出錯: {str(e)}")
        
                    # 顯示文本比對結果
                    if hasattr(st.session_state, 'comparison_results') and st.session_state.comparison_results and display_match_results is not None:
                        with st.expander("文本比對結果", expanded=True):
                            try:
                                # 根據搜尋方向顯示結果
                                pdf_first = getattr(st.session_state, 'search_direction', "PDF → Word") == "PDF → Word"
                                
                                display_match_results(
                                    st.session_state.comparison_results,
                                    word_content if not pdf_first else pdf_content,
                                    pdf_content if not pdf_first else word_content,
                                    st
                                )
                            except Exception as e:
                                st.error(f"顯示比對結果時出錯: {str(e)}")
            except Exception as e:
                st.error(f"處理文件時出錯: {str(e)}")
                st.error("如果出現API錯誤，請嘗試不提供API密鑰，系統將自動使用免費API模式。")

if __name__ == "__main__":
    main()
