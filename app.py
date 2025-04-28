import streamlit as st
import os, io
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from text_preview import TextPreview
from table_processor import TableProcessor
from comparison_algorithm import compare_documents, display_match_results, merge_word_paragraphs
import tempfile
import time
import numpy as np
import difflib

def main():
    # 設定頁面
    st.set_page_config(page_title="DraftMatch 文件比對系統", page_icon="📊", layout="wide")
    
    # 頁面標題
    st.title("DraftMatch 文件比對系統")
    st.write("上傳 Word 和 PDF 文件，預覽內容並進行智能比對分析。")
    
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
        api_key = st.text_input("Qwen OCR API密鑰", type="password")
        if api_key:
            text_preview.qwen_ocr.API_KEY = api_key
            table_processor.qwen_ocr.API_KEY = api_key
    
    # 初始化對象
    text_preview = TextPreview()
    table_processor = TableProcessor()
    
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
    
    # 設置搜索方向
    search_direction = st.sidebar.radio(
        "搜索方向",
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
            
            if st.button("開始比對文件", type="primary"):
                with st.spinner("正在進行文件比對..."):
                    # 進行文本比對
                    comparison_results = compare_documents(
                        word_content, 
                        pdf_content, 
                        similarity_threshold=similarity_threshold,
                        matching_method="hybrid"
                    )
                    st.session_state.comparison_results = comparison_results
                    st.success("比對完成！")

            # 顯示文本比對結果
            if hasattr(st.session_state, 'comparison_results') and st.session_state.comparison_results:
                with st.expander("文本比對結果", expanded=True):
                    display_match_results(
                        st.session_state.comparison_results,
                        word_content,
                        pdf_content,
                        st
                    )

if __name__ == "__main__":
    main()
