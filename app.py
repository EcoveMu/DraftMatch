import streamlit as st
import os, io
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from text_preview import TextPreview
from table_processor import TableProcessor
from comparison_algorithm import compare_documents, display_match_results
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
        # 保存臨時檔案
        word_path = "temp_word.docx"
        pdf_path = "temp_pdf.pdf"
        
        with open(word_path, "wb") as f:
            f.write(word_file.getvalue())
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getvalue())
        
        # 創建標籤頁
        tab1, tab2, tab3 = st.tabs(["內容預覽", "表格預覽", "比對結果"])
        
        with tab1:
            # 從文件中提取文本內容
            word_content = text_preview.extract_word_content(word_path)
            pdf_content = text_preview.extract_pdf_content(pdf_path)
            
            # 顯示文本內容
            text_preview.display_content(word_content, pdf_content)
        
        with tab2:
            # 提取表格內容
            word_tables = table_processor.extract_word_tables(word_path)
            pdf_tables = table_processor.extract_pdf_tables(pdf_path)
            
            # 顯示表格內容
            table_processor.display_tables(word_tables, pdf_tables)
        
        with tab3:
            st.header("比對結果")
            
            if st.button("開始比對文件", type="primary"):
                with st.spinner("正在進行文件比對..."):
                    # 預處理 Word 段落
                    word_paragraphs = word_content.get("paragraphs", [])
                    pdf_paragraphs = pdf_content.get("paragraphs", [])
                    
                    # 合併 Word 段落
                    merged_word_paragraphs = merge_word_paragraphs(word_paragraphs)
                    
                    # 進行比對
                    comparison_results = compare_pdf_first(pdf_paragraphs, merged_word_paragraphs)
                    
                    # 表格比對
                    if word_tables and pdf_tables:
                        table_results = []
                        for pdf_table in pdf_tables:
                            best_match = None
                            best_score = 0
                            for word_table in word_tables:
                                similarity, _ = table_processor.compare_tables(word_table['table'], pdf_table['table'])
                                if similarity > best_score:
                                    best_score = similarity
                                    best_match = (word_table, similarity)
                            
                            if best_match:
                                table_results.append({
                                    'pdf_table': pdf_table,
                                    'word_table': best_match[0],
                                    'similarity': best_match[1]
                                })
                        
                        st.session_state.table_comparison_results = table_results
                    
                    st.success("比對完成！")

            # 顯示比對結果
            if comparison_results:
                with st.expander("文本比對結果", expanded=True):
                    display_match_results(comparison_results)
                
                # 顯示表格比對結果
                if hasattr(st.session_state, 'table_comparison_results') and st.session_state.table_comparison_results:
                    with st.expander("表格比對結果", expanded=True):
                        for idx, result in enumerate(st.session_state.table_comparison_results):
                            st.subheader(f"表格 {idx+1} 比對結果")
                            st.write(f"相似度: {result['similarity']:.2%}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Word 表格:")
                                st.dataframe(result['word_table']['table'])
                                st.caption(f"位於頁面: {result['word_table'].get('page', 'N/A')}")
                            
                            with col2:
                                st.write("PDF 表格:")
                                st.dataframe(result['pdf_table']['table'])
                                st.caption(f"位於頁面: {result['pdf_table'].get('page', 'N/A')}")
                            
                            # 顯示差異報告
                            diff_report = table_processor.generate_diff_report(
                                result['word_table']['table'], 
                                result['pdf_table']['table']
                            )
                            if diff_report:
                                st.write("差異報告:")
                                st.json(diff_report)
                            
                            st.markdown("---")
        
        # 清理臨時檔案
        try:
            os.remove(word_path)
            os.remove(pdf_path)
        except:
            pass

if __name__ == "__main__":
    main()
