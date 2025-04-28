import streamlit as st
import os, io
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from text_preview import TextPreview, OCRManager
from table_processor import TableProcessor
import tempfile
import time
import numpy as np
import difflib
import sys
import traceback

# 嘗試導入所需庫，避免在部署時出現問題
try:
    # 主要功能模塊
    from comparison_algorithm import compare_documents
    
    # OCR相關庫
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError as e:
    st.error(f"導入錯誤: {str(e)}")
    if "pytesseract" in str(e):
        TESSERACT_AVAILABLE = False
        st.warning("Tesseract OCR 未安裝或不可用，將使用其他OCR引擎")
    traceback.print_exc()
    if "text_preview" in str(e) or "table_processor" in str(e) or "comparison_algorithm" in str(e):
        st.error("核心模塊缺失，應用無法運行")
        st.stop()

# 創建比對函數，用於比較PDF和Word內容
def compare_pdf_word(source_content, target_content, similarity_threshold=0.7):
    """比較源內容和目標內容，返回比對結果
    
    參數:
        source_content: 源內容段落列表
        target_content: 目標內容段落列表
        similarity_threshold: 相似度閾值
        
    返回:
        比對結果列表
    """
    try:
        # 嘗試使用comparison_algorithm中的函數
        from comparison_algorithm import compare_documents
        results = compare_documents(
            word_paragraphs=source_content,
            pdf_paragraphs=target_content,
            similarity_threshold=similarity_threshold
        )
        
        if results and "matches" in results:
            # 格式化結果
            formatted_results = []
            for match in results["matches"]:
                formatted_results.append({
                    "pdf_content": match.get("pdf_text", ""),
                    "pdf_page": match.get("pdf_page", "N/A"),
                    "word_content": match.get("word_text", ""),
                    "similarity": match.get("similarity", 0.0)
                })
            return formatted_results
        return []
    except Exception as e:
        # 如果發生錯誤，使用簡單的備用方法
        st.warning(f"使用核心比對算法時出錯: {str(e)}，使用簡單比對替代")
        formatted_results = []
        
        # 簡單比對邏輯
        for source_para in source_content:
            best_match = None
            best_sim = 0.0
            
            for target_para in target_content:
                source_text = source_para.get("content", "")
                target_text = target_para.get("content", "")
                
                # 使用difflib進行簡單比對
                sim = difflib.SequenceMatcher(None, source_text, target_text).ratio()
                
                if sim > best_sim and sim >= similarity_threshold:
                    best_sim = sim
                    best_match = target_para
            
            # 如果找到匹配，添加到結果
            if best_match:
                formatted_results.append({
                    "pdf_content": best_match.get("content", ""),
                    "pdf_page": best_match.get("page_num", "N/A"),
                    "word_content": source_para.get("content", ""),
                    "similarity": best_sim
                })
                
        return formatted_results

def main():
    # 設定頁面
    st.set_page_config(
        page_title="文檔比對工具",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 頁面標題
    st.title("文檔比對工具")
    st.write("上傳Word和PDF文件進行比對，檢測目錄項和標題")
    
    # 初始化對象
    text_preview = TextPreview()
    table_processor = TableProcessor()
    
    # 側邊欄設定
    with st.sidebar:
        st.header("設置")
        
        # OCR引擎選擇
        st.subheader("OCR引擎設定")
        ocr_manager = OCRManager()
        available_engines = ocr_manager.get_available_engines()
        
        # 轉換引擎名稱為更友好的顯示名稱
        engine_display_names = {
            "tesseract": "Tesseract OCR (本地)",
            "easyocr": "EasyOCR (本地)",
            "qwen": "千問 OCR (API)"
        }
        
        # 創建一個反向映射，從顯示名稱回到引擎名稱
        display_to_engine = {v: k for k, v in engine_display_names.items()}
        
        # 創建可用引擎的顯示名稱列表
        available_display_names = [engine_display_names.get(name, name) for name in available_engines.keys()]
        
        # 預設選擇千問OCR
        default_engine = "qwen" if "qwen" in available_engines else next(iter(available_engines.keys()))
        default_display = engine_display_names.get(default_engine, default_engine)
        
        # 引擎選擇
        selected_display = st.selectbox(
            "選擇OCR引擎",
            options=available_display_names,
            index=available_display_names.index(default_display) if default_display in available_display_names else 0,
            help="選擇用於提取PDF圖像文字的OCR引擎"
        )
        
        # 轉換回引擎名稱
        selected_engine = display_to_engine.get(selected_display, selected_display)
        
        # 千問OCR API設置
        if selected_engine == "qwen":
            st.info("千問OCR支持官方API和免費API，如果不提供API密鑰將自動使用免費API")
            api_key = st.text_input(
                "千問 API 密鑰 (選填)",
                type="password",
                help="輸入您的千問API密鑰，如果不提供將使用免費API",
                key="qwen_api_key"
            )
            
            if api_key:
                # 設置環境變數
                os.environ["QWEN_API_KEY"] = api_key
                # 直接設置到QwenOCR
                qwen_ocr = ocr_manager.get_engine_by_name("qwen")
                if qwen_ocr:
                    qwen_ocr.set_api_key(api_key)
                    st.success("API密鑰已設置")
        
        # 設置OCR引擎
        text_preview.set_ocr_engine(selected_engine)
        
        # 告知用戶當前使用的OCR引擎
        current_engine = text_preview.ocr
        st.info(f"當前使用: {current_engine.name}")
        
        # 顯示千問OCR模式
        if selected_engine == "qwen":
            qwen_mode = "免費API" if current_engine.should_use_free_api() else "官方API"
            st.info(f"千問OCR模式: {qwen_mode}")
        
        # 設置相似度閾值滑桿
        similarity_threshold = st.slider(
            "相似度閾值", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7, 
            step=0.05
        )
        
        # 設置搜索方向並添加說明
        st.subheader("搜尋方向")
        search_direction = st.radio(
            "選擇搜尋方向",
            options=["PDF → Word", "Word → PDF"],
            help="PDF → Word: 在Word中查找PDF的內容\nWord → PDF: 在PDF中查找Word的內容"
        )
        
        # 設置使用OCR
        use_ocr = st.checkbox("使用OCR處理PDF", value=True, help="使用光學字符識別(OCR)處理PDF文件中的圖像")
        
        # 顯示OCR引擎的相關說明
        if use_ocr:
            st.info(f"""
            當前選擇的OCR引擎: **{selected_display}**
            
            - Tesseract: 本地開源OCR引擎，支持多種語言
            - EasyOCR: 本地深度學習OCR引擎，準確度較高
            - 千問OCR: 阿里雲的OCR服務，準確度最高且支持免費API
            """)
    
    # 設置側邊欄
    st.sidebar.title("功能說明")
    st.sidebar.info("1. 上傳 Word 與 PDF 文件\n"
                    "2. 系統自動提取文字與表格\n"
                    "3. 如無法提取文字，自動使用 OCR\n"
                    "4. 選擇「文字比對」或「表格比對」標籤\n"
                    "5. 點擊相應按鈕開始比對")
    st.sidebar.markdown("---")
    st.sidebar.write("提示: 上傳文件後，可以先查看內容預覽，確認文字提取準確性")
    st.sidebar.markdown("---")
    
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
                if selected_engine == "qwen":
                    ocr_mode = "免費API" if text_preview.ocr.should_use_free_api() else "官方API"
                    st.info(f"使用千問OCR {ocr_mode}進行文件處理...")
                else:
                    st.info(f"使用{current_engine.name}進行文件處理...")
                
                # 提取文本和表格內容
                word_content = text_preview.extract_word_content(word_file)
                pdf_content = text_preview.extract_pdf_content(pdf_file)
                
                # 提取表格內容
                word_tables = table_processor.extract_word_tables(word_file)
                pdf_tables = table_processor.extract_pdf_tables(pdf_file.getvalue())
                
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
                    
                    # 根據搜索方向決定比較順序
                    if search_direction == "PDF → Word":
                        # 從PDF搜索Word (PDF是來源，Word是目標)
                        source_content = pdf_content
                        source_name = "PDF"
                        target_content = word_content
                        target_name = "Word"
                    else:  # "Word → PDF"
                        # 從Word搜索PDF (Word是來源，PDF是目標)
                        source_content = word_content
                        source_name = "Word"
                        target_content = pdf_content
                        target_name = "PDF"
                    
                    # 比較PDF和Word內容
                    st.subheader(f"比對結果 ({source_name} → {target_name})")
                    st.info(f"搜尋方向: 從{source_name}內容在{target_name}中尋找匹配")
                    
                    comparison_results = compare_pdf_word(source_content, target_content)
                    
                    # 根據搜索方向修改列名
                    col_names = {
                        "pdf_content": f"{source_name}內容",
                        "pdf_page": f"{source_name}頁碼",
                        "word_content": f"{target_name}內容",
                        "similarity": "相似度"
                    }
                    
                    # 顯示比較結果
                    if comparison_results:
                        df = pd.DataFrame(comparison_results)
                        # 根據搜索方向重命名列
                        df = df.rename(columns=col_names)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("未找到匹配的內容")
            except Exception as e:
                st.error(f"處理文件時出錯: {str(e)}")
                st.error("如果出現API錯誤，請嘗試不提供API密鑰，系統將自動使用免費API模式。")

if __name__ == "__main__":
    main()
