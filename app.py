import streamlit as st
import os, io
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from text_preview import TextPreview
from table_processor import TableProcessor
from comparison_algorithm import compare_pdf_first
from qwen_ocr import QwenOCR
from easyocr_wrapper import EasyOCR
from tesseract_wrapper import TesseractOCR
import html

def initialize_ocr():
    """根據用戶選擇初始化OCR實例"""
    ocr_instance = None
    if st.session_state.use_ocr:
        try:
            if st.session_state.ocr_engine == "qwen_builtin":
                ocr_instance = QwenOCR()  # 內建免費API
                if ocr_instance.is_available():
                    st.sidebar.success("Qwen OCR 初始化成功")
                else:
                    st.sidebar.error("Qwen OCR 初始化失敗")
            elif st.session_state.ocr_engine == "easyocr":
                ocr_instance = EasyOCR()  # 使用EasyOCR
                if ocr_instance.is_available():
                    st.sidebar.success("EasyOCR 初始化成功")
                else:
                    st.sidebar.error("EasyOCR 初始化失敗，請確保已安裝相關套件")
            elif st.session_state.ocr_engine == "tesseract":
                ocr_instance = TesseractOCR()  # 使用Tesseract
                if ocr_instance.is_available():
                    st.sidebar.success("Tesseract OCR 初始化成功")
                else:
                    st.sidebar.error("Tesseract OCR 初始化失敗，請確保已安裝Tesseract和pytesseract")
            elif st.session_state.ocr_engine == "ocr_custom" and st.session_state.ocr_api_key:
                ocr_instance = QwenOCR(api_key=st.session_state.ocr_api_key)
                if ocr_instance.is_available():
                    st.sidebar.success("自定義 OCR API 初始化成功")
                else:
                    st.sidebar.error("自定義 OCR API 初始化失敗，請檢查API Key")
        except Exception as e:
            st.sidebar.error(f"OCR 初始化錯誤: {str(e)}")
    return ocr_instance

def main():
    # 設定頁面
    st.set_page_config(page_title="文件比對系統", layout="wide")
    
    # 注入自定義 CSS
    st.markdown("""
    <style>
    /* 整體頁面樣式調整 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* 不再需要修復 Streamlit 滾動容器 */
    /* .main, .block-container, [data-testid="stAppViewContainer"], 
    .stApp, section[data-testid="stSidebar"] {
        overflow: visible !important;
    } */
    
    /* 移除舊的 Word 原稿固定顯示相關樣式 */
    
    /* 不再需要留出頂部空間 */
    /* .match-content-container {
        margin-top: calc(33vh + 100px);
        position: relative;
    } */
    
    /* 匹配區塊樣式 */
    .match-block {
        position: relative;
        margin-bottom: 25px;
        border: 1px solid #e9e9e9;
        border-radius: 5px;
        padding: 15px;
        background-color: white;
    }
    
    /* 移除滾動監視點相關樣式 */
    
    /* 新增 Word 原稿滾動區域樣式 */
    .word-scroll-panel {
        background-color: #f5f7fa;
        padding: 8px;
        border-left: 3px solid #4e8cff;
        max-height: 150px;
        overflow-y: auto;
        margin-bottom: 10px;
        font-family: monospace;
        white-space: pre-wrap;
        word-break: break-word;
    }
    
    /* 表格在滾動區域中的樣式 */
    .word-scroll-panel table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85em;
    }
    
    .word-scroll-panel table th {
        background-color: #f1f3f5;
        padding: 5px;
        border: 1px solid #dee2e6;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    
    .word-scroll-panel table td {
        padding: 4px;
        border: 1px solid #dee2e6;
    }
    
    .word-scroll-panel table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    
    /* 修改差異標示的顯示樣式 */
    .diff-content {
        margin-top: 15px;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #f0f0f0;
        background-color: white;
    }
    
    /* 新增 Word 原稿區域樣式 */
    .word-original-content, .word-original-table {
        background-color: #f8f9fa;
        padding: 12px 15px;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* 調整文本區域樣式 */
    .stTextArea textarea {
        background-color: #fffdf7;
        font-family: 'Courier New', monospace;
        border: 1px solid #ddd;
    }
    
    /* 調整收合區域樣式 */
    .streamlit-expander {
        border-radius: 5px;
        border: 1px solid #f0f0f0;
        margin-bottom: 1rem;
    }
    
    /* 調整匹配區域樣式 */
    .streamlit-expanderHeader {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 5px 10px;
        margin-bottom: 10px;
    }
    
    /* 匹配內容的容器 */
    .streamlit-expanderContent {
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 0 0 5px 5px;
    }
    
    /* 自定義按鈕樣式 - 深色霧藍漸層效果 */
    .stButton > button[data-testid="baseButton-primary"] {
        background-image: linear-gradient(to right, #2c3e50, #4a6790) !important;
        background-color: #2c3e50 !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }
    
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-image: linear-gradient(to right, #34495e, #546e8c) !important;
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button[data-testid="baseButton-primary"]:active {
        background-image: linear-gradient(to right, #243342, #3d5978) !important;
        box-shadow: 0 3px 6px rgba(50, 50, 93, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        transform: translateY(1px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 頁面標題
    st.title("文件比對系統")
    st.write("本系統用於比對 Word 原稿與 PDF 完稿，支援文字與表格比對，並可辨識無文字內容的文件。")
    
    # 初始化會話狀態
    if 'use_ocr' not in st.session_state:
        st.session_state.use_ocr = True
    if 'ocr_engine' not in st.session_state:
        st.session_state.ocr_engine = "qwen_builtin"
    if 'ocr_api_key' not in st.session_state:
        st.session_state.ocr_api_key = ""
    if 'use_enhanced_diff' not in st.session_state:
        st.session_state.use_enhanced_diff = True
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.85
    
    # 側邊欄設定
    with st.sidebar:
        st.header("功能說明")
        st.info("1. 上傳 Word 與 PDF 文件\n"
                "2. 系統自動提取文字與表格\n"
                "3. 如無法提取文字，自動使用 OCR\n"
                "4. 選擇「文字比對」或「表格比對」標籤\n"
                "5. 點擊相應按鈕開始比對")
        
        # OCR 設定
        st.divider()
        st.subheader("🔍 OCR 設定")
        st.session_state.use_ocr = st.checkbox("啟用 OCR", value=st.session_state.use_ocr)
        if st.session_state.use_ocr:
            ocr_labels = {
                "Qwen（內建）": "qwen_builtin",
                "EasyOCR（內建）": "easyocr",
                "Tesseract（內建）": "tesseract",
                "自定義 OCR API": "ocr_custom",
            }
            current = next(k for k, v in ocr_labels.items() if v == st.session_state.ocr_engine)
            ocr_label = st.radio("OCR 引擎", list(ocr_labels.keys()), horizontal=True, index=list(ocr_labels.keys()).index(current))
            st.session_state.ocr_engine = ocr_labels[ocr_label]
            
            # 顯示所選OCR引擎的說明
            ocr_descriptions = {
                "qwen_builtin": "**Qwen OCR**：使用通義千問API的視覺模型進行OCR，支援表格識別。無需額外安裝。",
                "easyocr": "**EasyOCR**：開源OCR工具，支援多語言，需要安裝額外相依套件。表格識別功能有限。",
                "tesseract": "**Tesseract**：最知名的開源OCR引擎，需要安裝Tesseract和pytesseract套件。",
                "ocr_custom": "**自定義API**：使用您提供的API密鑰調用Qwen API，如有商業需求請使用此選項。"
            }
            st.markdown(ocr_descriptions[st.session_state.ocr_engine])
            
            if st.session_state.ocr_engine == "ocr_custom":
                st.session_state.ocr_api_key = st.text_input("OCR API Key", type="password", value=st.session_state.ocr_api_key)
        
        # 差異標示設定
        st.divider()
        st.subheader("🔄 比對設定")
        st.session_state.use_enhanced_diff = st.checkbox("使用增強型差異標示", 
                                                          value=st.session_state.use_enhanced_diff,
                                                          help="啟用後，以PDF內容為主，灰色表示相同內容，紅色表示不同內容")
        
        if st.session_state.use_enhanced_diff:
            st.session_state.similarity_threshold = st.slider(
                "相似度閾值", 
                min_value=0.7, 
                max_value=0.95, 
                value=st.session_state.similarity_threshold,
                step=0.05,
                help="調整文本相似度的判斷標準，值越高要求越嚴格，相同內容越少"
            )
    
    # 檔案上傳區
    col1, col2 = st.columns(2)
    with col1:
        word_file = st.file_uploader("上傳 Word 原稿", type=['docx'], key="word_uploader")
    with col2:
        pdf_file = st.file_uploader("上傳 PDF 完稿", type=['pdf'], key="pdf_uploader")
    
    # 只有當兩個文件都上傳後才處理
    if word_file and pdf_file:
        # 保存臨時檔案
        word_path = "temp_word.docx"
        pdf_path = "temp_pdf.pdf"
        
        with open(word_path, "wb") as f:
            f.write(word_file.getvalue())
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getvalue())
        
        # 初始化OCR實例
        ocr_instance = initialize_ocr()
        
        # 初始化處理器，並傳入OCR實例
        text_previewer = TextPreview(ocr_instance=ocr_instance)
        table_processor = TableProcessor(ocr_instance=ocr_instance)
        
        # 提取內容
        with st.spinner("正在提取文件內容..."):
            try:
                word_content = text_previewer.extract_word_content(word_path)
                pdf_content = text_previewer.extract_pdf_content(pdf_path)
                
                # 提取表格 (使用 try-except 處理可能的錯誤)
                try:
                    word_tables = table_processor.extract_word_tables(word_path)
                except Exception as e:
                    st.warning(f"提取 Word 表格時發生錯誤: {str(e)}")
                    word_tables = []
                
                try:
                    pdf_tables = table_processor.extract_pdf_tables(pdf_path)
                except Exception as e:
                    st.warning(f"提取 PDF 表格時發生錯誤: {str(e)}")
                    pdf_tables = []
            except Exception as e:
                st.error(f"提取內容時發生錯誤: {str(e)}")
                # 清理臨時檔案
                try:
                    os.remove(word_path)
                    os.remove(pdf_path)
                except:
                    pass
                return
        
        # 頁籤區域
        tab1, tab2 = st.tabs(["文字比對", "表格比對"])
        
        # 文字比對頁籤
        with tab1:
            # 顯示文字內容預覽
            try:
                # 添加收合選項
                preview_expander = st.expander("內容預覽（點擊展開或收合）", expanded=True)
                with preview_expander:
                    need_refresh = text_previewer.display_content(word_content, pdf_content)
                    
                    # 如果需要重新提取
                    if need_refresh:
                        with st.spinner("重新提取內容..."):
                            pdf_content = text_previewer.extract_pdf_content(pdf_path)
                            text_previewer.display_content(word_content, pdf_content)
            except Exception as e:
                st.error(f"顯示內容時發生錯誤: {str(e)}")
            
            # 上方比對按鈕
            st.write("---")
            top_col1, top_col2, top_col3 = st.columns([1, 2, 1])
            with top_col2:
                top_compare_button = st.button("開始文字比對", key="start_text_comparison_top", 
                                           use_container_width=True, 
                                           type="primary")
            st.write("---")
            
            # 檢查上方按鈕是否被點擊
            start_comparison = False
            if top_compare_button:
                start_comparison = True
            
            # 如果任一按鈕被點擊，執行比對
            if start_comparison:
                try:
                    with st.spinner("正在進行文字比對..."):
                        # 準備資料
                        word_data = {'paragraphs': word_content}
                        pdf_data = {'paragraphs': pdf_content}
                        
                        # 執行比對
                        results = compare_pdf_first(word_data, pdf_data, ocr_instance=ocr_instance)
                    
                    # 顯示比對結果
                    st.subheader("文字比對結果")
                    
                    # 顯示統計
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("總PDF頁數", results['statistics']['total_pdf'])
                    with col2:
                        st.metric("匹配段落", results['statistics']['matched'])
                    with col3:
                        st.metric("未匹配段落", results['statistics']['unmatched_pdf'] + results['statistics']['unmatched_word'])
                    
                    # 詳細結果
                    if results['matches']:
                        for i, match in enumerate(results['matches']):
                            # 創建匹配區塊的容器
                            st.markdown(f'<div class="match-block">', unsafe_allow_html=True)
                            
                            # 移除滾動監視點，不再需要
                            
                            with st.expander(f"匹配 #{i+1} (相似度: {match['similarity']:.2%})", expanded=True):
                                st.write(f"PDF 頁碼: {match['pdf_page']}")
                                
                                # 修改Word原稿顯示方式，使其可調整大小
                                st.markdown('<div style="font-size: 0.9rem; font-weight: 500; margin-bottom: 3px; color: #1e88e5;">Word 原稿:</div>', unsafe_allow_html=True)
                                st.markdown(f'''
                                <div class="word-scroll-panel" style="resize: both; overflow: auto; min-height: 100px; max-height: 300px;">
                                    {html.escape(match["word_text"])}
                                </div>
                                ''', unsafe_allow_html=True)
                                
                                # 根據設置顯示差異標示
                                st.markdown('<div class="diff-content">', unsafe_allow_html=True)
                                if st.session_state.use_enhanced_diff:
                                    st.markdown("**PDF 內容差異標示** (灰色：相同內容，紅色：不同內容)")
                                    st.markdown(match['enhanced_diff_html'], unsafe_allow_html=True)
                                else:
                                    st.markdown("**標準差異標示**")
                                    st.markdown(match['diff_html'], unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # 差異摘要
                                if match.get('diff_summary'):
                                    st.markdown("**句子層級差異:**")
                                    for j, diff in enumerate(match['diff_summary']):
                                        st.write(f"- 相似度: {diff['similarity']:.2%}")
                                        st.write(f"  Word: {diff['word_sentence']}")
                                        st.write(f"  PDF: {diff['pdf_sentence']}")
                            
                            # 關閉匹配區塊容器
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("沒有找到匹配的內容")
                except Exception as e:
                    st.error(f"文字比對時發生錯誤: {str(e)}")
        
        # 表格比對頁籤
        with tab2:
            # 顯示表格內容預覽
            try:
                if word_tables and pdf_tables:
                    # 添加收合選項
                    table_preview_expander = st.expander("表格預覽（點擊展開或收合）", expanded=True)
                    with table_preview_expander:
                        word_tables, pdf_tables = table_processor.display_tables(word_tables, pdf_tables)
                else:
                    st.warning("沒有找到足夠的表格內容進行比對。請確保文件中包含表格。")
                
                # 如果有表格內容，則顯示上方和底部的比對按鈕
                if word_tables and pdf_tables:
                    # 上方表格比對按鈕
                    st.write("---")
                    table_top_col1, table_top_col2, table_top_col3 = st.columns([1, 2, 1])
                    with table_top_col2:
                        table_top_compare_button = st.button("開始表格比對", key="start_table_comparison_top", 
                                                use_container_width=True, 
                                                type="primary")
                    st.write("---")
                    
                    # 檢查上方按鈕是否被點擊
                    start_table_comparison = False
                    if table_top_compare_button:
                        start_table_comparison = True
                    
                    # 如果任一按鈕被點擊，執行表格比對
                    if start_table_comparison:
                        try:
                            with st.spinner("正在進行表格比對..."):
                                # 執行表格比對
                                table_results = []
                                for word_table in word_tables:
                                    best_match = None
                                    best_similarity = 0.0
                                    
                                    for pdf_table in pdf_tables:
                                        result = table_processor.compare_tables(word_table, pdf_table)
                                        if result['similarity'] > best_similarity:
                                            best_similarity = result['similarity']
                                            best_match = result
                                    
                                    if best_match:
                                        table_results.append(best_match)
                        
                            # 顯示比對結果
                            st.subheader("表格比對結果")
                            
                            # 移除全局的 Word 表格固定顯示區域
                            
                            if table_results:
                                for i, result in enumerate(table_results):
                                    # 創建表格匹配區塊的容器
                                    st.markdown(f'<div class="match-block">', unsafe_allow_html=True)
                                    
                                    # 移除滾動監視點
                                    
                                    with st.expander(f"表格匹配 #{i+1} (相似度: {result['similarity']:.2%})", expanded=True):
                                        st.write(f"Word 表格 {result['word_table']['index'] + 1} 與 PDF 表格 {result['pdf_table']['index'] + 1}")
                                        
                                        # 創建 Word 表格的 HTML 表示並直接顯示
                                        word_table_df = pd.DataFrame(result['word_table']['data'])
                                        word_table_html = word_table_df.to_html(index=False, classes='table table-bordered table-sm')
                                        
                                        # 使用更緊湊的樣式顯示 Word 表格
                                        st.markdown('<div style="font-size: 0.9rem; font-weight: 500; margin-bottom: 3px; color: #1e88e5;">Word 表格:</div>', unsafe_allow_html=True)
                                        st.markdown(f'''
                                        <div class="word-scroll-panel" style="resize: both; overflow: auto; min-height: 100px; max-height: 300px;">
                                            {word_table_html}
                                        </div>
                                        ''', unsafe_allow_html=True)
                                        
                                        # 顯示 PDF 表格
                                        st.markdown("**PDF 表格**")
                                        st.dataframe(pd.DataFrame(result['pdf_table']['data']), use_container_width=True)
                                        
                                        # 差異報告
                                        if result['diff_report']:
                                            st.markdown("**單元格差異:**")
                                            diff_df = []
                                            for diff in result['diff_report']:
                                                diff_row = {
                                                    "位置": f"({diff['row']}, {diff['col']})",
                                                    "Word內容": diff['word_value'],
                                                    "PDF內容": diff['pdf_value'],
                                                    "差異類型": "修改" if diff['type'] == 'modified' else "新增" if diff['type'] == 'added' else "刪除"
                                                }
                                                diff_df.append(diff_row)
                                            
                                            st.dataframe(pd.DataFrame(diff_df), use_container_width=True)
                                    
                                    # 關閉表格匹配區塊容器
                                    st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.warning("沒有找到匹配的表格")
                        except Exception as e:
                            st.error(f"表格比對時發生錯誤: {str(e)}")
            except Exception as e:
                st.error(f"顯示表格內容時發生錯誤: {str(e)}")
        
        # 清理臨時檔案
        try:
            os.remove(word_path)
            os.remove(pdf_path)
        except:
            pass

if __name__ == "__main__":
    main()
