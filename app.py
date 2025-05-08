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
                need_refresh = text_previewer.display_content(word_content, pdf_content)
                
                # 如果需要重新提取
                if need_refresh:
                    with st.spinner("重新提取內容..."):
                        pdf_content = text_previewer.extract_pdf_content(pdf_path)
                        text_previewer.display_content(word_content, pdf_content)
            except Exception as e:
                st.error(f"顯示內容時發生錯誤: {str(e)}")
            
            # 比對按鈕
            if st.button("開始文字比對", key="start_text_comparison"):
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
                            with st.expander(f"匹配 #{i+1} (相似度: {match['similarity']:.2%})"):
                                st.write(f"PDF 頁碼: {match['pdf_page']}")
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("**Word 原稿**")
                                    st.text_area("", match['word_text'], height=150, key=f"word_text_{i}")
                                with c2:
                                    st.markdown("**PDF 內容**")
                                    st.text_area("", match['pdf_text'], height=150, key=f"pdf_text_{i}")
                                
                                # 根據設置顯示差異標示
                                if st.session_state.use_enhanced_diff:
                                    st.markdown("**增強型差異標示 (PDF內容為主，灰色為相同，紅色為不同):**")
                                    st.markdown(match['enhanced_diff_html'], unsafe_allow_html=True)
                                else:
                                    st.markdown("**標準差異標示:**")
                                    st.markdown(match['diff_html'], unsafe_allow_html=True)
                                
                                # 差異摘要
                                if match.get('diff_summary'):
                                    st.markdown("**句子層級差異:**")
                                    for j, diff in enumerate(match['diff_summary']):
                                        st.write(f"- 相似度: {diff['similarity']:.2%}")
                                        st.write(f"  Word: {diff['word_sentence']}")
                                        st.write(f"  PDF: {diff['pdf_sentence']}")
                    else:
                        st.warning("沒有找到匹配的內容")
                except Exception as e:
                    st.error(f"文字比對時發生錯誤: {str(e)}")
        
        # 表格比對頁籤
        with tab2:
            # 顯示表格內容預覽
            try:
                if word_tables and pdf_tables:
                    word_tables, pdf_tables = table_processor.display_tables(word_tables, pdf_tables)
                else:
                    st.warning("沒有找到足夠的表格內容進行比對。請確保文件中包含表格。")
                
                # 比對按鈕
                if st.button("開始表格比對", key="start_table_comparison") and word_tables and pdf_tables:
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
                        
                        if table_results:
                            for i, result in enumerate(table_results):
                                with st.expander(f"表格匹配 #{i+1} (相似度: {result['similarity']:.2%})"):
                                    st.write(f"Word 表格 {result['word_table']['index'] + 1} 與 PDF 表格 {result['pdf_table']['index'] + 1}")
                                    
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.markdown("**Word 表格**")
                                        st.dataframe(pd.DataFrame(result['word_table']['data']), use_container_width=True, key=f"word_table_df_{i}")
                                    with c2:
                                        st.markdown("**PDF 表格**")
                                        st.dataframe(pd.DataFrame(result['pdf_table']['data']), use_container_width=True, key=f"pdf_table_df_{i}")
                                    
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
                                        
                                        st.dataframe(pd.DataFrame(diff_df), use_container_width=True, key=f"diff_df_{i}")
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
