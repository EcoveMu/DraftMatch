import streamlit as st
import io, os
import pandas as pd
from docx import Document
import fitz  # PyMuPDF
from text_preview import TextPreview
from table_processor import TableProcessor
from comparison_algorithm import compare_pdf_first

def main():
    st.set_page_config(page_title="文件比對系統", layout="wide")
    
    # 初始化處理器
    text_previewer = TextPreview()
    table_processor = TableProcessor()
    
    st.title("文件比對系統")
    
    # 檔案上傳
    col1, col2 = st.columns(2)
    with col1:
        word_file = st.file_uploader("上傳 Word 原稿", type=['docx'], key="word_uploader")
    with col2:
        pdf_file = st.file_uploader("上傳 PDF 完稿", type=['pdf'], key="pdf_uploader")
    
    if word_file and pdf_file:
        # 儲存上傳的檔案
        word_path = "temp_word.docx"
        pdf_path = "temp_pdf.pdf"
        
        with open(word_path, "wb") as f:
            f.write(word_file.getvalue())
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getvalue())
        
        # 提取內容
        word_content = text_previewer.extract_word_content(word_path)
        pdf_content = text_previewer.extract_pdf_content(pdf_path)
        
        # 提取表格
        word_tables = table_processor.extract_word_tables(word_path)
        pdf_tables = table_processor.extract_pdf_tables(pdf_path)
        
        # 建立分頁
        tab1, tab2 = st.tabs(["文字比對", "表格比對"])
        
        with tab1:
            # 顯示文字內容預覽
            if text_previewer.display_content(word_content, pdf_content):
                # 重新提取 PDF 內容
                pdf_content = text_previewer.extract_pdf_content(pdf_path)
                text_previewer.display_content(word_content, pdf_content)
            
            # 文字比對按鈕
            if st.button("開始文字比對", key="start_text_comparison"):
                # 準備比對資料
                word_data = {'paragraphs': word_content}
                pdf_data = {'paragraphs': pdf_content}
                
                # 執行比對
                results = compare_pdf_first(word_data, pdf_data)
                
                # 顯示比對結果
                st.title("文字比對結果")
                
                # 顯示統計資訊
                st.write(f"總共比對 {results['statistics']['total_pdf']} 頁 PDF")
                st.write(f"成功匹配 {results['statistics']['matched']} 段")
                st.write(f"未匹配 PDF 段落: {results['statistics']['unmatched_pdf']}")
                st.write(f"未匹配 Word 段落: {results['statistics']['unmatched_word']}")
                
                # 顯示詳細比對結果
                for i, match in enumerate(results['matches']):
                    st.write(f"PDF 頁碼: {match['pdf_page']}")
                    st.write(f"相似度: {match['similarity']:.2%}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Word 原稿")
                        st.text_area("", match['word_text'], height=150, key=f"result_word_{i}")
                    with col2:
                        st.subheader("PDF 內容")
                        st.text_area("", match['pdf_text'], height=150, key=f"result_pdf_{i}")
                    
                    st.write("差異標示:")
                    st.markdown(match['diff_html'], unsafe_allow_html=True)
                    
                    # 顯示差異摘要
                    if match.get('diff_summary'):
                        with st.expander(f"差異摘要 #{i+1}"):
                            for j, diff in enumerate(match['diff_summary']):
                                st.write(f"相似度: {diff['similarity']:.2%}")
                                st.write(f"Word: {diff['word_sentence']}")
                                st.write(f"PDF: {diff['pdf_sentence']}")
                                st.divider()
                    
                    st.divider()
        
        with tab2:
            # 顯示表格內容預覽
            word_tables, pdf_tables = table_processor.display_tables(word_tables, pdf_tables)
            
            # 表格比對按鈕
            if st.button("開始表格比對", key="start_table_comparison"):
                st.title("表格比對結果")
                
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
                
                # 顯示表格比對結果
                for i, result in enumerate(table_results):
                    st.write(f"Word 表格 {result['word_table']['index'] + 1} 與 PDF 表格 {result['pdf_table']['index'] + 1} 的比對結果")
                    st.write(f"相似度: {result['similarity']:.2%}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Word 表格")
                        st.dataframe(pd.DataFrame(result['word_table']['data']), use_container_width=True, key=f"comp_word_table_{i}")
                    with col2:
                        st.subheader("PDF 表格")
                        st.dataframe(pd.DataFrame(result['pdf_table']['data']), use_container_width=True, key=f"comp_pdf_table_{i}")
                    
                    # 顯示差異報告
                    if result['diff_report']:
                        with st.expander(f"表格差異報告 #{i+1}"):
                            for j, diff in enumerate(result['diff_report']):
                                st.write(f"位置: 第 {diff['row']} 行, 第 {diff['col']} 列")
                                if diff['type'] == 'modified':
                                    st.write(f"Word: {diff['word_value']}")
                                    st.write(f"PDF: {diff['pdf_value']}")
                                elif diff['type'] == 'added':
                                    st.write(f"PDF 新增: {diff['pdf_value']}")
                                else:  # deleted
                                    st.write(f"Word 刪除: {diff['word_value']}")
                                st.divider()
                    
                    st.divider()
        
        # 清理暫存檔案
        os.remove(word_path)
        os.remove(pdf_path)

if __name__ == "__main__":
    main() 