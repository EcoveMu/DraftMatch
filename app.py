import streamlit as st
import os, io
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from text_preview import TextPreview
from table_processor import TableProcessor
from comparison_algorithm import compare_pdf_first

def main():
    # 設定頁面
    st.set_page_config(page_title="文件比對系統", layout="wide")
    
    # 頁面標題
    st.title("文件比對系統")
    st.write("本系統用於比對 Word 原稿與 PDF 完稿，支援文字與表格比對，並可辨識無文字內容的文件。")
    
    # 側邊欄設定
    with st.sidebar:
        st.header("功能說明")
        st.info("1. 上傳 Word 與 PDF 文件\n"
                "2. 系統自動提取文字與表格\n"
                "3. 如無法提取文字，自動使用 OCR\n"
                "4. 選擇「文字比對」或「表格比對」標籤\n"
                "5. 點擊相應按鈕開始比對")
    
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
        
        # 初始化處理器
        text_previewer = TextPreview()
        table_processor = TableProcessor()
        
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
                        # 準備資料 (所有段落都參與比對，包括目錄項)
                        word_data = {'paragraphs': word_content}
                        pdf_data = {'paragraphs': pdf_content}
                        
                        # 執行比對
                        results = compare_pdf_first(word_data, pdf_data)
                    
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
                            # 獲取段落類型信息
                            word_indices = match.get('word_indices', [])
                            word_type = None
                            if word_indices and len(word_indices) > 0:
                                word_index = word_indices[0]
                                for para in word_content:
                                    if para['index'] == word_index:
                                        word_type = para.get('type', 'paragraph')
                                        break
                            
                            # 根據段落類型設置擴展器標題
                            expander_title = f"匹配 #{i+1} (相似度: {match['similarity']:.2%})"
                            if word_type:
                                type_labels = {
                                    'heading': '標題',
                                    'toc': '目錄項',
                                    'table_text': '表格內容',
                                    'metadata': '元數據',
                                    'header': '頁眉',
                                    'footer': '頁腳'
                                }
                                type_label = type_labels.get(word_type, '段落')
                                expander_title = f"{type_label} 匹配 #{i+1} (相似度: {match['similarity']:.2%})"
                            
                            with st.expander(expander_title):
                                st.write(f"PDF 頁碼: {match['pdf_page']}")
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("**Word 原稿**")
                                    st.text_area("", match['word_text'], height=150, key=f"word_text_{i}")
                                with c2:
                                    st.markdown("**PDF 內容**")
                                    st.text_area("", match['pdf_text'], height=150, key=f"pdf_text_{i}")
                                
                                st.markdown("**差異標示:**")
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
