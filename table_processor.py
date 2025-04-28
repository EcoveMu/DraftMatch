import streamlit as st
import pandas as pd
import numpy as np
from docx import Document
import fitz  # PyMuPDF
from text_preview import QwenOCR
import re
from PIL import Image
import io
import tempfile
import os
from typing import List, Dict, Any, Tuple

class TableProcessor:
    def __init__(self):
        self.qwen_ocr = QwenOCR()
        
    def extract_word_tables(self, word_file):
        """從 Word 文件中提取表格，同時識別包含目錄項目的表格"""
        # 創建臨時文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            tmp_file.write(word_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            doc = Document(tmp_path)
            tables = []
            
            for i, table in enumerate(doc.tables):
                table_data = []
                is_directory_table = False
                
                # 檢查表格的第一行是否包含目錄相關的關鍵詞
                if len(table.rows) > 0:
                    first_row_text = ' '.join([cell.text for cell in table.rows[0].cells])
                    if any(kw in first_row_text.lower() for kw in ['目錄', '大綱', 'contents', 'table of contents']):
                        is_directory_table = True
                
                for row in table.rows:
                    row_data = []
                    for cell in row.cells:
                        # 處理合併儲存格
                        text = cell.text.strip()
                        row_data.append(text)
                    
                    # 確保行數據不為空
                    if any(cell for cell in row_data):
                        table_data.append(row_data)
                
                if table_data:
                    # 標準化表格 (確保每行列數相同)
                    table_data = self.normalize_table(table_data)
                    
                    tables.append({
                        'index': i,
                        'data': table_data,
                        'page': None,  # Word 沒有頁碼概念
                        'is_directory': is_directory_table
                    })
            
            return tables
        
        finally:
            # 清理臨時文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def extract_pdf_tables(self, pdf_file):
        """從 PDF 文件中提取表格，同時識別包含目錄項目的表格"""
        # 創建臨時文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            doc = fitz.open(tmp_path)
            tables = []
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    
                    # 提取頁面文本
                    page_text = page.get_text()
                    
                    # 檢查頁面是否包含目錄關鍵詞
                    is_directory_page = any(kw in page_text.lower() for kw in ['目錄', '大綱', 'contents', 'table of contents'])
                    
                    # 將頁面轉換為圖片
                    pix = page.get_pixmap()
                    img_bytes = io.BytesIO()
                    pix.save(img_bytes, "png")
                    img_bytes.seek(0)
                    
                    # 使用 OCR 提取文字
                    try:
                        extracted_text = self.qwen_ocr.extract_text(img_bytes.getvalue())
                        
                        # 解析文本中的表格
                        table_data = self._parse_text_for_tables(extracted_text)
                        
                        # 如果通過文本解析找到表格
                        if table_data and len(table_data) >= 2:  # 至少有表頭和一行數據
                            tables.append({
                                'index': len(tables),
                                'data': table_data,
                                'page': page_num + 1,
                                'is_directory': is_directory_page
                            })
                            continue  # 成功找到表格，跳過其他方法
                    except Exception as e:
                        st.warning(f"OCR提取PDF第{page_num+1}頁表格失敗: {str(e)}")
                    
                    # 如果OCR方法失敗，嘗試使用PDF內置表格結構
                    try:
                        # 使用PyMuPDF的表格檢測功能
                        tabs = page.find_tables()
                        if tabs and len(tabs.tables) > 0:
                            for t_idx, tab in enumerate(tabs.tables):
                                rows = tab.rows
                                cols = tab.cols
                                table_data = []
                                
                                for r in range(rows):
                                    row_data = []
                                    for c in range(cols):
                                        cell = tab.cell(r, c)
                                        if cell:
                                            text = page.get_text("text", clip=cell.rect).strip()
                                            row_data.append(text)
                                        else:
                                            row_data.append("")
                                    
                                    if any(cell.strip() for cell in row_data):
                                        table_data.append(row_data)
                                
                                if table_data:
                                    tables.append({
                                        'index': len(tables),
                                        'data': table_data,
                                        'page': page_num + 1,
                                        'is_directory': is_directory_page
                                    })
                    except Exception as e:
                        st.warning(f"內置方法提取PDF第{page_num+1}頁表格失敗: {str(e)}")
                        
                except Exception as e:
                    st.warning(f"處理 PDF 第 {page_num + 1} 頁表格時出錯: {str(e)}")
            
            doc.close()
            return tables
        
        finally:
            # 清理臨時文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _parse_text_for_tables(self, text: str) -> List[List[str]]:
        """解析文本中的表格結構"""
        if not text or not text.strip():
            return []
        
        # 以多行文本形式分割
        lines = text.strip().split('\n')
        
        # 查找可能的表格行
        potential_table_lines = []
        for line in lines:
            # 過濾空行和只有少量字符的行
            if not line.strip() or len(line.strip()) < 5:
                continue
            
            # 如果行包含多個製表符或連續空格，可能是表格
            if '\t' in line or re.search(r'\s{2,}', line):
                potential_table_lines.append(line)
        
        if len(potential_table_lines) < 2:  # 至少需要兩行來形成表格
            return []
        
        # 分析這些行以檢測列分隔符
        delimiter_patterns = ['\t', '  ', '   ', '    ']
        best_delimiter = None
        max_consistent_columns = 0
        
        for delimiter in delimiter_patterns:
            column_counts = [len(re.split(delimiter, line.strip())) for line in potential_table_lines]
            
            # 檢查列數的一致性
            if len(set(column_counts)) <= 2 and min(column_counts) >= 2:
                avg_columns = sum(column_counts) / len(column_counts)
                if avg_columns > max_consistent_columns:
                    max_consistent_columns = avg_columns
                    best_delimiter = delimiter
        
        if not best_delimiter:
            return []
        
        # 使用最佳分隔符提取表格數據
        table_data = []
        for line in potential_table_lines:
            cells = [cell.strip() for cell in re.split(best_delimiter, line.strip()) if cell.strip()]
            if len(cells) >= 2:  # 至少有兩個單元格
                table_data.append(cells)
        
        return self.normalize_table(table_data)
    
    def normalize_table(self, table_data: List[List[str]]) -> List[List[str]]:
        """標準化表格數據，確保所有行具有相同的列數"""
        if not table_data:
            return []
        
        # 找出最大列數
        max_cols = max(len(row) for row in table_data)
        
        # 確保每行都有相同的列數
        normalized = []
        for row in table_data:
            normalized_row = row + [''] * (max_cols - len(row))
            normalized.append(normalized_row)
        
        return normalized
    
    def compare_tables(self, word_tables: List[Dict], pdf_tables: List[Dict]) -> List[Dict]:
        """比較 Word 和 PDF 文件中的表格"""
        comparison_results = []
        
        # 標記匹配狀態
        word_matched = [False] * len(word_tables)
        pdf_matched = [False] * len(pdf_tables)
        
        # 計算所有表格對的相似度
        for w_idx, word_table in enumerate(word_tables):
            for p_idx, pdf_table in enumerate(pdf_tables):
                # 如果已經匹配，跳過
                if word_matched[w_idx] or pdf_matched[p_idx]:
                    continue
                
                # 創建 DataFrame 以便比較
                word_df = pd.DataFrame(word_table['data'])
                pdf_df = pd.DataFrame(pdf_table['data'])
                
                # 計算相似度
                similarity = self.calculate_table_similarity(word_df, pdf_df)
                
                # 如果相似度超過閾值，認為是匹配的表格
                if similarity >= 0.6:  # 相似度閾值
                    # 生成差異報告
                    diff_report = self.generate_diff_report(word_df, pdf_df)
                    
                    comparison_results.append({
                        'word_table_index': w_idx,
                        'pdf_table_index': p_idx,
                        'word_table': word_table,
                        'pdf_table': pdf_table,
                        'similarity': similarity,
                        'diff_report': diff_report,
                        'is_directory': word_table.get('is_directory', False) or pdf_table.get('is_directory', False)
                    })
                    
                    # 標記為已匹配
                    word_matched[w_idx] = True
                    pdf_matched[p_idx] = True
                    break
        
        # 添加未匹配的表格
        for w_idx, matched in enumerate(word_matched):
            if not matched:
                comparison_results.append({
                    'word_table_index': w_idx,
                    'pdf_table_index': None,
                    'word_table': word_tables[w_idx],
                    'pdf_table': None,
                    'similarity': 0.0,
                    'diff_report': [],
                    'is_directory': word_tables[w_idx].get('is_directory', False)
                })
        
        for p_idx, matched in enumerate(pdf_matched):
            if not matched:
                comparison_results.append({
                    'word_table_index': None,
                    'pdf_table_index': p_idx,
                    'word_table': None,
                    'pdf_table': pdf_tables[p_idx],
                    'similarity': 0.0,
                    'diff_report': [],
                    'is_directory': pdf_tables[p_idx].get('is_directory', False)
                })
        
        return comparison_results
    
    def calculate_table_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """計算表格相似度，考慮單元格內容和表格結構"""
        # 確保兩個 DataFrame 具有相同的形狀
        max_rows = max(len(df1), len(df2))
        max_cols = max(len(df1.columns), len(df2.columns))
        
        if max_rows == 0 or max_cols == 0:
            return 0.0
        
        # 調整 DataFrame 大小
        df1_resized = df1.reindex(index=range(max_rows), columns=range(max_cols), fill_value='')
        df2_resized = df2.reindex(index=range(max_rows), columns=range(max_cols), fill_value='')
        
        # 計算單元格相似度
        total_cells = max_rows * max_cols
        if total_cells == 0:
            return 0.0
        
        similar_cells = 0
        for i in range(max_rows):
            for j in range(max_cols):
                # 獲取單元格內容
                cell1 = str(df1_resized.iloc[i, j]).strip()
                cell2 = str(df2_resized.iloc[i, j]).strip()
                
                # 完全相同
                if cell1 == cell2:
                    similar_cells += 1
                # 兩個單元格都有內容但不同
                elif cell1 and cell2:
                    # 計算文本相似度
                    from difflib import SequenceMatcher
                    ratio = SequenceMatcher(None, cell1, cell2).ratio()
                    similar_cells += ratio
        
        # 考慮表格形狀相似度 (行數和列數)
        shape_similarity = 1.0 - (abs(len(df1) - len(df2)) / max(max_rows, 1) + 
                                 abs(len(df1.columns) - len(df2.columns)) / max(max_cols, 1)) / 2
        
        # 總體相似度 (70% 內容, 30% 結構)
        content_similarity = similar_cells / total_cells if total_cells > 0 else 0
        return 0.7 * content_similarity + 0.3 * shape_similarity
    
    def generate_diff_report(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[Dict]:
        """生成表格差異報告，突出顯示變化的內容"""
        report = []
        
        # 確保兩個 DataFrame 具有相同的形狀
        max_rows = max(len(df1), len(df2))
        max_cols = max(len(df1.columns), len(df2.columns))
        
        if max_rows == 0 or max_cols == 0:
            return report
        
        # 調整 DataFrame 大小
        df1_resized = df1.reindex(index=range(max_rows), columns=range(max_cols), fill_value='')
        df2_resized = df2.reindex(index=range(max_rows), columns=range(max_cols), fill_value='')
        
        # 檢查每個單元格
        for i in range(max_rows):
            for j in range(max_cols):
                # 獲取單元格內容
                cell1 = str(df1_resized.iloc[i, j]).strip()
                cell2 = str(df2_resized.iloc[i, j]).strip()
                
                # 如果內容不同，添加到報告
                if cell1 != cell2:
                    # 判斷差異類型
                    if not cell1:
                        diff_type = 'added'  # 添加
                    elif not cell2:
                        diff_type = 'deleted'  # 刪除
                    else:
                        diff_type = 'modified'  # 修改
                        
                        # 檢查是否包含重要數字的變化
                        if re.search(r'\d+', cell1) and re.search(r'\d+', cell2):
                            # 提取數字
                            nums1 = re.findall(r'\d+', cell1)
                            nums2 = re.findall(r'\d+', cell2)
                            if nums1 != nums2:
                                diff_type = 'number_changed'  # 數字變化
                    
                    # 計算文本差異
                    from difflib import SequenceMatcher
                    matcher = SequenceMatcher(None, cell1, cell2)
                    diff_html = ""
                    
                    # 生成差異 HTML
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == 'equal':
                            diff_html += cell1[i1:i2]
                        elif tag == 'delete':
                            diff_html += f"<span style='background-color:#ffcccc;text-decoration:line-through;'>{cell1[i1:i2]}</span>"
                        elif tag == 'insert':
                            diff_html += f"<span style='background-color:#ccffcc;'>{cell2[j1:j2]}</span>"
                        elif tag == 'replace':
                            diff_html += f"<span style='background-color:#ffcccc;text-decoration:line-through;'>{cell1[i1:i2]}</span>"
                            diff_html += f"<span style='background-color:#ccffcc;'>{cell2[j1:j2]}</span>"
                    
                    report.append({
                        'row': i + 1,
                        'col': j + 1,
                        'word_value': cell1,
                        'pdf_value': cell2,
                        'type': diff_type,
                        'diff_html': diff_html
                    })
        
        return report
    
    def display_tables(self, word_tables: List[Dict], pdf_tables: List[Dict]):
        """在 Streamlit 界面中顯示表格內容並比較差異"""
        if not word_tables and not pdf_tables:
            st.warning("未在文件中檢測到表格。")
            return
        
        # 執行表格比較
        comparison_results = self.compare_tables(word_tables, pdf_tables)
        
        # 計算統計信息
        total_word_tables = len(word_tables)
        total_pdf_tables = len(pdf_tables)
        matched_tables = sum(1 for r in comparison_results 
                             if r['word_table_index'] is not None and r['pdf_table_index'] is not None)
        
        # 建立側邊欄統計
        with st.sidebar:
            st.subheader("表格統計資訊")
            st.metric("Word 表格數", total_word_tables)
            st.metric("PDF 表格數", total_pdf_tables)
            st.metric("匹配表格數", matched_tables)
            
            if total_word_tables > 0 and total_pdf_tables > 0:
                match_rate = matched_tables / min(total_word_tables, total_pdf_tables)
                st.progress(match_rate, f"表格匹配率: {match_rate * 100:.1f}%")
        
        # 建立表格顯示區域
        st.subheader("表格比對結果")
        
        # 先顯示已匹配的表格
        matched_results = [r for r in comparison_results 
                           if r['word_table_index'] is not None and r['pdf_table_index'] is not None]
        
        if matched_results:
            st.write("### 已匹配表格")
            for idx, result in enumerate(matched_results):
                is_directory = result.get('is_directory', False)
                similarity = result['similarity']
                
                with st.expander(
                    f"表格 {idx+1}: Word {result['word_table_index']+1} ↔ PDF {result['pdf_table_index']+1} "
                    f"(相似度: {similarity:.2f}) {' - 目錄表格' if is_directory else ''}"
                ):
                    col1, col2 = st.columns(2)
                    
                    # Word 表格
                    with col1:
                        st.markdown(f"**Word 表格{' (目錄)' if is_directory else ''}:**")
                        st.dataframe(
                            pd.DataFrame(result['word_table']['data']),
                            use_container_width=True
                        )
                    
                    # PDF 表格
                    with col2:
                        st.markdown(f"**PDF 表格 (頁碼: {result['pdf_table']['page']}){' (目錄)' if is_directory else ''}:**")
                        st.dataframe(
                            pd.DataFrame(result['pdf_table']['data']),
                            use_container_width=True
                        )
                    
                    # 差異報告
                    if result['diff_report']:
                        st.markdown("**差異報告:**")
                        diff_df = pd.DataFrame(result['diff_report'])
                        st.dataframe(
                            diff_df[['row', 'col', 'word_value', 'pdf_value', 'type']],
                            use_container_width=True
                        )
                        
                        # 顯示具體差異
                        st.markdown("**單元格差異預覽:**")
                        for diff in result['diff_report'][:5]:  # 限制顯示前5個差異
                            st.markdown(f"**位置 ({diff['row']}, {diff['col']})** - 類型: {diff['type']}")
                            st.markdown(diff['diff_html'], unsafe_allow_html=True)
        
        # 顯示未匹配的表格
        unmatched_word = [r for r in comparison_results 
                          if r['word_table_index'] is not None and r['pdf_table_index'] is None]
        
        if unmatched_word:
            st.write("### 未匹配的 Word 表格")
            for idx, result in enumerate(unmatched_word):
                is_directory = result.get('is_directory', False)
                with st.expander(f"Word 表格 {result['word_table_index']+1}{' (目錄表格)' if is_directory else ''}"):
                    st.dataframe(
                        pd.DataFrame(result['word_table']['data']),
                        use_container_width=True
                    )
        
        unmatched_pdf = [r for r in comparison_results 
                         if r['word_table_index'] is None and r['pdf_table_index'] is not None]
        
        if unmatched_pdf:
            st.write("### 未匹配的 PDF 表格")
            for idx, result in enumerate(unmatched_pdf):
                is_directory = result.get('is_directory', False)
                with st.expander(f"PDF 表格 {result['pdf_table_index']+1} (頁碼: {result['pdf_table']['page']}){' (目錄表格)' if is_directory else ''}"):
                    st.dataframe(
                        pd.DataFrame(result['pdf_table']['data']),
                        use_container_width=True
                    )
        
        return comparison_results 