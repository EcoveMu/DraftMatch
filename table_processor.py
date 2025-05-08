import streamlit as st
import pandas as pd
import numpy as np
from docx import Document
import fitz  # PyMuPDF
from qwen_ocr import QwenOCR
import re
from PIL import Image
import io
import tempfile
import os

class TableProcessor:
    def __init__(self):
        self.qwen_ocr = QwenOCR()
        
    def extract_word_tables(self, word_file):
        """從 Word 文件中提取表格"""
        doc = Document(word_file)
        tables = []
        
        for i, table in enumerate(doc.tables):
            table_data = []
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    # 處理合併儲存格
                    text = cell.text.strip()
                    if text:
                        row_data.append(text)
                    else:
                        row_data.append("")  # 空儲存格
                table_data.append(row_data)
            
            if table_data:
                tables.append({
                    'index': i,
                    'data': table_data,
                    'page': None  # Word 沒有頁碼概念
                })
        
        return tables
    
    def extract_pdf_tables(self, pdf_file):
        """從 PDF 文件中提取表格"""
        doc = fitz.open(pdf_file)
        tables = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                
                # 將頁面轉換為圖片
                pix = page.get_pixmap()
                
                # 保存為臨時圖像文件
                temp_dir = tempfile.mkdtemp()
                temp_img_path = os.path.join(temp_dir, f"page_{page_num}.png")
                pix.save(temp_img_path)
                
                # 嘗試使用 OCR 提取文字
                text = ""
                try:
                    # 使用 QwenOCR 提取文字
                    text = self.qwen_ocr.extract_text_from_image(temp_img_path)
                    
                    # 使用 QwenOCR 提取表格
                    extracted_tables = self.qwen_ocr.extract_tables_from_image(temp_img_path)
                    
                    # 如果直接從QwenOCR獲取到表格
                    if isinstance(extracted_tables, list) and extracted_tables:
                        for table_idx, table_data in enumerate(extracted_tables):
                            if table_data and len(table_data) >= 2 and len(table_data[0]) >= 2:
                                tables.append({
                                    'index': len(tables),
                                    'data': table_data,
                                    'page': page_num + 1
                                })
                        
                        # 清理臨時文件並繼續下一頁
                        os.remove(temp_img_path)
                        os.rmdir(temp_dir)
                        continue
                    
                except Exception as e:
                    st.warning(f"使用 QwenOCR 提取 PDF 第 {page_num + 1} 頁文字失敗: {str(e)}")
                    # 如果 QwenOCR 失敗，嘗試其他方法
                    try:
                        # 使用提取 PDF 文字的原生方法
                        text = page.get_text()
                    except Exception as e2:
                        st.warning(f"使用 get_text 提取 PDF 第 {page_num + 1} 頁文字失敗: {str(e2)}")
                
                # 清理臨時文件
                os.remove(temp_img_path)
                os.rmdir(temp_dir)
                
                # 如果沒有成功提取文字，則繼續下一頁
                if not text or not text.strip():
                    continue
                    
                # 嘗試解析表格
                table_data = []
                lines = text.strip().split('\n')
                
                # 簡單表格解析邏輯
                # 查找含有多個空格或製表符的行，這些可能是表格行
                current_table = []
                for line in lines:
                    # 如果行包含多個連續空格或製表符，可能是表格
                    if '\t' in line or '  ' in line:
                        # 拆分單元格 (基於製表符或多個空格)
                        cells = re.split(r'\t+|\s{2,}', line.strip())
                        cells = [cell.strip() for cell in cells if cell.strip()]
                        
                        if len(cells) >= 2:  # 至少有兩列才視為表格行
                            current_table.append(cells)
                    elif len(current_table) > 0:
                        # 如果已經開始收集表格行，且遇到非表格行，檢查是否應該結束表格
                        if len(current_table) >= 2:  # 至少有兩行才視為表格
                            table_data = current_table
                            break
                        current_table = []
                
                # 如果找到表格，添加到結果中
                if table_data and len(table_data) >= 2:  # 至少有表頭和一行數據
                    # 標準化表格（確保每行有相同數量的列）
                    max_cols = max(len(row) for row in table_data)
                    normalized_table = []
                    for row in table_data:
                        normalized_table.append(row + [''] * (max_cols - len(row)))
                    
                    tables.append({
                        'index': len(tables),
                        'data': normalized_table,
                        'page': page_num + 1
                    })
            except Exception as e:
                st.warning(f"處理 PDF 第 {page_num + 1} 頁表格時出錯: {str(e)}")
        
        doc.close()
        return tables
    
    def normalize_table(self, table_data):
        """標準化表格數據"""
        # 確保所有行具有相同的列數
        max_cols = max(len(row) for row in table_data)
        normalized = []
        for row in table_data:
            normalized_row = row + [''] * (max_cols - len(row))
            normalized.append(normalized_row)
        return normalized
    
    def compare_tables(self, word_table, pdf_table):
        """比較兩個表格的差異"""
        # 標準化表格
        word_data = self.normalize_table(word_table['data'])
        pdf_data = self.normalize_table(pdf_table['data'])
        
        # 創建 DataFrame 以便比較
        word_df = pd.DataFrame(word_data)
        pdf_df = pd.DataFrame(pdf_data)
        
        # 計算相似度
        similarity = self.calculate_table_similarity(word_df, pdf_df)
        
        # 生成差異報告
        diff_report = self.generate_diff_report(word_df, pdf_df)
        
        return {
            'similarity': similarity,
            'word_table': word_table,
            'pdf_table': pdf_table,
            'diff_report': diff_report
        }
    
    def calculate_table_similarity(self, df1, df2):
        """計算表格相似度"""
        # 確保兩個 DataFrame 具有相同的形狀
        max_rows = max(len(df1), len(df2))
        max_cols = max(len(df1.columns), len(df2.columns))
        
        df1 = df1.reindex(index=range(max_rows), columns=range(max_cols), fill_value='')
        df2 = df2.reindex(index=range(max_rows), columns=range(max_cols), fill_value='')
        
        # 計算單元格相似度
        total_cells = max_rows * max_cols
        if total_cells == 0:
            return 0.0
            
        similar_cells = 0
        for i in range(max_rows):
            for j in range(max_cols):
                cell1 = str(df1.iloc[i, j]).strip()
                cell2 = str(df2.iloc[i, j]).strip()
                if cell1 == cell2:
                    similar_cells += 1
                elif cell1 and cell2:  # 兩個單元格都有內容
                    # 使用編輯距離計算相似度
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, cell1, cell2).ratio()
                    if similarity > 0.8:  # 相似度閾值
                        similar_cells += similarity
        
        return similar_cells / total_cells
    
    def generate_diff_report(self, df1, df2):
        """生成表格差異報告"""
        report = []
        
        # 確保兩個 DataFrame 具有相同的形狀
        max_rows = max(len(df1), len(df2))
        max_cols = max(len(df1.columns), len(df2.columns))
        
        df1 = df1.reindex(index=range(max_rows), columns=range(max_cols), fill_value='')
        df2 = df2.reindex(index=range(max_rows), columns=range(max_cols), fill_value='')
        
        for i in range(max_rows):
            for j in range(max_cols):
                cell1 = str(df1.iloc[i, j]).strip()
                cell2 = str(df2.iloc[i, j]).strip()
                
                if cell1 != cell2:
                    report.append({
                        'row': i + 1,
                        'col': j + 1,
                        'word_value': cell1,
                        'pdf_value': cell2,
                        'type': 'modified' if cell1 and cell2 else 'added' if not cell1 else 'deleted'
                    })
        
        return report
    
    def display_tables(self, word_tables, pdf_tables):
        """顯示表格內容"""
        st.title("表格內容預覽")
        
        # 建立側邊欄
        with st.sidebar:
            st.title("表格統計資訊")
            st.write(f"Word 表格數: {len(word_tables)}")
            st.write(f"PDF 表格數: {len(pdf_tables)}")
        
        # 建立兩個分頁
        tab1, tab2 = st.tabs(["Word 表格", "PDF 表格"])
        
        with tab1:
            st.subheader("Word 表格內容")
            if not word_tables:
                st.warning("Word 文件中沒有找到任何表格")
            else:
                for table in word_tables:
                    st.write(f"表格 {table['index'] + 1}:")
                    df = pd.DataFrame(table['data'])
                    st.dataframe(df, use_container_width=True, key=f"word_table_{table['index']}")
        
        with tab2:
            st.subheader("PDF 表格內容")
            if not pdf_tables:
                st.warning("PDF 文件中沒有找到任何表格")
            else:
                for table in pdf_tables:
                    st.write(f"頁碼 {table['page']} - 表格 {table['index'] + 1}:")
                    df = pd.DataFrame(table['data'])
                    st.dataframe(df, use_container_width=True, key=f"pdf_table_{table['index']}")
        
        return word_tables, pdf_tables 