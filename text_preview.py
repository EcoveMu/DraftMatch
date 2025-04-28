import streamlit as st
import pandas as pd
from docx import Document
import fitz  # PyMuPDF
from qwen_ocr import QwenOCR
import os
import io
from PIL import Image
import numpy as np

class TextPreview:
    def __init__(self):
        self.qwen_ocr = QwenOCR()
        
    def extract_word_content(self, word_file):
        """從 Word 文件中提取內容，如果沒有文字則使用 OCR"""
        doc = Document(word_file)
        paragraphs = []
        
        # 先嘗試提取文字內容
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                paragraphs.append({
                    'index': i,
                    'content': para.text.strip()
                })
        
        # 如果沒有文字內容，使用 OCR 處理
        if not paragraphs:
            st.info("Word 文件中沒有找到文字內容，將使用 OCR 進行提取...")
            
            # 將 Word 轉換為 PDF
            pdf_bytes = io.BytesIO()
            doc.save(pdf_bytes)
            pdf_bytes.seek(0)
            
            # 使用 PyMuPDF 讀取 PDF
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                # 將頁面轉換為圖片
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # 使用 Qwen OCR 提取文字
                text = self.qwen_ocr.extract_text(img)
                if text.strip():
                    paragraphs.append({
                        'index': page_num,
                        'content': text.strip()
                    })
            
            pdf_doc.close()
        
        return paragraphs
    
    def extract_pdf_content(self, pdf_file):
        """從 PDF 文件中提取內容，如果沒有文字則使用 OCR"""
        doc = fitz.open(pdf_file)
        paragraphs = []
        
        # 先嘗試直接提取文字
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            
            if text:
                paragraphs.append({
                    'index': page_num,
                    'content': text
                })
            else:
                # 如果沒有文字，使用 OCR
                st.info(f"PDF 第 {page_num + 1} 頁沒有找到文字內容，將使用 OCR 進行提取...")
                
                # 將頁面轉換為圖片
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # 使用 Qwen OCR 提取文字
                text = self.qwen_ocr.extract_text(img)
                if text.strip():
                    paragraphs.append({
                        'index': page_num,
                        'content': text.strip()
                    })
        
        doc.close()
        return paragraphs
    
    def display_content(self, word_content, pdf_content):
        """顯示內容預覽"""
        st.title("內容預覽")
        
        # 建立側邊欄
        with st.sidebar:
            st.title("內容統計資訊")
            st.write(f"Word 段落數: {len(word_content)}")
            st.write(f"PDF 頁數: {len(pdf_content)}")
            
            # 如果沒有內容，提供重新提取的選項
            if not word_content or not pdf_content:
                return st.button("重新提取內容", key="refresh_content_btn")
        
        # 建立兩個分頁
        tab1, tab2 = st.tabs(["Word 內容", "PDF 內容"])
        
        with tab1:
            st.subheader("Word 文件內容")
            if not word_content:
                st.warning("Word 文件中沒有找到任何內容")
            else:
                for para in word_content:
                    st.write(f"段落 {para['index'] + 1}:")
                    st.text_area("", para['content'], height=100, key=f"word_para_{para['index']}")
        
        with tab2:
            st.subheader("PDF 文件內容")
            if not pdf_content:
                st.warning("PDF 文件中沒有找到任何內容")
            else:
                for para in pdf_content:
                    st.write(f"頁碼 {para['index'] + 1}:")
                    st.text_area("", para['content'], height=100, key=f"pdf_para_{para['index']}")
        
        return False 