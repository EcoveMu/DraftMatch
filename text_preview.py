import streamlit as st
import pandas as pd
from docx import Document
import fitz  # PyMuPDF
import re
import requests
import io
import json
from PIL import Image
import numpy as np
import base64
import os
from io import BytesIO

class QwenOCR:
    """阿里雲千問OCR API封裝類"""
    def __init__(self):
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        # 用戶需要將此替換為自己的API密鑰
        self.api_key = os.environ.get("QWEN_API_KEY", "")
    
    def extract_text(self, image_bytes):
        """使用阿里雲千問OCR API提取圖像中的文字
        
        參數:
            image_bytes: 圖像的字節數據
        
        返回:
            提取的文字
        """
        if not self.api_key:
            st.error("請設置環境變數 QWEN_API_KEY 以使用千問OCR功能")
            return ""
        
        # 將圖像轉換為Base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # 設置請求頭
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 設置請求體
        prompt = "請識別圖片中的所有文字，保留原有格式，包括標點符號和換行。"
        payload = {
            "model": "qwen-vl-max",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": base64_image},
                            {"text": prompt}
                        ]
                    }
                ]
            },
            "parameters": {}
        }
        
        try:
            # 發送請求
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                text = result["output"]["choices"][0]["message"]["content"]
                return text
            else:
                st.error(f"OCR API請求失敗: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            st.error(f"OCR處理出錯: {str(e)}")
            return ""

class TextPreview:
    """文字預覽類，用於從Word和PDF中提取文字並顯示"""
    
    def __init__(self):
        self.ocr = QwenOCR()
    
    def extract_word_content(self, file):
        """從Word文件中提取文字，識別目錄項目
        
        參數:
            file: 上傳的Word文件
        
        返回:
            段落列表，每個段落是一個包含索引和內容的字典
        """
        doc = Document(file)
        paragraphs = []
        
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if not text:
                continue
            
            # 檢測是否為目錄項（目錄項通常包含...或頁碼，字體大小不同）
            is_directory = False
            is_heading = False
            
            # 檢查是否包含典型的目錄模式（文字 + 點/空格 + 數字）
            directory_pattern = r'^.*?[\.\s]+\d+$'
            if re.match(directory_pattern, text):
                is_directory = True
            
            # 檢查是否為標題（通常以數字開頭，如"0.2 目錄"）
            heading_pattern = r'^\d+(\.\d+)*\s+.*$'
            if re.match(heading_pattern, text):
                is_heading = True
            
            # 如果有樣式信息，可以通過樣式進一步確認
            if hasattr(para, 'style') and para.style and para.style.name:
                style_name = para.style.name.lower()
                if 'toc' in style_name or 'content' in style_name:
                    is_directory = True
                if 'heading' in style_name or 'title' in style_name:
                    is_heading = True
            
            # 添加段落
            paragraph_type = 'normal'
            if is_directory:
                paragraph_type = 'directory'
            elif is_heading:
                paragraph_type = 'heading'
                
            paragraphs.append({
                'index': i,
                'content': text,
                'type': paragraph_type,
                'page_num': None  # Word文檔沒有頁碼概念，但添加此欄位以與PDF結構一致
            })
        
        return paragraphs
    
    def extract_pdf_content(self, file):
        """從PDF文件中提取文字，使用Qwen OCR API
        
        參數:
            file: 上傳的PDF文件
        
        返回:
            段落列表，每個段落是一個包含頁碼和內容的字典
        """
        # 保存上傳的文件
        temp_file = "temp.pdf"
        with open(temp_file, "wb") as f:
            f.write(file.getvalue())
        
        # 打開PDF文件
        doc = fitz.open(temp_file)
        paragraphs = []
        
        # 處理每一頁
        for page_num, page in enumerate(doc):
            # 獲取頁面圖像
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes(output="png")
            
            # 使用Qwen OCR獲取文本
            text = self.ocr.extract_text(img_bytes)
            
            if text:
                # 分割文本為段落
                page_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                for i, para_text in enumerate(page_paragraphs):
                    # 檢測是否為目錄項
                    is_directory = False
                    is_heading = False
                    
                    # 檢查是否包含典型的目錄模式（文字 + 點/空格 + 數字）
                    directory_pattern = r'^.*?[\.\s]+\d+$'
                    if re.match(directory_pattern, para_text):
                        is_directory = True
                    
                    # 檢查是否為標題（通常以數字開頭，如"0.2 目錄"）
                    heading_pattern = r'^\d+(\.\d+)*\s+.*$'
                    if re.match(heading_pattern, para_text):
                        is_heading = True
                    
                    # 決定段落類型
                    paragraph_type = 'normal'
                    if is_directory:
                        paragraph_type = 'directory'
                    elif is_heading:
                        paragraph_type = 'heading'
                    
                    # 添加段落
                    paragraphs.append({
                        'page_num': page_num + 1,  # 使用物理頁碼（從1開始）
                        'content': para_text,
                        'index': len(paragraphs),
                        'type': paragraph_type
                    })
        
        # 關閉和刪除臨時文件
        doc.close()
        try:
            os.remove(temp_file)
        except:
            pass
        
        return paragraphs
    
    def display_content(self, word_content, pdf_content):
        """在Streamlit界面中顯示Word和PDF內容
        
        參數:
            word_content: Word文件內容
            pdf_content: PDF文件內容
        """
        # 添加統計信息到側邊欄
        with st.sidebar:
            st.subheader("文檔統計")
            st.metric("Word段落數", len(word_content))
            st.metric("PDF段落數", len(pdf_content))
            
            # 目錄項統計
            word_dir_count = sum(1 for p in word_content if p.get('type') == 'directory')
            pdf_dir_count = sum(1 for p in pdf_content if p.get('type') == 'directory')
            
            # 標題項統計
            word_heading_count = sum(1 for p in word_content if p.get('type') == 'heading')
            pdf_heading_count = sum(1 for p in pdf_content if p.get('type') == 'heading')
            
            st.metric("Word目錄項", word_dir_count)
            st.metric("PDF目錄項", pdf_dir_count)
            st.metric("Word標題項", word_heading_count)
            st.metric("PDF標題項", pdf_heading_count)
        
        # 創建兩個標籤頁，分別顯示Word和PDF內容
        tab1, tab2 = st.tabs(["Word 內容", "PDF 內容"])
        
        with tab1:
            st.subheader("Word文件內容")
            
            # 創建一個DataFrame來顯示Word內容
            word_df = []
            for para in word_content:
                # 根據段落類型設置顯示文本
                if para.get('type') == 'directory':
                    type_text = "目錄項"
                elif para.get('type') == 'heading':
                    type_text = "標題項"
                else:
                    type_text = "一般段落"
                
                word_df.append({
                    "段落索引": para.get('index', ''),
                    "類型": type_text,
                    "內容": para.get('content', ''),
                    "頁碼": para.get('page_num', 'N/A')  # 顯示為N/A而非None
                })
            
            # 顯示為表格
            if word_df:
                st.dataframe(pd.DataFrame(word_df), use_container_width=True)
            else:
                st.info("未檢測到Word內容")
        
        with tab2:
            st.subheader("PDF文件內容")
            
            # 創建一個DataFrame來顯示PDF內容
            pdf_df = []
            for para in pdf_content:
                # 根據段落類型設置顯示文本
                if para.get('type') == 'directory':
                    type_text = "目錄項"
                elif para.get('type') == 'heading':
                    type_text = "標題項"
                else:
                    type_text = "一般段落"
                
                # 確保頁碼不為None
                page_num = para.get('page_num', '')
                if page_num is None:
                    page_num = 'N/A'
                
                pdf_df.append({
                    "頁碼": page_num,
                    "段落索引": para.get('index', ''),
                    "類型": type_text,
                    "內容": para.get('content', '')
                })
            
            # 顯示為表格
            if pdf_df:
                st.dataframe(pd.DataFrame(pdf_df), use_container_width=True)
            else:
                st.info("未檢測到PDF內容") 