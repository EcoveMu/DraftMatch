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
import tempfile
from io import BytesIO
import pytesseract
import traceback
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class BaseOCR:
    """OCR基礎類，所有OCR引擎都繼承此類"""
    def __init__(self):
        self.name = "基礎OCR"
    
    def extract_text(self, image_bytes):
        """提取圖像中的文字"""
        raise NotImplementedError("必須在子類中實現")
    
    def is_available(self):
        """檢查OCR引擎是否可用"""
        return True

class TesseractOCR(BaseOCR):
    """使用Tesseract OCR引擎提取文字"""
    def __init__(self, lang="chi_tra+eng"):
        super().__init__()
        self.name = "Tesseract OCR"
        self.lang = lang
        self._check_tesseract()
    
    def _check_tesseract(self):
        """檢查Tesseract是否已安裝並可用"""
        try:
            pytesseract.get_tesseract_version()
            self._available = True
        except Exception as e:
            self._available = False
            st.warning(f"Tesseract OCR 未安裝或不可用: {str(e)}")
    
    def is_available(self):
        return self._available
    
    def extract_text(self, image_bytes):
        """使用Tesseract提取圖像中的文字"""
        if not self.is_available():
            return "Tesseract OCR未安裝或不可用"
        
        try:
            # 將圖像字節轉換為PIL圖像
            img = Image.open(BytesIO(image_bytes))
            
            # 使用Tesseract提取文字
            text = pytesseract.image_to_string(img, lang=self.lang)
            return text
        except Exception as e:
            st.error(f"Tesseract OCR處理出錯: {str(e)}")
            return ""

class EasyOCR(BaseOCR):
    """使用EasyOCR引擎提取文字"""
    def __init__(self, lang=["ch_tra", "en"]):
        super().__init__()
        self.name = "EasyOCR"
        self.lang = lang
        self._reader = None
        self._available = EASYOCR_AVAILABLE
    
    def _initialize_reader(self):
        """初始化EasyOCR讀取器"""
        if not self._reader and EASYOCR_AVAILABLE:
            try:
                self._reader = easyocr.Reader(self.lang)
                return True
            except Exception as e:
                st.warning(f"EasyOCR初始化失敗: {str(e)}")
                self._available = False
                return False
        return self._reader is not None
    
    def is_available(self):
        return self._available
    
    def extract_text(self, image_bytes):
        """使用EasyOCR提取圖像中的文字"""
        if not self.is_available():
            return "EasyOCR未安裝或不可用"
        
        # 初始化讀取器
        if not self._initialize_reader():
            return "EasyOCR初始化失敗"
        
        try:
            # 將圖像字節轉換為numpy數組
            img = Image.open(BytesIO(image_bytes))
            img_np = np.array(img)
            
            # 使用EasyOCR提取文字
            results = self._reader.readtext(img_np)
            
            # 整合結果
            texts = [result[1] for result in results]
            return "\n".join(texts)
        except Exception as e:
            st.error(f"EasyOCR處理出錯: {str(e)}")
            traceback.print_exc()
            return ""

class QwenOCR(BaseOCR):
    """阿里雲千問OCR API封裝類，支持官方API和免費API"""
    def __init__(self):
        super().__init__()
        self.name = "千問OCR"
        # 官方API設置
        self.official_api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self.official_model = "qwen-vl-max"
        
        # 免費API設置
        self.free_api_url = "https://api.qwen-2.com/v1/chat/completions"
        self.free_model = "qwen2.5-7b-instruct"
        
        # 用戶設置的API密鑰
        self._user_api_key = ""
        # 從環境變數獲取的API密鑰
        self._env_api_key = os.environ.get("QWEN_API_KEY", "")
        # 預設API密鑰（已刪除無效密鑰）
        self._default_api_key = ""
    
    @property
    def api_key(self):
        """獲取API密鑰，優先使用用戶設置的密鑰"""
        if self._user_api_key:
            return self._user_api_key
        elif self._env_api_key:
            return self._env_api_key
        else:
            return self._default_api_key
    
    @api_key.setter
    def api_key(self, value):
        """設置API密鑰（環境變數）"""
        if value:
            self._env_api_key = value
        else:
            self._env_api_key = ""
    
    def set_api_key(self, key):
        """直接設置API密鑰（用戶指定）"""
        if key:
            self._user_api_key = key
            return True
        return False
    
    def should_use_free_api(self):
        """判斷是否使用免費API"""
        return not self.api_key or self.api_key.strip() == ""
    
    def is_available(self):
        """QwenOCR總是可用，因為可以使用免費API"""
        return True
    
    def extract_text(self, image_bytes):
        """使用OCR API提取圖像中的文字
        
        參數:
            image_bytes: 圖像的字節數據
        
        返回:
            提取的文字
        """
        try:
            # 將圖像轉換為Base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # 判斷使用哪種API
            if self.should_use_free_api():
                return self._extract_text_free_api(base64_image)
            else:
                return self._extract_text_official_api(base64_image)
        except Exception as e:
            st.error(f"Qwen OCR處理出錯: {str(e)}")
            traceback.print_exc()
            return ""
    
    def _extract_text_official_api(self, base64_image):
        """使用官方API提取文本"""
        # 設置請求頭
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 設置請求體
        payload = {
            "model": self.official_model,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一個專業的OCR助手，請提取圖像中的所有文本內容，保持原始格式。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"image": base64_image},
                            {"text": "請提取這個圖像中的所有文字，保留原有格式，包括標點符號和換行。"}
                        ]
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        
        try:
            # 發送請求
            response = requests.post(self.official_api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                text = result["output"]["choices"][0]["message"]["content"]
                return text
            else:
                st.error(f"Qwen官方API請求失敗: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            st.error(f"Qwen官方API請求出錯: {str(e)}")
            return ""
    
    def _extract_text_free_api(self, base64_image):
        """使用免費API提取文本"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.free_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一個專業的OCR助手，請提取圖像中的所有文本內容，保持原始格式。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                        {
                            "type": "text",
                            "text": "請提取這個圖像中的所有文字，保留原有格式，包括標點符號和換行。"
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.free_api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                return text
            else:
                st.error(f"Qwen免費API請求失敗: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            st.error(f"Qwen免費API請求出錯: {str(e)}")
            return ""

class OCRManager:
    """OCR管理器，用於管理多個OCR引擎"""
    def __init__(self):
        self.engines = {
            "tesseract": TesseractOCR(),
            "easyocr": EasyOCR(),
            "qwen": QwenOCR()
        }
        self.current_engine_name = "qwen"  # 默認使用千問OCR，因為它始終可用
    
    def get_available_engines(self):
        """獲取所有可用的OCR引擎"""
        return {name: engine for name, engine in self.engines.items() if engine.is_available()}
    
    def set_engine(self, engine_name):
        """設置當前使用的OCR引擎"""
        if engine_name in self.engines:
            self.current_engine_name = engine_name
            return True
        return False
    
    def get_current_engine(self):
        """獲取當前使用的OCR引擎"""
        return self.engines[self.current_engine_name]
    
    def extract_text(self, image_bytes):
        """使用當前OCR引擎提取文字"""
        engine = self.get_current_engine()
        return engine.extract_text(image_bytes)
    
    def get_current_engine_name(self):
        """獲取當前引擎名稱"""
        return self.current_engine_name
    
    def get_engine_by_name(self, name):
        """根據名稱獲取引擎"""
        return self.engines.get(name)

class TextPreview:
    """文字預覽類，用於從Word和PDF中提取文字並顯示"""
    
    def __init__(self):
        self.ocr_manager = OCRManager()
    
    @property
    def ocr(self):
        """獲取當前OCR引擎"""
        return self.ocr_manager.get_current_engine()
    
    def set_ocr_engine(self, engine_name):
        """設置OCR引擎"""
        return self.ocr_manager.set_engine(engine_name)
    
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
        """從PDF文件中提取文字，使用當前OCR引擎
        
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
        
        # 提取OCR引擎信息
        current_engine = self.ocr_manager.get_current_engine()
        st.info(f"使用 {current_engine.name} 進行OCR文本提取...")
        
        # 處理每一頁
        for page_num, page in enumerate(doc):
            # 獲取頁面圖像
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes(output="png")
            
            # 使用當前OCR引擎獲取文本
            text = current_engine.extract_text(img_bytes)
            
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