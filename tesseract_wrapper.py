import tempfile
import os
import io
import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
import numpy as np
import re

class TesseractOCR:
    """
    使用Tesseract進行OCR文本識別的類。
    實現與QwenOCR相容的接口，以便在系統中無縫替換。
    """
    def __init__(self):
        try:
            # 延遲導入pytesseract，因為它可能不是必需的依賴
            import pytesseract
            self.pytesseract = pytesseract
            self.available = True
        except ImportError:
            st.warning("Tesseract OCR未安裝。請安裝Tesseract OCR並使用pip安裝pytesseract庫: pip install pytesseract")
            self.pytesseract = None
            self.available = False
        except Exception as e:
            st.warning(f"初始化Tesseract OCR時出錯: {str(e)}")
            self.pytesseract = None
            self.available = False
    
    def is_available(self):
        """檢查OCR引擎是否可用"""
        return self.available and self.pytesseract is not None
    
    def extract_text_from_image(self, image_path):
        """從圖像中提取文本"""
        if not self.is_available():
            return "Tesseract OCR未初始化"
        
        try:
            # 讀取圖像
            image = Image.open(image_path)
            
            # 使用Tesseract提取文本
            text = self.pytesseract.image_to_string(image, lang='chi_tra+eng')
            
            return text
        except Exception as e:
            return f"提取文本時出錯: {str(e)}"
    
    def extract_text_from_pdf_page(self, pdf_path, page_num):
        """從PDF文件的指定頁面提取文本"""
        try:
            # 將PDF頁面轉換為圖像
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            # 保存為臨時圖像文件
            temp_dir = tempfile.mkdtemp()
            temp_img_path = os.path.join(temp_dir, f"pdf_page_{page_num}.png")
            pix.save(temp_img_path)
            
            # 使用OCR提取文本
            text = self.extract_text_from_image(temp_img_path)
            
            # 刪除臨時文件
            os.remove(temp_img_path)
            os.rmdir(temp_dir)
            
            return text, temp_img_path
        except Exception as e:
            return f"從PDF提取文本時出錯: {str(e)}", None
    
    def extract_tables_from_image(self, image_path):
        """從圖像中提取表格內容"""
        if not self.is_available():
            return []
        
        try:
            # 讀取圖像
            image = Image.open(image_path)
            
            # 使用Tesseract提取文本（帶有位置信息）
            data = self.pytesseract.image_to_data(image, lang='chi_tra+eng', output_type=self.pytesseract.Output.DICT)
            
            # 將提取的數據轉換為表格形式
            text_data = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # 過濾低置信度的結果
                    text = data['text'][i]
                    if text.strip():  # 忽略空白文本
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        line_num = data['line_num'][i]
                        block_num = data['block_num'][i]
                        text_data.append((text, x, y, line_num, block_num))
            
            # 嘗試通過行號和區塊號識別表格結構
            if not text_data:
                return []
                
            # 將文本按行和區塊分組
            line_groups = {}
            for text, x, y, line_num, block_num in text_data:
                key = (block_num, line_num)
                if key not in line_groups:
                    line_groups[key] = []
                line_groups[key].append((x, text))
            
            # 識別可能的表格行
            table_lines = []
            for key in sorted(line_groups.keys()):
                line_elements = line_groups[key]
                # 按x坐標排序
                sorted_elements = [text for _, text in sorted(line_elements, key=lambda item: item[0])]
                
                # 檢查是否可能是表格行（至少有2個元素且包含分隔符）
                line_text = " ".join(sorted_elements)
                if (len(sorted_elements) >= 2 and 
                    (re.search(r'\t|\s{2,}', line_text) or 
                     any(re.search(r'[|:;\t]', elem) for elem in sorted_elements))):
                    table_lines.append(sorted_elements)
            
            # 檢查是否形成表格結構
            if len(table_lines) < 2:  # 至少需要兩行
                return []
            
            # 標準化表格（確保每行有相同數量的列）
            max_cols = max(len(row) for row in table_lines)
            normalized_table = []
            for row in table_lines:
                normalized_table.append(row + [''] * (max_cols - len(row)))
            
            return [normalized_table]  # 返回列表，與QwenOCR保持一致
        except Exception as e:
            print(f"提取表格時出錯: {str(e)}")
            return []
    
    def extract_text(self, image):
        """直接從PIL Image對象中提取文本"""
        try:
            # 使用Tesseract直接處理PIL圖像
            if not self.is_available():
                return "Tesseract OCR未初始化"
            
            # 使用pytesseract直接處理PIL圖像
            text = self.pytesseract.image_to_string(image, lang='chi_tra+eng')
            return text
        except Exception as e:
            return f"提取文本時出錯: {str(e)}" 