import tempfile
import os
import io
import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
import numpy as np

class EasyOCR:
    """
    使用EasyOCR進行OCR文本識別的類。
    實現與QwenOCR相容的接口，以便在系統中無縫替換。
    """
    def __init__(self):
        try:
            # 延遲導入EasyOCR，因為它可能不是必需的依賴
            import easyocr
            self.reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)  # 使用繁體中文和英文
            self.available = True
        except ImportError:
            st.warning("EasyOCR未安裝。請使用pip安裝: pip install easyocr")
            self.reader = None
            self.available = False
        except Exception as e:
            st.warning(f"初始化EasyOCR時出錯: {str(e)}")
            self.reader = None
            self.available = False
    
    def is_available(self):
        """檢查OCR引擎是否可用"""
        return self.available and self.reader is not None
    
    def extract_text_from_image(self, image_path):
        """從圖像中提取文本"""
        if not self.is_available():
            return "EasyOCR未初始化"
        
        try:
            # 讀取圖像
            image = Image.open(image_path)
            # 轉換為numpy數組
            img_array = np.array(image)
            
            # 使用EasyOCR提取文本
            results = self.reader.readtext(img_array)
            
            # 組織提取的文本
            extracted_text = ""
            for detection in results:
                text = detection[1]
                confidence = detection[2]
                if confidence > 0.3:  # 過濾低置信度的結果
                    extracted_text += text + " "
            
            # 整理格式
            lines = []
            current_line = ""
            for detection in results:
                text = detection[1]
                box = detection[0]
                # 簡單的行分組啟發式方法
                if len(current_line) > 0 and abs(box[0][1] - prev_y) > 10:  # 如果y坐標差異大，視為新行
                    lines.append(current_line)
                    current_line = text
                else:
                    if len(current_line) > 0:
                        current_line += " "
                    current_line += text
                prev_y = box[0][1]
            
            if current_line:
                lines.append(current_line)
            
            return "\n".join(lines)
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
        # EasyOCR本身不直接支援表格識別，但我們可以進行基本的表格結構推斷
        if not self.is_available():
            return []
        
        try:
            # 讀取圖像
            image = Image.open(image_path)
            img_array = np.array(image)
            
            # 使用EasyOCR提取文本和位置
            results = self.reader.readtext(img_array)
            
            # 按y坐標分組，找出表格結構
            if not results:
                return []
            
            # 按y坐標分組
            y_groups = {}
            for detection in results:
                box = detection[0]  # 格式為[[x1,y1],[x2,y1],[x3,y3],[x4,y4]]
                text = detection[1]
                y_coord = int((box[0][1] + box[2][1]) / 2)  # 取中間點y坐標
                
                # 將y坐標量化，允許小範圍內的偏差
                y_quantized = y_coord // 20 * 20
                
                if y_quantized not in y_groups:
                    y_groups[y_quantized] = []
                
                # 保存文本和x坐標
                x_coord = int((box[0][0] + box[1][0]) / 2)  # 取中間點x坐標
                y_groups[y_quantized].append((x_coord, text))
            
            # 檢查是否形成表格結構
            if len(y_groups) < 2:  # 至少需要兩行
                return []
            
            # 將分組按y坐標排序
            sorted_y = sorted(y_groups.keys())
            
            # 構建表格
            table = []
            for y in sorted_y:
                # 按x坐標排序一行中的單元格
                row = [text for _, text in sorted(y_groups[y], key=lambda item: item[0])]
                if row:  # 確保行不為空
                    table.append(row)
            
            # 檢查表格是否有合理的結構
            if len(table) < 2 or max(len(row) for row in table) < 2:
                return []  # 不是有效的表格
            
            # 標準化表格（確保每行有相同數量的列）
            max_cols = max(len(row) for row in table)
            normalized_table = []
            for row in table:
                normalized_table.append(row + [''] * (max_cols - len(row)))
            
            return [normalized_table]  # 返回列表，與QwenOCR保持一致
        except Exception as e:
            print(f"提取表格時出錯: {str(e)}")
            return []
    
    def extract_text(self, image):
        """直接從PIL Image對象中提取文本"""
        try:
            # 如果是PIL圖像，轉換為臨時文件
            temp_dir = tempfile.mkdtemp()
            temp_img_path = os.path.join(temp_dir, "temp_image.png")
            image.save(temp_img_path)
            
            # 使用現有方法提取文本
            text = self.extract_text_from_image(temp_img_path)
            
            # 清理臨時文件
            os.remove(temp_img_path)
            os.rmdir(temp_dir)
            
            return text
        except Exception as e:
            return f"提取文本時出錯: {str(e)}" 