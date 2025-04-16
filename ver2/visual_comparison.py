import os
import tempfile
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import numpy as np
import cv2
import requests
import json

class VisualComparison:
    """
    用於視覺化比對結果的類
    """
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def highlight_pdf_page(self, pdf_path, page_num, text_to_highlight, output_path=None):
        """
        在PDF頁面上標記指定文本，並返回標記後的圖像
        
        參數:
        - pdf_path: PDF文件路徑
        - page_num: 頁碼（從1開始）
        - text_to_highlight: 要標記的文本
        - output_path: 輸出圖像路徑（可選）
        
        返回:
        - 標記後的圖像路徑
        """
        try:
            # 打開PDF文件
            doc = fitz.open(pdf_path)
            
            # 檢查頁碼是否有效
            if page_num < 1 or page_num > len(doc):
                return None
            
            # 獲取指定頁面
            page = doc[page_num - 1]
            
            # 將頁面轉換為圖像
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            # 將圖像保存到臨時文件
            if output_path is None:
                output_path = os.path.join(self.temp_dir, f"highlighted_page_{page_num}.png")
            
            pix.save(output_path)
            
            # 查找文本位置
            text_instances = page.search_for(text_to_highlight)
            
            if not text_instances:
                # 如果找不到完全匹配的文本，嘗試查找部分匹配
                words = text_to_highlight.split()
                for word in words:
                    if len(word) > 3:  # 忽略太短的詞
                        text_instances.extend(page.search_for(word))
            
            if text_instances:
                # 打開保存的圖像
                img = Image.open(output_path)
                draw = ImageDraw.Draw(img)
                
                # 獲取頁面尺寸
                page_width = pix.width
                page_height = pix.height
                
                # 標記文本位置
                for inst in text_instances:
                    # 將PDF坐標轉換為圖像坐標
                    x0 = inst.x0 * (300/72)
                    y0 = inst.y0 * (300/72)
                    x1 = inst.x1 * (300/72)
                    y1 = inst.y1 * (300/72)
                    
                    # 繪製半透明黃色高亮
                    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 0, 128), outline=(255, 0, 0, 255))
                
                # 保存標記後的圖像
                img.save(output_path)
            
            # 關閉PDF文件
            doc.close()
            
            return output_path
        
        except Exception as e:
            print(f"標記PDF頁面時出錯: {str(e)}")
            return None
    
    def create_side_by_side_comparison(self, original_text, edited_text, diff_html, output_path=None):
        """
        創建原始文本和編輯後文本的並排比較圖像
        
        參數:
        - original_text: 原始文本
        - edited_text: 編輯後文本
        - diff_html: 差異HTML
        - output_path: 輸出圖像路徑（可選）
        
        返回:
        - 比較圖像路徑
        """
        try:
            # 創建一個空白圖像
            width = 1200
            height = 800
            img = Image.new('RGB', (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # 嘗試加載字體
            try:
                font = ImageFont.truetype("Arial", 16)
                title_font = ImageFont.truetype("Arial", 20)
            except:
                # 如果找不到Arial字體，使用默認字體
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()
            
            # 繪製標題
            draw.text((20, 20), "原始文本", fill=(0, 0, 0), font=title_font)
            draw.text((width // 2 + 20, 20), "美編後文本", fill=(0, 0, 0), font=title_font)
            
            # 繪製分隔線
            draw.line([(width // 2, 0), (width // 2, height)], fill=(200, 200, 200), width=2)
            draw.line([(0, 60), (width, 60)], fill=(200, 200, 200), width=2)
            
            # 繪製文本
            # 將文本分行
            original_lines = self._wrap_text(original_text, font, width // 2 - 40)
            edited_lines = self._wrap_text(edited_text, font, width // 2 - 40)
            
            # 繪製原始文本
            y = 80
            for line in original_lines:
                draw.text((20, y), line, fill=(0, 0, 0), font=font)
                y += 20
            
            # 繪製編輯後文本
            y = 80
            for line in edited_lines:
                draw.text((width // 2 + 20, y), line, fill=(0, 0, 0), font=font)
                y += 20
            
            # 保存圖像
            if output_path is None:
                output_path = os.path.join(self.temp_dir, "side_by_side_comparison.png")
            
            img.save(output_path)
            
            return output_path
        
        except Exception as e:
            print(f"創建並排比較圖像時出錯: {str(e)}")
            return None
    
    def _wrap_text(self, text, font, max_width):
        """
        將文本按照最大寬度分行
        
        參數:
        - text: 要分行的文本
        - font: 字體
        - max_width: 最大寬度
        
        返回:
        - 分行後的文本列表
        """
        lines = []
        
        # 按照換行符分割
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            if not paragraph:
                lines.append('')
                continue
            
            # 分割段落為單詞
            words = paragraph.split(' ')
            current_line = words[0]
            
            for word in words[1:]:
                # 檢查添加單詞後的寬度
                test_line = current_line + ' ' + word
                # 使用getsize獲取文本寬度
                try:
                    text_width = font.getsize(test_line)[0]
                except:
                    # 如果getsize不可用，使用估計值
                    text_width = len(test_line) * 8
                
                if text_width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            
            lines.append(current_line)
        
        return lines
    
    def create_diff_visualization(self, original_text, edited_text, output_path=None):
        """
        創建差異可視化圖像
        
        參數:
        - original_text: 原始文本
        - edited_text: 編輯後文本
        - output_path: 輸出圖像路徑（可選）
        
        返回:
        - 差異可視化圖像路徑
        """
        try:
            # 使用difflib計算差異
            import difflib
            
            # 將文本分割為字符
            original_chars = list(original_text)
            edited_chars = list(edited_text)
            
            # 計算差異
            matcher = difflib.SequenceMatcher(None, original_chars, edited_chars)
            diff = []
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    diff.extend([('equal', c) for c in original_chars[i1:i2]])
                elif tag == 'delete':
                    diff.extend([('delete', c) for c in original_chars[i1:i2]])
                elif tag == 'insert':
                    diff.extend([('insert', c) for c in edited_chars[j1:j2]])
                elif tag == 'replace':
                    diff.extend([('delete', c) for c in original_chars[i1:i2]])
                    diff.extend([('insert', c) for c in edited_chars[j1:j2]])
            
            # 創建一個空白圖像
            width = 1200
            height = 600
            img = Image.new('RGB', (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # 嘗試加載字體
            try:
                font = ImageFont.truetype("Arial", 16)
                title_font = ImageFont.truetype("Arial", 20)
            except:
                # 如果找不到Arial字體，使用默認字體
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()
            
            # 繪製標題
            draw.text((20, 20), "差異可視化", fill=(0, 0, 0), font=title_font)
            
            # 繪製分隔線
            draw.line([(0, 60), (width, 60)], fill=(200, 200, 200), width=2)
            
            # 繪製差異
            x = 20
            y = 80
            line_height = 20
            
            for tag, char in diff:
                # 檢查是否需要換行
                if x > width - 40:
                    x = 20
                    y += line_height
                
                # 根據標籤設置顏色
                if tag == 'equal':
                    color = (0, 0, 0)  # 黑色
                elif tag == 'delete':
                    color = (255, 0, 0)  # 紅色
                elif tag == 'insert':
                    color = (0, 128, 0)  # 綠色
                
                # 繪製字符
                draw.text((x, y), char, fill=color, font=font)
                
                # 更新x坐標
                try:
                    x += font.getsize(char)[0]
                except:
                    # 如果getsize不可用，使用估計值
                    x += 16
            
            # 保存圖像
            if output_path is None:
                output_path = os.path.join(self.temp_dir, "diff_visualization.png")
            
            img.save(output_path)
            
            return output_path
        
        except Exception as e:
            print(f"創建差異可視化圖像時出錯: {str(e)}")
            return None
    
    def extract_pdf_page_as_image(self, pdf_path, page_num, output_path=None):
        """
        將PDF頁面提取為圖像
        
        參數:
        - pdf_path: PDF文件路徑
        - page_num: 頁碼（從1開始）
        - output_path: 輸出圖像路徑（可選）
        
        返回:
        - 圖像路徑
        """
        try:
            # 打開PDF文件
            doc = fitz.open(pdf_path)
            
            # 檢查頁碼是否有效
            if page_num < 1 or page_num > len(doc):
                return None
            
            # 獲取指定頁面
            page = doc[page_num - 1]
            
            # 將頁面轉換為圖像
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            # 將圖像保存到臨時文件
            if output_path is None:
                output_path = os.path.join(self.temp_dir, f"page_{page_num}.png")
            
            pix.save(output_path)
            
            # 關閉PDF文件
            doc.close()
            
            return output_path
        
        except Exception as e:
            print(f"提取PDF頁面為圖像時出錯: {str(e)}")
            return None
    
    def highlight_text_in_image(self, image_path, text_to_highlight, output_path=None):
        """
        在圖像中標記指定文本
        
        參數:
        - image_path: 圖像路徑
        - text_to_highlight: 要標記的文本
        - output_path: 輸出圖像路徑（可選）
        
        返回:
        - 標記後的圖像路徑
        """
        try:
            # 使用OCR查找文本位置
            import pytesseract
            from PIL import Image
            
            # 打開圖像
            img = Image.open(image_path)
            
            # 使用pytesseract獲取文本位置
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='chi_tra+eng')
            
            # 創建繪圖對象
            draw = ImageDraw.Draw(img)
            
            # 查找文本位置
            found = False
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i]
                if text and (text in text_to_highlight or text_to_highlight in text):
                    # 獲取文本位置
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    # 繪製半透明黃色高亮
                    draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 0, 128), outline=(255, 0, 0, 255))
                    found = True
            
            # 如果沒有找到文本，嘗試使用模糊匹配
            if not found:
                # 將文本分割為單詞
                words = text_to_highlight.split()
                for word in words:
                    if len(word) > 3:  # 忽略太短的詞
                        for i in range(len(ocr_data['text'])):
                            text = ocr_data['text'][i]
                            if text and word in text:
                                # 獲取文本位置
                                x = ocr_data['left'][i]
                                y = ocr_data['top'][i]
                                w = ocr_data['width'][i]
                                h = ocr_data['height'][i]
                                
                                # 繪製半透明黃色高亮
                                draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 0, 128), outline=(255, 0, 0, 255))
                                found = True
            
            # 保存標記後的圖像
            if output_path is None:
                output_path = os.path.join(self.temp_dir, "highlighted_image.png")
            
            img.save(output_path)
            
            return output_path if found else None
        
        except Exception as e:
            print(f"在圖像中標記文本時出錯: {str(e)}")
            return None
    
    def highlight_text_with_qwen(self, image_path, text_to_highlight, api_key=None, api_url=None, output_path=None):
        """
        使用Qwen API在圖像中標記指定文本
        
        參數:
        - image_path: 圖像路徑
        - text_to_highlight: 要標記的文本
        - api_key: Qwen API密鑰（可選）
        - api_url: Qwen API URL（可選）
        - output_path: 輸出圖像路徑（可選）
        
        返回:
        - 標記後的圖像路徑
        """
        try:
            # 使用免費API
            use_free_api = api_url is None and (api_key is None or api_key.strip() == "")
            free_api_url = "https://api.qwen-2.com/v1/vision/text-detection"
            
            # 讀取圖像
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # 將圖像轉換為base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            
            if use_free_api:
                # 使用免費API
                headers = {
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "qwen-vl-plus",
                    "input": {
                        "image": image_base64,
                        "text": f"請檢測圖像中的文本，特別是找出與以下文本相似的部分：{text_to_highlight}"
                    }
                }
                
                response = requests.post(free_api_url, headers=headers, json=payload)
            else:
                # 使用自定義API
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "qwen-vl-plus",
                    "input": {
                        "image": image_base64,
                        "text": f"請檢測圖像中的文本，特別是找出與以下文本相似的部分：{text_to_highlight}"
                    }
                }
                
                response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # 解析結果
                if use_free_api:
                    text_detection = result.get("output", {}).get("text", "")
                else:
                    text_detection = result.get("output", {}).get("text", "")
                
                # 解析檢測到的文本位置
                text_positions = []
                try:
                    # 嘗試從回應中提取文本位置
                    if "我在圖像中檢測到以下文本" in text_detection:
                        # 使用正則表達式提取位置信息
                        import re
                        matches = re.findall(r'"([^"]+)".*?位置：\((\d+),(\d+),(\d+),(\d+)\)', text_detection)
                        for match in matches:
                            text, x1, y1, x2, y2 = match
                            text_positions.append({
                                "text": text,
                                "box": [int(x1), int(y1), int(x2), int(y2)]
                            })
                exce
(Content truncated due to size limit. Use line ranges to read in chunks)