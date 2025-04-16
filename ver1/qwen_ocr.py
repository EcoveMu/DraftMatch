import requests
import base64
import os
import tempfile
import fitz  # PyMuPDF
import io
from PIL import Image

class QwenOCR:
    """
    使用Qwen API進行OCR的類
    """
    def __init__(self, api_key=None, custom_api_url=None):
        self.api_key = api_key
        self.custom_api_url = custom_api_url
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.model = "qwen-vl-plus"  # 使用Qwen-VL-Plus模型， 支持視覺理解
    
    def is_available(self):
        """檢查API是否可用"""
        return self.api_key is not None and len(self.api_key) > 0
    
    def extract_text_from_image(self, image_path):
        """從圖像中提取文本"""
        if not self.is_available():
            return "API密鑰未設置，無法使用Qwen OCR"
        
        try:
            # 讀取圖像並轉換為base64
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # 準備API請求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一個專業的OCR助手，請提取圖像中的所有文本內容，保持原始格式。對於表格，請以結構化方式提取內容。"
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": base64_image
                                },
                                {
                                    "text": "請提取這個圖像中的所有文本內容，包括表格。保持原始格式，不要添加任何解釋。"
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message"
                }
            }
            
            # 發送API請求
            url = self.custom_api_url if self.custom_api_url else self.base_url
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result["output"]["choices"][0]["message"]["content"]
                return extracted_text
            else:
                return f"API請求失敗: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"提取文本時出錯: {str(e)}"
    
    def extract_text_from_pdf(self, pdf_file):
        """從PDF文件中提取文本"""
        if not self.is_available():
            return "API密鑰未設置，無法使用Qwen OCR"
        
        try:
            # 保存上傳的PDF文件到臨時文件
            temp_dir = tempfile.mkdtemp()
            temp_pdf_path = os.path.join(temp_dir, "temp.pdf")
            
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            # 打開PDF文件
            doc = fitz.open(temp_pdf_path)
            
            # 提取每一頁的文本
            results = {}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # 將圖像保存到臨時文件
                img_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                pix.save(img_path)
                
                # 使用OCR提取文本
                text = self.extract_text_from_image(img_path)
                
                # 保存結果
                results[page_num + 1] = text
            
            # 關閉PDF文件
            doc.close()
            
            return results
        
        except Exception as e:
            return f"從PDF提取文本時出錯: {str(e)}"
    
    def extract_tables_from_pdf(self, pdf_file):
        """從PDF文件中提取表格"""
        if not self.is_available():
            return "API密鑰未設置，無法使用Qwen表格提取"
        
        try:
            # 保存上傳的PDF文件到臨時文件
            temp_dir = tempfile.mkdtemp()
            temp_pdf_path = os.path.join(temp_dir, "temp.pdf")
            
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            # 打開PDF文件
            doc = fitz.open(temp_pdf_path)
            
            # 提取每一頁的表格
            results = {}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # 將圖像保存到臨時文件
                img_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                pix.save(img_path)
                
                # 使用OCR提取表格
                tables = self.extract_tables_from_image(img_path)
                
                # 保存結果
                if tables:
                    results[page_num + 1] = tables
            
            # 關閉PDF文件
            doc.close()
            
            return results
        
        except Exception as e:
            return f"從PDF提取表格時出錯: {str(e)}"
    
    def extract_tables_from_image(self, image_path):
        """從圖像中提取表格"""
        if not self.is_available():
            return None
        
        try:
            # 讀取圖像並轉換為base64
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # 準備API請求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一個專業的表格提取助手，請從圖像中提取所有表格內容，並以JSON格式返回。"
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": base64_image
                                },
                                {
                                    "text": "請從這個圖像中提取所有表格，並以JSON格式返回。每個表格應該是一個二維數組，表示行和列。如果沒有表格，請返回空數組。"
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message"
                }
            }
            
            # 發送API請求
            url = self.custom_api_url if self.custom_api_url else self.base_url
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                extracted_content = result["output"]["choices"][0]["message"]["content"]
                
                # 嘗試從回應中提取JSON
                try:
                    # 尋找JSON部分
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', extracted_content, re.DOTALL)
                    
                    if json_match:
                        json_content = json_match.group(1)
                        import json
                        tables = json.loads(json_content)
                        return tables
                    else:
                        # 嘗試直接解析整個內容
                        import json
                        tables = json.loads(extracted_content)
                        return tables
                except:
                    # 如果無法解析JSON，返回None
                    return None
            else:
                return None
        
        except Exception as e:
            return None
