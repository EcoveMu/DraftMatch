import requests
import base64
from PIL import Image
import io
import json
import os

class QwenOCR:
    """
    使用Qwen API進行OCR文本識別的類
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.model = "qwen-vl-plus"  # 使用Qwen-VL-Plus模型，支持視覺理解
    
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
            response = requests.post(self.base_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result["output"]["choices"][0]["message"]["content"]
                return extracted_text
            else:
                return f"API請求失敗: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"提取文本時出錯: {str(e)}"
    
    def extract_text_from_pdf_page(self, pdf_path, page_num):
        """從PDF頁面提取文本"""
        try:
            import fitz  # PyMuPDF
            
            # 將PDF頁面轉換為圖像
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
            
            # 保存為臨時圖像文件
            temp_img_path = f"/tmp/pdf_page_{page_num}.png"
            pix.save(temp_img_path)
            
            # 使用OCR提取文本
            text = self.extract_text_from_image(temp_img_path)
            
            # 刪除臨時文件
            os.remove(temp_img_path)
            
            return text
        
        except Exception as e:
            return f"從PDF提取文本時出錯: {str(e)}"
    
    def extract_tables_from_image(self, image_path):
        """從圖像中提取表格"""
        if not self.is_available():
            return "API密鑰未設置，無法使用Qwen表格提取"
        
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
            response = requests.post(self.base_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                extracted_content = result["output"]["choices"][0]["message"]["content"]
                
                # 嘗試從回應中提取JSON
                try:
                    # 尋找JSON部分
                    json_start = extracted_content.find("```json")
                    json_end = extracted_content.rfind("```")
                    
                    if json_start != -1 and json_end != -1:
                        json_content = extracted_content[json_start+7:json_end].strip()
                        tables = json.loads(json_content)
                        return tables
                    else:
                        # 嘗試直接解析整個內容
                        tables = json.loads(extracted_content)
                        return tables
                except:
                    # 如果無法解析JSON，返回原始文本
                    return {"raw_response": extracted_content}
            else:
                return {"error": f"API請求失敗: {response.status_code} - {response.text}"}
        
        except Exception as e:
            return {"error": f"提取表格時出錯: {str(e)}"}

# 測試函數
def test_qwen_ocr(api_key, image_path):
    """測試Qwen OCR功能"""
    ocr = QwenOCR(api_key)
    result = ocr.extract_text_from_image(image_path)
    print(f"提取的文本: {result}")
    return result

# 主函數
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python qwen_api.py <api_key> <image_path>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    image_path = sys.argv[2]
    test_qwen_ocr(api_key, image_path)
