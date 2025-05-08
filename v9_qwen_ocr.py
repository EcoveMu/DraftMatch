import requests
import base64
import os
import tempfile
import json
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image, ImageDraw
import io

class QwenOCR:
    """
    使用Qwen API進行OCR文本識別的類。
    如果未提供 api_key，將自動使用免費API；否則使用提供的API端點。
    可選地可以透過 api_url 指定自定義的Qwen API端點。
    """
    def __init__(self, api_key=None, api_url=None):
        self.api_key = api_key
        # 官方API的基礎URL（默認值）
        self.api_url = api_url or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.model = "qwen-vl-plus"  # 使用Qwen-VL-Plus模型，支持視覺理解
        # 判斷是否使用免費API
        self.use_free_api = api_url is None and (api_key is None or api_key.strip() == "")
        self.free_api_url = "https://api.qwen-2.com/v1/chat/completions"
    
    def is_available(self):
        """檢查API是否可用。若未提供API密鑰則使用免費API。"""
        # 始終視為可用，因為如果沒有設置api_key則會自動使用免費API
        return True
    
    def extract_text_from_image(self, image_path):
        """從圖像中提取文本。"""
        try:
            # 讀取圖像並轉換為base64編碼
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            if self.use_free_api:
                # 使用免費API請求OCR
                return self._extract_text_free_api(base64_image)
            else:
                # 使用官方API請求OCR
                return self._extract_text_official_api(base64_image)
        except Exception as e:
            return f"提取文本時出錯: {str(e)}"
    
    def _extract_text_official_api(self, base64_image):
        """使用官方API提取文本內容。"""
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
                            {"image": base64_image},
                            {"text": "請提取這個圖像中的所有文本內容，包括表格。保持原始格式，不要添加任何解釋。"}
                        ]
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        # 發送API請求
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            extracted_text = result["output"]["choices"][0]["message"]["content"]
            return extracted_text
        else:
            return f"API請求失敗: {response.status_code} - {response.text}"
    
    def _extract_text_free_api(self, base64_image):
        """使用免費API提取文本內容。"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "qwen2.5-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一個專業的OCR助手，請提取圖像中的所有文本內容，保持原始格式。對於表格，請以結構化方式提取內容。"
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
                            "text": "請提取這個圖像中的所有文本內容，包括表格。保持原始格式，不要添加任何解釋。"
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
                extracted_text = result["choices"][0]["message"]["content"]
                return extracted_text
            else:
                return f"免費API請求失敗: {response.status_code} - {response.text}"
        except Exception as e:
            return f"免費API請求出錯: {str(e)}"
    
    def extract_text_from_pdf_page(self, pdf_path, page_num):
        """從PDF文件的指定頁面提取文本。"""
        try:
            # 將PDF頁面轉換為圖像（300 DPI）
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
        """從圖像中提取表格內容。"""
        try:
            # 讀取圖像並轉換為base64
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            if self.use_free_api:
                return self._extract_tables_free_api(base64_image)
            else:
                return self._extract_tables_official_api(base64_image)
        except Exception as e:
            return {"error": f"提取表格時出錯: {str(e)}"}
    
    def _extract_tables_official_api(self, base64_image):
        """使用官方API提取表格內容，返回JSON格式。"""
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
                            {"image": base64_image},
                            {"text": "請從這個圖像中提取所有表格，並以JSON格式返回。每個表格應該是一個二維數組，表示行和列。如果沒有表格，請返回空數組。"}
                        ]
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            extracted_content = result["output"]["choices"][0]["message"]["content"]
            # 嘗試從回應中解析JSON
            try:
                json_start = extracted_content.find("```json")
                json_end = extracted_content.rfind("```")
                if json_start != -1 and json_end != -1:
                    json_content = extracted_content[json_start+7:json_end].strip()
                    tables = json.loads(json_content)
                    return tables
                else:
                    tables = json.loads(extracted_content)
                    return tables
            except:
                return {"raw_response": extracted_content}
        else:
            return {"error": f"API請求失敗: {response.status_code} - {response.text}"}
    
    def _extract_tables_free_api(self, base64_image):
        """使用免費API提取表格內容，返回JSON格式。"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "qwen2.5-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一個專業的表格提取助手，請從圖像中提取所有表格內容，並以JSON格式返回。"
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
                            "text": "請從這個圖像中提取所有表格，並以JSON格式返回。每個表格應該是一個二維數組，表示行和列。如果沒有表格，請返回空數組。"
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
                extracted_content = result["choices"][0]["message"]["content"]
                try:
                    json_start = extracted_content.find("```json")
                    json_end = extracted_content.rfind("```")
                    if json_start != -1 and json_end != -1:
                        json_content = extracted_content[json_start+7:json_end].strip()
                        tables = json.loads(json_content)
                        return tables
                    else:
                        tables = json.loads(extracted_content)
                        return tables
                    # 如果無法解析JSON，返回原始文本（已在上方嘗試）
                except:
                    return {"raw_response": extracted_content}
            else:
                return {"error": f"免費API請求失敗: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": f"免費API請求出錯: {str(e)}"}
    
    def highlight_text_in_image(self, image_path, text_to_highlight, output_path=None):
        """在圖像中標記指定的文本位置，並返回標記後的圖像路徑。"""
        try:
            # 將圖像讀取並轉換為base64
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            # 構造請求以獲取文本坐標
            if self.use_free_api:
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": "qwen2.5-7b-instruct",
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一個專業的OCR助手，請找出圖像中指定文本的位置。"
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
                                    "text": f"請找出圖像中以下文本的位置，並以JSON格式返回坐標 (x1, y1, x2, y2)，表示包含該文本的矩形框左上角和右下角的坐標：\\n\\n{text_to_highlight}\\n\\n如果找不到完全匹配的文本，請找出最相似的部分。"
                                }
                            ]
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4000
                }
                response = requests.post(self.free_api_url, headers=headers, json=payload)
            else:
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
                                "content": "你是一個專業的OCR助手，請找出圖像中指定文本的位置。"
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"image": base64_image},
                                    {"text": f"請找出圖像中以下文本的位置，並以JSON格式返回坐標 (x1, y1, x2, y2)，表示包含該文本的矩形框左上角和右下角的坐標：\\n\\n{text_to_highlight}\\n\\n如果找不到完全匹配的文本，請找出最相似的部分。"}
                                ]
                            }
                        ]
                    },
                    "parameters": {"result_format": "message"}
                }
                response = requests.post(self.api_url, headers=headers, json=payload)
            # 解析API回應
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"] if self.use_free_api else result["output"]["choices"][0]["message"]["content"]
                try:
                    # 尋找JSON格式的坐標
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    if json_start != -1 and json_end != -1:
                        json_content = content[json_start:json_end]
                        coordinates = json.loads(json_content)
                        # 打開圖像並繪製矩形
                        image = Image.open(image_path)
                        draw = ImageDraw.Draw(image)
                        if isinstance(coordinates, dict) and "coordinates" in coordinates:
                            coords = coordinates["coordinates"]
                            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                        elif isinstance(coordinates, list) and len(coordinates) == 4:
                            x1, y1, x2, y2 = coordinates
                        elif isinstance(coordinates, dict) and "x1" in coordinates:
                            x1 = coordinates["x1"]; y1 = coordinates["y1"]; x2 = coordinates["x2"]; y2 = coordinates["y2"]
                        else:
                            # 如果無法解析，默認標記整個圖像
                            x1, y1, x2, y2 = 0, 0, image.width, image.height
                        # 繪製紅色矩形框標記文本位置
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                        # 保存標記後的圖像
                        if output_path:
                            image.save(output_path)
                        else:
                            temp_dir = tempfile.mkdtemp()
                            output_path = os.path.join(temp_dir, "highlighted_image.png")
                            image.save(output_path)
                        return output_path
                    else:
                        return None
                except Exception as e:
                    return None
            else:
                return None
        except Exception as e:
            return None
