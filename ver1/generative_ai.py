import requests
import base64
from PIL import Image
import io
import json
import os
import tempfile
import fitz  # PyMuPDF
import streamlit as st

class QwenAI:
    """
    使用Qwen API進行AI功能的類
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
            # 將PDF頁面轉換為圖像
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
            
            # 保存為臨時圖像文件
            temp_dir = tempfile.mkdtemp()
            temp_img_path = os.path.join(temp_dir, f"pdf_page_{page_num}.png")
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
    
    def analyze_comparison_results(self, original_text, edited_text, comparison_results):
        """分析比對結果，提供智能見解"""
        if not self.is_available():
            return "API密鑰未設置，無法使用Qwen分析功能"
        
        try:
            # 準備API請求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 將比對結果轉換為簡化的格式
            simplified_results = {
                "total_paragraphs": comparison_results["statistics"]["total_paragraphs"],
                "similar_paragraphs": comparison_results["statistics"]["similar_paragraphs"],
                "different_paragraphs": comparison_results["statistics"]["different_paragraphs"],
                "examples": []
            }
            
            # 添加一些差異示例
            for result in comparison_results["paragraph_results"][:3]:
                if not result["is_similar"]:
                    simplified_results["examples"].append({
                        "original": result["original_text"],
                        "edited": result["matched_text"],
                        "similarity": result["exact_similarity"]
                    })
            
            payload = {
                "model": "qwen-max",  # 使用純文本模型
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一個專業的文本比對分析專家，請分析原始文本和編輯後文本的差異，提供專業見解。"
                        },
                        {
                            "role": "user",
                            "content": f"""
                            請分析以下文本比對結果，並提供專業見解：
                            
                            比對統計信息：
                            - 總段落數：{simplified_results['total_paragraphs']}
                            - 相似段落數：{simplified_results['similar_paragraphs']}
                            - 不同段落數：{simplified_results['different_paragraphs']}
                            
                            差異示例：
                            {json.dumps(simplified_results['examples'], ensure_ascii=False, indent=2)}
                            
                            請提供以下分析：
                            1. 主要差異類型（例如：添加內容、刪除內容、修改格式等）
                            2. 差異的嚴重程度評估
                            3. 可能需要特別注意的地方
                            4. 改進建議
                            
                            請以簡潔專業的語言回答，不要添加無關內容。
                            """
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
                analysis = result["output"]["choices"][0]["message"]["content"]
                return analysis
            else:
                return f"API請求失敗: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"分析比對結果時出錯: {str(e)}"
    
    def semantic_comparison(self, text1, text2):
        """使用生成式AI進行語義比對"""
        if not self.is_available():
            return 0.0, "API密鑰未設置，無法使用Qwen語義比對"
        
        try:
            # 準備API請求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "qwen-max",  # 使用純文本模型
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一個專業的文本比對專家，請比較兩段文本的語義相似度，並給出0到1之間的分數，1表示完全相同，0表示完全不同。"
                        },
                        {
                            "role": "user",
                            "content": f"""
                            請比較以下兩段文本的語義相似度：
                            
                            文本1：
                            {text1}
                            
                            文本2：
                            {text2}
                            
                            請給出一個0到1之間的分數，表示這兩段文本的語義相似度。1表示完全相同，0表示完全不同。
                            只需要回覆一個數字，不要添加任何其他內容。
                            """
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
                response_text = result["output"]["choices"][0]["message"]["content"]
                
                # 嘗試從回應中提取數字
                try:
                    # 使用正則表達式提取數字
                    import re
                    match = re.search(r'(\d+\.\d+|\d+)', response_text)
                    if match:
                        similarity = float(match.group(1))
                        # 確保相似度在0到1之間
                        similarity = max(0.0, min(1.0, similarity))
                        return similarity, None
                    else:
                        return 0.5, f"無法從回應中提取數字: {response_text}"
                except Exception as e:
                    return 0.5, f"處理回應時出錯: {str(e)}"
            else:
                return 0.0, f"API請求失敗: {response.status_code} - {response.text}"
        
        except Exception as e:
            return 0.0, f"語義比對時出錯: {str(e)}"
    
    def generate_summary_report(self, comparison_results):
        """生成比對結果摘要報告"""
        if not self.is_available():
            return "API密鑰未設置，無法使用Qwen報告生成功能"
        
        try:
            # 準備API請求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 將比對結果轉換為簡化的格式
            stats = comparison_results["statistics"]
            
            # 計算相似度百分比
            if stats['total_paragraphs'] > 0:
                paragraph_similarity = stats['similar_paragraphs'] / stats['total_paragraphs'] * 100
            else:
                paragraph_similarity = 0
            
            if stats['total_tables'] > 0:
                table_similarity = stats['similar_tables'] / stats['total_tables'] * 100
            else:
                table_similarity = 0
            
            total_items = stats['total_paragraphs'] + stats['total_tables']
            similar_items = stats['similar_paragraphs'] + stats['similar_tables']
            
            if total_items > 0:
                overall_similarity = similar_items / total_items * 100
            else:
                overall_similarity = 0
            
            # 獲取一些差異示例
            diff_examples = []
            for result in comparison_results["paragraph_results"]:
                if not result["is_similar"]:
                    diff_examples.append({
                        "original": result["original_text"],
                        "edited": result["matched_text"],
                        "similarity": result["exact_similarity"]
                    })
                    if len(diff_examples) >= 3:
                        break
            
            payload = {
                "model": "qwen-max",  # 使用純文本模型
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一個專業的文本比對報告生成專家，請根據比對結果生成簡潔明了的摘要報告。"
                        },
                        {
                            "role": "user",
                            "content": f"""
                            請根據以下比對結果生成摘要報告：
                            
                            統計信息：
                            - 總段落數：{stats['total_paragraphs']}
                            - 相似段落數：{stats['similar_paragraphs']}
                            - 不同段落數：{stats['different_paragraphs']}
                            - 總表格數：{stats['total_tables']}
                            - 相似表格數：{stats['similar_tables']}
                            - 不同表格數：{stats['different_tables']}
                            - 段落相似度：{paragraph_similarity:.2f}%
                            - 表格相似度：{table_similarity:.2f}%
                            - 整體相似度：{overall_similarity:.2f}%
                            
                            差異示例：
                            {json.dumps(diff_examples, ensure_ascii=False, indent=2)}
                            
                            請生成一份簡潔明了的摘要報告，包括：
                            1. 整體比對結果概述
                            2. 主要差異類型和分布
                            3. 需要特別注意的地方
                            4. 改進建議
                            
                            請使用專業的語言，並以markdown格式輸出。
                            """
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
                report = result["output"]["choices"][0]["message"]["content"]
                return report
            else:
                return f"API請求失敗: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"生成報告時出錯: {str(e)}"

# 測試函數
def test_qwen_ai(api_key):
    """測試Qwen AI功能"""
    ai = QwenAI(api_key)
    
   
(Content truncated due to size limit. Use line ranges to read in chunks)