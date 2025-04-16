import requests
import base64
from PIL import Image
import io
import json
import os

class CustomAI:
    """
    使用自定義API進行AI功能的類
    """
    def __init__(self, api_key=None, api_url=None, model_name=None):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name or "gpt-4o"  # 默認模型
        
        # 使用免費API
        self.use_free_api = api_url is None and (api_key is None or api_key.strip() == "")
        self.free_api_url = "https://api.qwen-2.com/v1/chat/completions"
    
    def is_available(self):
        """檢查API是否可用"""
        return True  # 始終可用，因為我們有免費API選項
    
    def semantic_comparison(self, text1, text2):
        """使用生成式AI進行語義比對"""
        try:
            if self.use_free_api:
                return self._semantic_comparison_free_api(text1, text2)
            else:
                return self._semantic_comparison_custom_api(text1, text2)
        
        except Exception as e:
            return 0.0, f"語義比對時出錯: {str(e)}"
    
    def _semantic_comparison_free_api(self, text1, text2):
        """使用免費API進行語義比對"""
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen2.5-7b-instruct",
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
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        try:
            response = requests.post(self.free_api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
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
            return 0.0, f"免費API請求出錯: {str(e)}"
    
    def _semantic_comparison_custom_api(self, text1, text2):
        """使用自定義API進行語義比對"""
        # 檢測API類型
        if "openai" in self.api_url.lower():
            return self._semantic_comparison_openai(text1, text2)
        elif "anthropic" in self.api_url.lower():
            return self._semantic_comparison_anthropic(text1, text2)
        elif "dashscope" in self.api_url.lower() or "aliyun" in self.api_url.lower():
            return self._semantic_comparison_qwen(text1, text2)
        else:
            # 默認使用OpenAI格式
            return self._semantic_comparison_openai(text1, text2)
    
    def _semantic_comparison_openai(self, text1, text2):
        """使用OpenAI API進行語義比對"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
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
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
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
            return 0.0, f"OpenAI API請求出錯: {str(e)}"
    
    def _semantic_comparison_anthropic(self, text1, text2):
        """使用Anthropic API進行語義比對"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
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
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["content"][0]["text"]
                
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
            return 0.0, f"Anthropic API請求出錯: {str(e)}"
    
    def _semantic_comparison_qwen(self, text1, text2):
        """使用Qwen API進行語義比對"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
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
                "result_format": "message",
                "temperature": 0.1,
                "max_tokens": 100
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            
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
            return 0.0, f"Qwen API請求出錯: {str(e)}"
    
    def analyze_comparison_results(self, original_text, edited_text, comparison_results):
        """分析比對結果，提供智能見解"""
        try:
            if self.use_free_api:
                return self._analyze_comparison_results_free_api(original_text, edited_text, comparison_results)
            else:
                return self._analyze_comparison_results_custom_api(original_text, edited_text, comparison_results)
        
        except Exception as e:
            return f"分析比對結果時出錯: {str(e)}"
    
    def _analyze_comparison_results_free_api(self, original_text, edited_text, comparison_results):
        """使用免費API分析比對結果"""
        headers = {
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
            "model": "qwen2.5-7b-instruct",
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
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.free_api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                return analysis
            else:
                return f"API請求失敗: {response.status_code} - {response.text}"
        except Exception as e:
            return f"免費API請求出錯: {str(e)}"
    
    def _analyze_comparison_results_custom_api(self, original_text, edited_text, comparison_results):
        """使用自定義API分析比對結果"""
        # 檢測API類型
        if "openai" in self.api_url.lower():
            return self._analyze_comparison_results_openai(original_text, edited_text, comparison_results)
        elif "anthropic" in self.api_url.lower():
            return self._analyze_comparison_results_anthropic(original_text, edited_text, comparison_results)
        elif "dashscope" in self.api_url.lower() or "aliyun" in self.api_url.lower():
            return self._analyze_comparison_results_qwen(original_text, edited_text, comparison_results)
        else:
            # 默認使用OpenAI格式
            return self._analyze_comparison_results_openai(original_text, edited_text, comparison_results)
    
    def _analyze_comparison_results_openai(self, original_text, edited_text, comparison_results):
        """使用OpenAI API分析比對結果"""
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
            "model": self.model_name,
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
                 """} ]}
