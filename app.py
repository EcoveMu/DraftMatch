import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
import time
import json
import io
import re
import difflib
import base64
import shutil
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF

# 導入本地模組
from enhanced_extraction import extract_and_process_documents
from qwen_api import QwenOCR
from custom_ai import CustomAI

# 檢查sentence-transformers是否可用
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# 檢查是否可以導入生成式AI模組
try:
    from qwen_ai import QwenAI
    QWEN_AI_AVAILABLE = True
except ImportError:
    QWEN_AI_AVAILABLE = False

# 設置頁面配置
st.set_page_config(
    page_title="期刊比對系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
def load_css():
    css = """
    <style>
        .diff-removed {
            background-color: #ffcccc;
            text-decoration: line-through;
            color: black;
        }
        .diff-added {
            background-color: #ccffcc;
            color: black;
        }
        .diff-char-removed {
            background-color: #ffcccc;
            text-decoration: line-through;
            display: inline;
            color: black;
        }
        .diff-char-added {
            background-color: #ccffcc;
            display: inline;
            color: black;
        }
        .comparison-result {
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            color: black;
        }
        .similar {
            border-left: 5px solid green;
        }
        .different {
            border-left: 5px solid red;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: black;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e6f0ff;
            border-bottom: 2px solid #4c83ff;
        }
        .highlight {
            background-color: yellow;
            color: black;
        }
        .table-container {
            overflow-x: auto;
        }
        .table-container table {
            width: 100%;
            border-collapse: collapse;
        }
        .table-container th, .table-container td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            color: black;
        }
        .table-container th {
            background-color: #f2f2f2;
        }
        .table-container tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .diff-warning {
            background-color: #fff3cd;
            color: black;
        }
        .diff-error {
            background-color: #f8d7da;
            color: black;
        }
        .summary-card {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            color: black;
        }
        .summary-card h3 {
            margin-top: 0;
            color: #333;
        }
        .metric-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .metric-box {
            background-color: white;
            border-radius: 5px;
            padding: 10px;
            margin: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            flex: 1;
            min-width: 120px;
            color: black;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 5px 0;
            color: black;
        }
        .metric-label {
            font-size: 14px;
            color: #333;
        }
        .ai-analysis {
            background-color: #f0f7ff;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            border-left: 5px solid #4c83ff;
            color: black;
        }
        .pdf-preview {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            background-color: white;
        }
        .pdf-preview img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        .pdf-preview-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown li {
            color: black !important;
        }
        .stText {
            color: black !important;
        }
        .stTextInput > div > div > input {
            color: black !important;
        }
        .stSelectbox > div > div > div {
            color: black !important;
        }
        .stSlider > div > div > div {
            color: black !important;
        }
        .stCheckbox > div > div > label {
            color: black !important;
        }
        .stExpander > div > div > div > div > p {
            color: black !important;
        }
        .stExpander > div > div > div > div > div > p {
            color: black !important;
        }
        .table-tab {
            margin-top: 20px;
        }
        .ai-model-section {
            background-color: #E3F2FD;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .success-message {
            color: #2E7D32;
            background-color: #E8F5E9;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .error-message {
            color: #C62828;
            background-color: #FFEBEE;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .diff-section {
            margin-top: 10px;
            margin-bottom: 10px;
            padding: 10px;
            border: 1px solid #E0E0E0;
            border-radius: 5px;
        }
        .diff-navigation {
            margin-top: 10px;
            margin-bottom: 10px;
            text-align: center;
        }
        .diff-count {
            font-weight: bold;
            margin-left: 10px;
            margin-right: 10px;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 檢查Java是否安裝
def is_java_installed():
    try:
        result = os.system("java -version > /dev/null 2>&1")
        return result == 0
    except:
        return False

# 檢查EasyOCR是否可用
def is_easyocr_available():
    try:
        import easyocr
        return True
    except ImportError:
        return False

# 檢查tabula-py是否可用
def is_tabula_available():
    if not is_java_installed():
        return False
    try:
        import tabula
        return True
    except ImportError:
        return False

# 檢查sentence-transformers是否可用
def is_sentence_transformers_available():
    return SENTENCE_TRANSFORMERS_AVAILABLE

# 加載語義模型
@st.cache_resource
def load_semantic_model():
    if is_sentence_transformers_available():
        try:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            return model
        except Exception as e:
            st.error(f"加載語義模型失敗: {e}")
            return None
    return None

# 生成式AI模型類
class GenerativeAI:
    def __init__(self, model_name, api_key=None, api_url=None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.is_available = self._check_availability()
        self.custom_ai = None
        
        # 如果是自定義AI，初始化CustomAI實例
        if self.model_name == "自定義AI":
            self.custom_ai = CustomAI(api_key=api_key, api_url=api_url)
    
    def _check_availability(self):
        """檢查模型是否可用"""
        if self.model_name == "自定義AI":
            # 自定義AI始終可用，因為它有免費API選項
            return True
        elif self.model_name in ["BERT多語言模型", "MPNet中文模型", "RoBERTa中文模型"]:
            # 檢查本地模型
            try:
                from transformers import AutoModel, AutoTokenizer
                return True
            except ImportError:
                st.warning(f"未安裝transformers庫，{self.model_name}不可用")
                return False
        elif self.model_name in ["OpenAI API", "Anthropic API", "Gemini API", "Qwen API"]:
            # 檢查API模型
            if self.api_key is not None and len(self.api_key) > 0:
                return True
            else:
                st.warning(f"{self.model_name}需要API Key才能使用")
                return False
        return False
    
    def match_paragraphs(self, source_paragraphs, target_paragraphs):
        """使用生成式AI匹配段落"""
        if not self.is_available:
            st.warning(f"{self.model_name}不可用，將使用基本匹配算法")
            return None
        
        try:
            if self.model_name == "自定義AI":
                return self._match_with_custom_ai(source_paragraphs, target_paragraphs)
            elif self.model_name == "BERT多語言模型":
                return self._match_with_bert(source_paragraphs, target_paragraphs)
            elif self.model_name == "MPNet中文模型":
                return self._match_with_mpnet(source_paragraphs, target_paragraphs)
            elif self.model_name == "RoBERTa中文模型":
                return self._match_with_roberta(source_paragraphs, target_paragraphs)
            elif self.model_name == "OpenAI API":
                return self._match_with_openai(source_paragraphs, target_paragraphs)
            elif self.model_name == "Anthropic API":
                return self._match_with_anthropic(source_paragraphs, target_paragraphs)
            elif self.model_name == "Gemini API":
                return self._match_with_gemini(source_paragraphs, target_paragraphs)
            elif self.model_name == "Qwen API":
                return self._match_with_qwen(source_paragraphs, target_paragraphs)
        except Exception as e:
            st.error(f"{self.model_name}匹配失敗: {str(e)}")
            st.info("將使用基本匹配算法作為替代")
            return None
        
        return None
    
    def _match_with_custom_ai(self, source_paragraphs, target_paragraphs):
        """使用自定義AI匹配段落"""
        if not self.custom_ai:
            return None
        
        # 計算相似度矩陣
        similarity_matrix = np.zeros((len(source_paragraphs), len(target_paragraphs)))
        for i, source_para in enumerate(source_paragraphs):
            for j, target_para in enumerate(target_paragraphs):
                similarity, error = self.custom_ai.semantic_comparison(source_para['content'], target_para['content'])
                if error:
                    st.warning(f"計算相似度時出錯: {error}")
                similarity_matrix[i, j] = similarity
        
        # 使用貪婪算法找到最佳匹配
        matches = []
        used_target_indices = set()
        
        # 為每個source段落找到最佳匹配的target段落
        for i in range(len(source_paragraphs)):
            best_similarity = -1
            best_j = -1
            
            # 找到最相似的未使用的target段落
            for j in range(len(target_paragraphs)):
                if j not in used_target_indices and similarity_matrix[i, j] > best_similarity:
                    best_similarity = similarity_matrix[i, j]
                    best_j = j
            
            # 如果找到匹配，則添加到結果中
            if best_j != -1:
                matches.append({
                    "doc1_index": i,
                    "doc2_index": best_j,
                    "similarity": best_similarity
                })
                used_target_indices.add(best_j)
            else:
                # 如果沒有未使用的target段落，則選擇最相似的段落
                best_j = np.argmax(similarity_matrix[i])
                matches.append({
                    "doc1_index": i,
                    "doc2_index": best_j,
                    "similarity": similarity_matrix[i, best_j]
                })
        
        return matches
    
    def _match_with_bert(self, source_paragraphs, target_paragraphs):
        """使用BERT多語言模型匹配段落"""
        try:
            from transformers import BertModel, BertTokenizer
            import torch
            
            # 加載模型
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            model = BertModel.from_pretrained('bert-base-multilingual-cased')
            
            # 計算嵌入
            source_embeddings = []
            for para in source_paragraphs:
                inputs = tokenizer(para['content'], return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                source_embeddings.append(embeddings.squeeze().numpy())
            
            target_embeddings = []
            for para in target_paragraphs:
                inputs = tokenizer(para['content'], return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                target_embeddings.append(embeddings.squeeze().numpy())
            
            # 計算相似度矩陣
            similarity_matrix = np.zeros((len(source_paragraphs), len(target_paragraphs)))
            for i, source_emb in enumerate(source_embeddings):
                for j, target_emb in enumerate(target_embeddings):
                    similarity = np.dot(source_emb, target_emb) / (np.linalg.norm(source_emb) * np.linalg.norm(target_emb))
                    similarity_matrix[i, j] = similarity
            
            # 使用匈牙利算法找到最佳匹配
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
                
                # 構建匹配結果
                matches = []
                for i, j in zip(row_ind, col_ind):
                    matches.append({
                        "doc1_index": i,
                        "doc2_index": j,
                        "similarity": similarity_matrix[i, j]
                    })
                
                return matches
            except ImportError:
                # 使用貪婪算法作為替代
                matches = []
                used_target_indices = set()
                
                # 為每個source段落找到最佳匹配的target段落
                for i in range(len(source_paragraphs)):
                    best_similarity = -1
                    best_j = -1
                    
                    # 找到最相似的未使用的target段落
                    for j in range(len(target_paragraphs)):
                        if j not in used_target_indices and similarity_matrix[i, j] > best_similarity:
                            best_similarity = similarity_matrix[i, j]
                            best_j = j
                    
                    # 如果找到匹配，則添加到結果中
                    if best_j != -1:
                        matches.append({
                            "doc1_index": i,
                            "doc2_index": best_j,
                            "similarity": best_similarity
                        })
                        used_target_indices.add(best_j)
                    else:
                        # 如果沒有未使用的target段落，則選擇最相似的段落
                        best_j = np.argmax(similarity_matrix[i])
                        matches.append({
                            "doc1_index": i,
                            "doc2_index": best_j,
                            "similarity": similarity_matrix[i, best_j]
                        })
                
                return matches
            
        except Exception as e:
            st.error(f"BERT匹配失敗: {e}")
            return None
    
    def _match_with_qwen(self, source_paragraphs, target_paragraphs):
        """使用Qwen API匹配段落"""
        try:
            import requests
            
            # 準備數據
            source_texts = [p['content'] for p in source_paragraphs]
            target_texts = [p['content'] for p in target_paragraphs]
            
            # 構建提示
            prompt = f"""
            我有兩組文本段落，需要找出它們之間的最佳匹配關係。
            
            第一組段落（原始文本）：
            {json.dumps(source_texts, ensure_ascii=False)}
            
            第二組段落（目標文本）：
            {json.dumps(target_texts, ensure_ascii=False)}
            
            請分析這些段落，找出每個原始段落在目標段落中的最佳匹配。返回一個JSON數組，每個元素包含：
            1. "source_index": 原始段落的索引（從0開始）
            2. "target_index": 匹配的目標段落的索引（從0開始）
            3. "similarity": 相似度評分（0到1之間）
            
            只返回JSON數組，不要有其他解釋。
            """
            
            # 準備API請求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "qwen-max",
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一個專業的文本匹配助手，擅長分析文本相似度和找出最佳匹配。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message"
                }
            }
            
            # 發送API請求
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                result_text = result["output"]["choices"][0]["message"]["content"]
                
                # 提取JSON
                import re
                json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    matches_data = json.loads(json_str)
                    
                    # 轉換為所需格式
                    matches = []
                    for item in matches_data:
                        matches.append({
                            "doc1_index": item["source_index"],
                            "doc2_index": item["target_index"],
                            "similarity": item["similarity"]
                        })
                    
                    return matches
            
            return None
        
        except Exception as e:
            st.error(f"Qwen API匹配失敗: {e}")
            return None
    
    def analyze_comparison_results(self, comparison_results):
        """使用生成式AI分析比對結果"""
        if not self.is_available:
            return "AI分析不可用，請檢查API設置。"
        
        try:
            # 準備數據
            total_paragraphs = comparison_results["statistics"]["total_paragraphs"]
            similar_paragraphs = comparison_results["statistics"]["similar_paragraphs"]
            different_paragraphs = comparison_results["statistics"]["different_paragraphs"]
            
            paragraph_similarity_percentage = (similar_paragraphs / total_paragraphs * 100) if total_paragraphs > 0 else 100
            
            # 提取不同的段落
            different_paragraph_details = []
            for result in comparison_results["paragraph_results"]:
                if not result["is_similar"]:
                    different_paragraph_details.append({
                        "original_text": result["original_text"],
                        "matched_text": result["matched_text"],
                        "exact_similarity": result["exact_similarity"]
                    })
            
            # 如果是自定義AI，使用CustomAI進行分析
            if self.model_name == "自定義AI" and self.custom_ai:
                # 提取原始文本和編輯後文本
                original_text = "\n".join([p["original_text"] for p in comparison_results["paragraph_results"]])
                edited_text = "\n".join([p["matched_text"] for p in comparison_results["paragraph_results"] if p["matched_text"]])
                
                return self.custom_ai.analyze_comparison_results(original_text, edited_text, comparison_results)
            
            # 構建提示
            prompt = f"""
            我需要你分析文件比對的結果，並提供專業的分析報告。
            
            比對統計信息：
            - 總段落數：{total_paragraphs}
            - 相似段落數：{similar_paragraphs}
            - 不同段落數：{different_paragraphs}
            - 段落相似度百分比：{paragraph_similarity_percentage:.2f}%
            
            不同段落的詳細信息（最多顯示前5個）：
            {json.dumps(different_paragraph_details[:5], ensure_ascii=False, indent=2)}
            
            請提供以下分析：
            1. 整體相似度評估
            2. 主要差異類型分類（例如：格式差異、內容遺漏、內容替換等）
            3. 差異的嚴重程度評估
            4. 建議的後續校對重點
            
            請用專業但易於理解的語言撰寫分析報告。
            """
            
            if self.model_name == "Qwen API":
                return self._analyze_with_qwen(prompt)
            elif self.model_name == "OpenAI API":
                return self._analyze_with_openai(prompt)
            else:
                return "所選AI模型不支持分析功能。"
        
        except Exception as e:
            return f"AI分析失敗: {str(e)}"
    
    def _analyze_with_qwen(self, prompt):
        """使用Qwen API分析比對結果"""
        try:
            import requests
            
            # 準備API請求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "qwen-max",
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一個專業的文本比對分析師，擅長分析文件比對結果並提供專業見解。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message"
                }
            }
            
            # 發送API請求
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["output"]["choices"][0]["message"]["content"]
                return analysis
            else:
                return f"API請求失敗: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"Qwen API分析失敗: {str(e)}"
    
    def _analyze_with_openai(self, prompt):
        """使用OpenAI API分析比對結果"""
        try:
            import openai
            
            # 設置API密鑰
            openai.api_key = self.api_key
            
            # 調用API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一個專業的文本比對分析師，擅長分析文件比對結果並提供專業見解。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            # 解析結果
            analysis = response.choices[0].message.content
            return analysis
        
        except Exception as e:
            return f"OpenAI API分析失敗: {str(e)}"

# 簡化版的比對算法，不依賴sentence-transformers
def exact_matching(text1, text2, ignore_space=True, ignore_punctuation=True, ignore_case=True, ignore_newline=True):
    """精確比對兩段文本的相似度"""
    # 文本預處理
    if ignore_space:
        text1 = re.sub(r'\s+', ' ', text1)
        text2 = re.sub(r'\s+', ' ', text2)
    
    if ignore_punctuation:
        text1 = re.sub(r'[.,;:!?，。；：！？]', '', text1)
        text2 = re.sub(r'[.,;:!?，。；：！？]', '', text2)
    
    if ignore_case:
        text1 = text1.lower()
        text2 = text2.lower()
    
    if ignore_newline:
        text1 = text1.replace('\n', ' ')
        text2 = text2.replace('\n', ' ')
    
    # 使用SequenceMatcher計算相似度
    matcher = difflib.SequenceMatcher(None, text1, text2)
    similarity = matcher.ratio()
    
    # 生成差異
    diff = list(difflib.ndiff(text1.splitlines(), text2.splitlines()))
    
    return similarity, diff

# 語意比對函數
def semantic_matching(text1, text2, model=None):
    """語意比對（使用Sentence-BERT或簡化版）"""
    if model is not None and is_sentence_transformers_available():
        try:
            # 使用Sentence-BERT計算語義相似度
            embedding1 = model.encode(text1)
            embedding2 = model.encode(text2)
            
            # 計算餘弦相似度
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            st.warning(f"語義模型計算失敗: {e}，使用簡化版語義比對")
    
    # 簡化版語義比對（詞袋模型）
    words1 = set(re.sub(r'[^\w\s]', '', text1.lower()).split())
    words2 = set(re.sub(r'[^\w\s]', '', text2.lower()).split())
    
    if not words1 or not words2:
        return 0.0
    
    # 計算Jaccard相似度
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

# 上下文感知匹配
def context_aware_matching(text1, text2, context1=None, context2=None, ignore_options=None, model=None):
    """上下文感知匹配"""
    if ignore_options is None:
        ignore_options = {}
    
    # 基本相似度計算
    exact_sim, diff = exact_matching(
        text1, text2,
        ignore_space=ignore_options.get("ignore_space", True),
        ignore_punctuation=ignore_options.get("ignore_punctuation", True),
        ignore_case=ignore_options.get("ignore_case", True),
        ignore_newline=ignore_options.get("ignore_newline", True)
    )
    
    # 語義相似度計算
    semantic_sim = semantic_matching(text1, text2, model)
    
    # 如果有上下文信息，計算上下文相似度
    context_sim = 0.0
    if context1 and context2:
        # 提取前後文
        prev_sim = semantic_matching(context1.get("previous_text", ""), context2.get("previous_text", ""), model)
        next_sim = semantic_matching(context1.get("next_text", ""), context2.get("next_text", ""), model)
        
        # 上下文相似度為前後文相似度的平均值
        context_sim = (prev_sim + next_sim) / 2
    
    # 綜合相似度計算（權重可調整）
    exact_weight = 0.5
    semantic_weight = 0.3
    context_weight = 0.2
    
    if not context1 or not context2:
        # 如果沒有上下文，調整權重
        exact_weight = 0.6
        semantic_weight = 0.4
        context_weight = 0.0
    
    combined_similarity = (
        exact_weight * exact_sim +
        semantic_weight * semantic_sim +
        context_weight * context_sim
    )
    
    return combined_similarity, diff, {
        "exact_similarity": exact_sim,
        "semantic_similarity": semantic_sim,
        "context_similarity": context_sim
    }

# 混合比對函數
def hybrid_matching(text1, text2, context1=None, context2=None, ignore_options=None, model=None):
    """混合比對（結合精確比對和語義比對）"""
    if ignore_options is None:
        ignore_options = {}
    
    # 精確比對
    exact_sim, diff = exact_matching(
        text1, text2,
        ignore_space=ignore_options.get("ignore_space", True),
        ignore_punctuation=ignore_options.get("ignore_punctuation", True),
        ignore_case=ignore_options.get("ignore_case", True),
        ignore_newline=ignore_options.get("ignore_newline", True)
    )
    
    # 語義比對
    semantic_sim = semantic_matching(text1, text2, model)
    
    # 綜合相似度計算（權重可調整）
    exact_weight = 0.6
    semantic_weight = 0.4
    
    combined_similarity = exact_weight * exact_sim + semantic_weight * semantic_sim
    
    return combined_similarity, diff, {
        "exact_similarity": exact_sim,
        "semantic_similarity": semantic_sim
    }

# 比對文檔函數
def compare_documents(doc1, doc2, ignore_options=None, comparison_mode='exact', similarity_threshold=0.6, ai_instance=None, semantic_model=None):
    """比對兩個文檔的內容"""
    if ignore_options is None:
        ignore_options = {}
    
    # 初始化結果
    paragraph_results = []
    table_results = []
    
    # 如果使用生成式AI比對
    if comparison_mode == '生成式AI比對' and ai_instance is not None:
        # 使用AI匹配段落
        matches = ai_instance.match_paragraphs(doc1["paragraphs"], doc2["paragraphs"])
        
        if matches:
            # 使用AI匹配結果
            for match in matches:
                doc1_index = match["doc1_index"]
                doc2_index = match["doc2_index"]
                similarity = match["similarity"]
                
                if doc1_index < len(doc1["paragraphs"]) and doc2_index < len(doc2["paragraphs"]):
                    para1 = doc1["paragraphs"][doc1_index]
                    para2 = doc2["paragraphs"][doc2_index]
                    
                    # 使用精確比對獲取差異
                    _, diff = exact_matching(
                        para1['content'], para2['content'],
                        ignore_space=ignore_options.get("ignore_space", True),
                        ignore_punctuation=ignore_options.get("ignore_punctuation", True),
                        ignore_case=ignore_options.get("ignore_case", True),
                        ignore_newline=ignore_options.get("ignore_newline", True)
                    )
                    
                    # 判斷是否相似
                    is_similar = similarity >= similarity_threshold
                    
                    # 添加結果
                    paragraph_results.append({
                        "original_index": doc1_index,
                        "original_text": para1["content"],
                        "matched_index": doc2_index,
                        "matched_text": para2["content"],
                        "matched_page": para2.get("page", "未找到"),
                        "exact_similarity": similarity,
                        "semantic_similarity": similarity,
                        "is_similar": is_similar
                    })
            
            # 處理未匹配的段落
            matched_doc1_indices = set(match["doc1_index"] for match in matches)
            for i, para in enumerate(doc1["paragraphs"]):
                if i not in matched_doc1_indices:
                    paragraph_results.append({
                        "original_index": i,
                        "original_text": para["content"],
                        "matched_index": -1,
                        "matched_text": "",
                        "matched_page": "未找到",
                        "exact_similarity": 0.0,
                        "semantic_similarity": 0.0,
                        "is_similar": False
                    })
        else:
            # 如果AI匹配失敗，使用混合比對作為後備
            comparison_mode = '混合比對'
    
    # 如果不是使用生成式AI比對，或者AI比對失敗
    if comparison_mode != '生成式AI比對' or not paragraph_results:
        # 比對段落
        for i, para1 in enumerate(doc1["paragraphs"]):
            best_match = None
            best_similarity = 0
            best_index = -1
            best_page = "未找到"
            best_details = {}
            
            for j, para2 in enumerate(doc2["paragraphs"]):
                # 根據比對模式選擇比對方法
                if comparison_mode == '精確比對':
                    # 使用精確比對
                    sim, diff = exact_matching(
                        para1['content'], para2['content'],
                        ignore_space=ignore_options.get("ignore_space", True),
                        ignore_punctuation=ignore_options.get("ignore_punctuation", True),
                        ignore_case=ignore_options.get("ignore_case", True),
                        ignore_newline=ignore_options.get("ignore_newline", True)
                    )
                    details = {"exact_similarity": sim}
                
                elif comparison_mode == '語意比對':
                    # 使用語義比對
                    sim = semantic_matching(para1['content'], para2['content'], semantic_model)
                    _, diff = exact_matching(para1['content'], para2['content'])  # 仍然需要差異信息
                    details = {"semantic_similarity": sim}
                
                elif comparison_mode == '混合比對':
                    # 使用混合比對
                    sim, diff, details = hybrid_matching(
                        para1['content'], para2['content'],
                        None, None,
                        ignore_options,
                        semantic_model
                    )
                
                elif comparison_mode == '上下文感知比對':
                    # 使用上下文感知比對
                    # 提取上下文信息
                    context1 = para1.get('context', {})
                    context2 = para2.get('context', {})
                    
                    sim, diff, details = context_aware_matching(
                        para1['content'], para2['content'],
                        context1, context2,
                        ignore_options,
                        semantic_model
                    )
                
                else:
                    # 默認使用精確比對
                    sim, diff = exact_matching(
                        para1['content'], para2['content'],
                        ignore_space=ignore_options.get("ignore_space", True),
                        ignore_punctuation=ignore_options.get("ignore_punctuation", True),
                        ignore_case=ignore_options.get("ignore_case", True),
                        ignore_newline=ignore_options.get("ignore_newline", True)
                    )
                    details = {"exact_similarity": sim}
                
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = para2
                    best_index = j
                    best_page = para2.get("page", "未找到")
                    best_details = details
            
            # 判斷是否相似
            is_similar = best_similarity >= similarity_threshold
            
            # 添加結果
            result = {
                "original_index": i,
                "original_text": para1["content"],
                "matched_index": best_index,
                "matched_text": best_match["content"] if best_match else "",
                "matched_page": best_page,
                "exact_similarity": best_details.get("exact_similarity", best_similarity),
                "is_similar": is_similar
            }
            
            # 添加其他相似度信息
            if "semantic_similarity" in best_details:
                result["semantic_similarity"] = best_details["semantic_similarity"]
            if "context_similarity" in best_details:
                result["context_similarity"] = best_details["context_similarity"]
            
            paragraph_results.append(result)
    
    # 比對表格
    for i, table1 in enumerate(doc1.get("tables", [])):
        best_match = None
        best_similarity = 0
        best_index = -1
        best_page = "未找到"
        
        for j, table2 in enumerate(doc2.get("tables", [])):
            # 計算表格相似度
            table_similarity = calculate_table_similarity(table1["content"], table2["content"])
            
            if table_similarity > best_similarity:
                best_similarity = table_similarity
                best_match = table2
                best_index = j
                best_page = table2.get("page", "未找到")
        
        # 判斷是否相似
        is_similar = best_similarity >= similarity_threshold
        
        # 添加結果
        table_results.append({
            "original_index": i,
            "original_table": table1["content"],
            "matched_index": best_index,
            "matched_table": best_match["content"] if best_match else None,
            "matched_page": best_page,
            "similarity": best_similarity,
            "is_similar": is_similar
        })
    
    # 計算統計信息
    statistics = {
        "total_paragraphs": len(paragraph_results),
        "similar_paragraphs": sum(1 for r in paragraph_results if r["is_similar"]),
        "different_paragraphs": sum(1 for r in paragraph_results if not r["is_similar"]),
        "total_tables": len(table_results),
        "similar_tables": sum(1 for r in table_results if r["is_similar"]),
        "different_tables": sum(1 for r in table_results if not r["is_similar"])
    }
    
    return {
        "paragraph_results": paragraph_results,
        "table_results": table_results,
        "statistics": statistics
    }

def calculate_table_similarity(table1, table2):
    """計算兩個表格的相似度"""
    # 如果表格為空，返回0
    if not table1 or not table2:
        return 0
    
    # 將表格轉換為文本
    text1 = "\n".join([" ".join(row) for row in table1])
    text2 = "\n".join([" ".join(row) for row in table2])
    
    # 使用精確比對
    similarity, _ = exact_matching(text1, text2)
    
    return similarity

def format_diff_html(diff, mode="字符級別"):
    """將差異格式化為HTML"""
    if not diff:
        return ""
    
    if mode == "字符級別":
        # 字符級別差異
        result = []
        for line in diff:
            if line.startswith('- '):
                result.append(f'<span class="diff-char-removed">{line[2:]}</span>')
            elif line.startswith('+ '):
                result.append(f'<span class="diff-char-added">{line[2:]}</span>')
            elif line.startswith('  '):
                result.append(line[2:])
        return "".join(result)
    
    elif mode == "詞語級別":
        # 詞語級別差異
        result = []
        for line in diff:
            if line.startswith('- '):
                result.append(f'<span class="diff-removed">{line[2:]}</span><br>')
            elif line.startswith('+ '):
                result.append(f'<span class="diff-added">{line[2:]}</span><br>')
            elif line.startswith('  '):
                result.append(f'{line[2:]}<br>')
        return "".join(result)
    
    else:  # 行級別
        # 行級別差異
        result = []
        for line in diff:
            if line.startswith('- '):
                result.append(f'<div class="diff-removed">{line[2:]}</div>')
            elif line.startswith('+ '):
                result.append(f'<div class="diff-added">{line[2:]}</div>')
            elif line.startswith('  '):
                result.append(f'<div>{line[2:]}</div>')
        return "".join(result)

def generate_comparison_report(comparison_results, diff_display_mode="字符級別", show_all_content=False):
    """生成比對報告"""
    # 處理段落比對結果
    paragraph_details = []
    for result in comparison_results["paragraph_results"]:
        # 生成差異HTML
        diff_html = ""
        if result["matched_text"]:
            # 使用精確比對
            _, diff = exact_matching(result["original_text"], result["matched_text"])
            diff_html = format_diff_html(diff, diff_display_mode)
        
        # 添加詳細信息
        paragraph_details.append({
            "original_index": result["original_index"],
            "original_text": result["original_text"],
            "matched_text": result["matched_text"],
            "matched_page": result["matched_page"],
            "exact_similarity": result["exact_similarity"],
            "semantic_similarity": result.get("semantic_similarity", 0.0),
            "context_similarity": result.get("context_similarity", 0.0),
            "is_similar": result["is_similar"],
            "diff_html": diff_html
        })
    
    # 處理表格比對結果
    table_details = []
    for result in comparison_results["table_results"]:
        table_details.append({
            "original_index": result["original_index"],
            "original_table": result["original_table"],
            "matched_table": result["matched_table"],
            "matched_page": result["matched_page"],
            "similarity": result["similarity"],
            "is_similar": result["is_similar"]
        })
    
    # 計算摘要信息
    total_paragraphs = comparison_results["statistics"]["total_paragraphs"]
    similar_paragraphs = comparison_results["statistics"]["similar_paragraphs"]
    different_paragraphs = comparison_results["statistics"]["different_paragraphs"]
    
    total_tables = comparison_results["statistics"]["total_tables"]
    similar_tables = comparison_results["statistics"]["similar_tables"]
    different_tables = comparison_results["statistics"]["different_tables"]
    
    # 計算相似度百分比
    paragraph_similarity_percentage = (similar_paragraphs / total_paragraphs * 100) if total_paragraphs > 0 else 100
    table_similarity_percentage = (similar_tables / total_tables * 100) if total_tables > 0 else 100
    
    # 生成摘要
    summary = {
        "total_paragraphs": total_paragraphs,
        "similar_paragraphs": similar_paragraphs,
        "different_paragraphs": different_paragraphs,
        "paragraph_similarity_percentage": paragraph_similarity_percentage,
        "total_tables": total_tables,
        "similar_tables": similar_tables,
        "different_tables": different_tables,
        "table_similarity_percentage": table_similarity_percentage
    }
    
    return {
        "summary": summary,
        "paragraph_details": paragraph_details,
        "table_details": table_details
    }

# 生成PDF頁面預覽並標記差異
def generate_pdf_preview_with_diff(pdf_path, page_number, diff_text, temp_dir):
    """生成PDF頁面預覽並標記差異"""
    try:
        # 打開PDF文件
        doc = fitz.open(pdf_path)
        
        # 檢查頁碼是否有效
        if page_number < 1 or page_number > len(doc):
            return None, f"無效的頁碼: {page_number}，PDF共有{len(doc)}頁"
        
        # 獲取頁面
        page = doc[page_number - 1]
        
        # 將頁面轉換為圖像
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
        
        # 保存為臨時圖像文件
        img_path = os.path.join(temp_dir, f"page_{page_number}.png")
        pix.save(img_path)
        
        # 使用PIL打開圖像
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        
        # 嘗試加載字體
        try:
            font = ImageFont.truetype("Arial", 14)
        except IOError:
            font = ImageFont.load_default()
        
        # 在圖像上標記差異
        # 這裡需要實現一個算法來定位差異文本在頁面上的位置
        # 由於這是一個複雜的任務，這裡只是簡單地在頁面底部添加一個差異說明
        
        # 在頁面底部添加差異說明
        draw.rectangle([(10, img.height - 50), (img.width - 10, img.height - 10)], fill=(255, 240, 240))
        draw.text((20, img.height - 40), f"差異: {diff_text[:100]}...", fill=(255, 0, 0), font=font)
        
        # 保存修改後的圖像
        marked_img_path = os.path.join(temp_dir, f"page_{page_number}_marked.png")
        img.save(marked_img_path)
        
        # 關閉PDF文件
        doc.close()
        
        return marked_img_path, None
    
    except Exception as e:
        return None, f"生成PDF預覽時出錯: {str(e)}"

# 初始化會話狀態
def init_session_state():
    if 'word_data' not in st.session_state:
        st.session_state.word_data = None
    if 'pdf_data' not in st.session_state:
        st.session_state.pdf_data = None
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'comparison_report' not in st.session_state:
        st.session_state.comparison_report = None
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = None
    if 'ai_summary_report' not in st.session_state:
        st.session_state.ai_summary_report = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'current_paragraph_index' not in st.session_state:
        st.session_state.current_paragraph_index = 0
    if 'current_table_index' not in st.session_state:
        st.session_state.current_table_index = 0
    if 'show_all_content' not in st.session_state:
        st.session_state.show_all_content = False
    if 'diff_display_mode' not in st.session_state:
        st.session_state.diff_display_mode = "字符級別"
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = "混合比對"
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.8
    if 'ignore_options' not in st.session_state:
        st.session_state.ignore_options = {
            "ignore_space": True,
            "ignore_punctuation": True,
            "ignore_case": True,
            "ignore_newline": True
        }
    if 'use_ocr' not in st.session_state:
        st.session_state.use_ocr = True
    if 'ocr_engine' not in st.session_state:
        st.session_state.ocr_engine = "自動選擇"
    if 'qwen_api_key' not in st.session_state:
        st.session_state.qwen_api_key = ""
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = "Qwen API"
    if 'ai_api_key' not in st.session_state:
        st.session_state.ai_api_key = ""
    if 'ai_api_url' not in st.session_state:
        st.session_state.ai_api_url = ""
    if 'pdf_page_images' not in st.session_state:
        st.session_state.pdf_page_images = {}
    if 'highlighted_images' not in st.session_state:
        st.session_state.highlighted_images = {}
    if 'use_example_files' not in st.session_state:
        st.session_state.use_example_files = False
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

# 側邊欄設置
def sidebar_settings():
    with st.sidebar:
        st.title("期刊比對系統")
        
        # 顯示系統狀態
        with st.expander("系統狀態", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if is_java_installed():
                    st.success("Java已安裝")
                else:
                    st.error("Java未安裝，表格提取功能可能受限")
                
                if is_easyocr_available():
                    st.success("OpenOCR (EasyOCR) 已安裝")
                else:
                    st.warning("OpenOCR未安裝，將使用Tesseract作為替代")
            
            with col2:
                if is_tabula_available():
                    st.success("表格提取工具已安裝")
                else:
                    st.error("表格提取工具未安裝或無法使用")
                
                if is_sentence_transformers_available():
                    st.success("語義模型已安裝")
                else:
                    st.warning("語義模型未安裝，語義比對功能將使用簡化版")
        
        # 系統設置
        st.header("系統設置")
        
        # 示例文件選項
        st.session_state.use_example_files = st.checkbox("使用示例文件進行演示", value=st.session_state.use_example_files)
        
        # 比對設置
        st.subheader("比對設置")
        st.session_state.comparison_mode = st.radio(
            "比對模式",
            ["精確比對", "語意比對", "混合比對", "上下文感知比對", "生成式AI比對"],
            index=2
        )
        
        # 如果選擇生成式AI比對，顯示模型選擇
        if st.session_state.comparison_mode == "生成式AI比對":
            st.markdown('<div class="ai-model-section">', unsafe_allow_html=True)
            st.subheader("生成式AI設置")
            
            st.session_state.ai_model = st.selectbox(
                "選擇AI模型",
                ["BERT多語言模型", "MPNet中文模型", "RoBERTa中文模型", 
                 "OpenAI API", "Anthropic API", "Gemini API", "Qwen API", "自定義AI"],
                index=6,
                help="選擇用於段落匹配的生成式AI模型"
            )
            
            # 如果選擇API模型，顯示API Key輸入框
            if st.session_state.ai_model in ["OpenAI API", "Anthropic API", "Gemini API", "Qwen API"]:
                st.session_state.ai_api_key = st.text_input(f"{st.session_state.ai_model} Key", 
                                                          type="password", 
                                                          value=st.session_state.ai_api_key,
                                                          help=f"輸入您的{st.session_state.ai_model} Key")
                
                if st.session_state.ai_model == "OpenAI API":
                    st.info("OpenAI API使用GPT-4模型，提供高精度段落匹配")
                elif st.session_state.ai_model == "Anthropic API":
                    st.info("Anthropic API使用Claude模型，提供高精度段落匹配")
                elif st.session_state.ai_model == "Gemini API":
                    st.info("Gemini API使用Google的Gemini Pro模型，提供高精度段落匹配")
                elif st.session_state.ai_model == "Qwen API":
                    st.info("Qwen API使用阿里巴巴的Qwen Max模型，提供高精度段落匹配")
            elif st.session_state.ai_model == "自定義AI":
                st.session_state.ai_api_key = st.text_input("API Key (可選)", 
                                                          type="password", 
                                                          value=st.session_state.ai_api_key,
                                                          help="輸入您的API Key，如果不提供將使用免費API")
                
                st.session_state.ai_api_url = st.text_input("API URL (可選)", 
                                                          value=st.session_state.ai_api_url,
                                                          help="輸入API URL，如果不提供將使用免費API")
                
                st.info("自定義AI支援多種API格式（OpenAI、Anthropic、Qwen），也可以使用免費API")
            else:
                st.info("本地模型無需API Key，但精度可能低於API模型")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 相似度閾值
        st.session_state.similarity_threshold = st.slider(
            "相似度閾值",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.similarity_threshold,
            step=0.05,
            help="相似度低於此閾值的段落將被標記為不一致"
        )
        
        # 忽略選項
        st.subheader("忽略選項")
        st.session_state.ignore_options["ignore_space"] = st.checkbox("忽略空格", value=st.session_state.ignore_options["ignore_space"])
        st.session_state.ignore_options["ignore_punctuation"] = st.checkbox("忽略標點符號", value=st.session_state.ignore_options["ignore_punctuation"])
        st.session_state.ignore_options["ignore_case"] = st.checkbox("忽略大小寫", value=st.session_state.ignore_options["ignore_case"])
        st.session_state.ignore_options["ignore_newline"] = st.checkbox("忽略換行", value=st.session_state.ignore_options["ignore_newline"])
        
        # OCR設置
        st.subheader("OCR設置")
        st.session_state.ocr_engine = st.radio(
            "OCR引擎",
            ["自動選擇", "Tesseract", "OpenOCR (EasyOCR)", "Qwen API"],
            index=0,
            help="選擇用於提取PDF文本的OCR引擎"
        )
        
        # 如果選擇Qwen API，顯示API Key輸入框
        if st.session_state.ocr_engine == "Qwen API":
            st.session_state.qwen_api_key = st.text_input("Qwen API Key", 
                                                        type="password", 
                                                        value=st.session_state.qwen_api_key,
                                                        help="輸入您的Qwen API Key")
            st.info("Qwen API提供高精度OCR和表格識別，特別適合複雜排版的PDF")
        else:
            st.session_state.use_ocr = st.checkbox("使用OCR提取", value=st.session_state.use_ocr, help="啟用OCR可以提高文本提取質量，但會增加處理時間")
        
        # 顯示設置
        st.subheader("顯示設置")
        st.session_state.diff_display_mode = st.selectbox(
            "差異顯示模式",
            ["字符級別", "詞語級別", "行級別"],
            index=0
        )
        
        st.session_state.show_all_content = st.checkbox("顯示所有內容", value=st.session_state.show_all_content)
        
        # 系統資訊
        st.subheader("系統資訊")
        st.info("本系統用於比對原始Word文件與美編後PDF文件的內容差異，幫助校對人員快速找出不一致之處。")

# 文件上傳區域
def file_upload_section():
    st.header("文件上傳")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始Word文件")
        word_file = st.file_uploader("上傳原始Word文件", type=["docx"], key="word_uploader", disabled=st.session_state.use_example_files)
        
        if word_file:
            st.success(f"已上傳: {word_file.name}")
    
    with col2:
        st.subheader("美編後PDF文件")
        pdf_file = st.file_uploader("上傳美編後PDF文件", type=["pdf"], key="pdf_uploader", disabled=st.session_state.use_example_files)
        
        if pdf_file:
            st.success(f"已上傳: {pdf_file.name}")
    
    # 使用示例文件
    if st.session_state.use_example_files:
        st.warning("示例文件功能需要上傳您自己的文件。請取消勾選「使用示例文件進行演示」選項，然後上傳您的文件。")
        word_file = None
        pdf_file = None
    
    if word_file and pdf_file:
        if st.button("開始比對", key="start_comparison"):
            with st.spinner("正在提取文件內容並進行比對..."):
                # 初始化OCR實例
                ocr_instance = None
                if st.session_state.ocr_engine == "Qwen API" and st.session_state.qwen_api_key:
                    ocr_instance = QwenOCR(st.session_state.qwen_api_key)
                
                # 初始化AI實例
                ai_instance = None
                if st.session_state.comparison_mode == "生成式AI比對":
                    ai_instance = GenerativeAI(
                        st.session_state.ai_model, 
                        st.session_state.ai_api_key,
                        st.session_state.ai_api_url
                    )
                
                # 加載語義模型
                semantic_model = None
                if st.session_state.comparison_mode in ["語意比對", "混合比對", "上下文感知比對"]:
                    semantic_model = load_semantic_model()
                
                # 提取文件內容
                word_data, pdf_data = extract_and_process_documents(
                    word_file, 
                    pdf_file, 
                    st.session_state.use_ocr, 
                    st.session_state.ocr_engine,
                    ocr_instance
                )
                
                st.session_state.word_data = word_data
                st.session_state.pdf_data = pdf_data
                
                # 進行比對
                comparison_results = compare_documents(
                    word_data,
                    pdf_data,
                    st.session_state.ignore_options,
                    st.session_state.comparison_mode,
                    st.session_state.similarity_threshold,
                    ai_instance,
                    semantic_model
                )
                
                st.session_state.comparison_results = comparison_results
                
                # 生成比對報告
                comparison_report = generate_comparison_report(
                    comparison_results,
                    st.session_state.diff_display_mode,
                    st.session_state.show_all_content
                )
                
                st.session_state.comparison_report = comparison_report
                
                # 如果使用生成式AI，生成AI分析報告
                if ai_instance and ai_instance.is_available:
                    with st.spinner("正在使用AI分析比對結果..."):
                        ai_analysis = ai_instance.analyze_comparison_results(comparison_results)
                        st.session_state.ai_analysis = ai_analysis
                
                # 提取PDF頁面圖像
                with st.spinner("正在提取PDF頁面圖像..."):
                    # 保存上傳的PDF文件到臨時文件
                    temp_pdf_path = os.path.join(st.session_state.temp_dir, "temp.pdf")
                    
                    with open(temp_pdf_path, "wb") as f:
                        f.write(pdf_file.getvalue())
                    
                    # 打開PDF文件
                    pdf_doc = fitz.open(temp_pdf_path)
                    
                    # 提取每一頁的圖像
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc[page_num]
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        
                        # 將圖像保存到臨時文件
                        img_path = os.path.join(st.session_state.temp_dir, f"page_{page_num+1}.png")
                        pix.save(img_path)
                        
                        # 將圖像讀取為PIL圖像
                        img = Image.open(img_path)
                        
                        # 將圖像轉換為bytes
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        # 保存到session_state
                        st.session_state.pdf_page_images[page_num+1] = img_byte_arr
                    
                    # 關閉PDF文件
                    pdf_doc.close()
                
                # 為不同的段落生成標記後的PDF頁面預覽
                with st.spinner("正在生成標記後的PDF頁面預覽..."):
                    for result in comparison_report["paragraph_details"]:
                        if not result["is_similar"] and result["matched_page"] != "未找到":
                            try:
                                page_num = int(result["matched_page"])
                                
                                # 生成標記後的頁面預覽
                                marked_img_path, error = generate_pdf_preview_with_diff(
                                    temp_pdf_path,
                                    page_num,
                                    result["original_text"],
                                    st.session_state.temp_dir
                                )
                                
                                if marked_img_path and not error:
                                    # 讀取標記後的圖像
                                    with open(marked_img_path, "rb") as f:
                                        img_bytes = f.read()
                                    
                                    # 保存到session_state
                                    key = f"{page_num}_{result['original_index']}"
                                    st.session_state.highlighted_images[key] = img_bytes
                            except:
                                pass
                
                st.session_state.processing_complete = True

# 顯示比對結果
def display_comparison_results():
    if st.session_state.processing_complete and st.session_state.comparison_results and st.session_state.comparison_report:
        st.header("比對結果")
        
        # 顯示摘要信息
        st.subheader("摘要")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "段落相似度",
                f"{st.session_state.comparison_report['summary']['paragraph_similarity_percentage']:.2f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "表格相似度",
                f"{st.session_state.comparison_report['summary']['table_similarity_percentage']:.2f}%",
                delta=None
            )
        
        with col3:
            total_items = st.session_state.comparison_report['summary']['total_paragraphs'] + st.session_state.comparison_report['summary']['total_tables']
            similar_items = st.session_state.comparison_report['summary']['similar_paragraphs'] + st.session_state.comparison_report['summary']['similar_tables']
            
            if total_items > 0:
                overall_similarity = similar_items / total_items * 100
            else:
                overall_similarity = 0
            
            st.metric(
                "整體相似度",
                f"{overall_similarity:.2f}%",
                delta=None
            )
        
        # 顯示AI分析報告
        if st.session_state.ai_analysis:
            with st.expander("AI分析報告", expanded=True):
                st.markdown('<div class="ai-analysis">', unsafe_allow_html=True)
                st.markdown(st.session_state.ai_analysis)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # 顯示詳細統計信息
        with st.expander("詳細統計信息"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**段落統計**")
                st.markdown(f"總段落數: {st.session_state.comparison_report['summary']['total_paragraphs']}")
                st.markdown(f"相似段落數: {st.session_state.comparison_report['summary']['similar_paragraphs']}")
                st.markdown(f"不同段落數: {st.session_state.comparison_report['summary']['different_paragraphs']}")
            
            with col2:
                st.markdown("**表格統計**")
                st.markdown(f"總表格數: {st.session_state.comparison_report['summary']['total_tables']}")
                st.markdown(f"相似表格數: {st.session_state.comparison_report['summary']['similar_tables']}")
                st.markdown(f"不同表格數: {st.session_state.comparison_report['summary']['different_tables']}")
        
        # 創建標籤頁
        tab1, tab2 = st.tabs(["段落比對結果", "表格比對結果"])
        
        # 段落比對結果標籤頁
        with tab1:
            # 顯示段落比對結果
            st.subheader("段落比對結果")
            
            # 過濾結果
            if st.session_state.show_all_content:
                paragraph_details = st.session_state.comparison_report["paragraph_details"]
            else:
                paragraph_details = [detail for detail in st.session_state.comparison_report["paragraph_details"] if not detail["is_similar"]]
            
            # 排序結果，將相似度最低的放在前面
            paragraph_details.sort(key=lambda x: x["exact_similarity"])
            
            # 顯示段落比對結果
            for i, detail in enumerate(paragraph_details):
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"**段落 {detail['original_index'] + 1}**")
                        st.markdown(f"相似度: {detail['exact_similarity']:.2f}")
                        
                        # 如果有語義相似度，顯示
                        if "semantic_similarity" in detail:
                            st.markdown(f"語義相似度: {detail['semantic_similarity']:.2f}")
                        
                        # 如果有上下文相似度，顯示
                        if "context_similarity" in detail:
                            st.markdown(f"上下文相似度: {detail['context_similarity']:.2f}")
                        
                        st.markdown(f"頁碼: {detail['matched_page']}")
                    
                    with col2:
                        # 顯示原始文本和匹配文本
                        st.markdown("**原始文本:**")
                        st.markdown(detail["original_text"])
                        
                        st.markdown("**美編後文本:**")
                        if detail["matched_text"]:
                            st.markdown(detail["matched_text"])
                        else:
                            st.markdown("未找到匹配文本")
                        
                        # 顯示差異
                        if detail["diff_html"]:
                            st.markdown("**差異:**")
                            st.markdown(detail["diff_html"], unsafe_allow_html=True)
                    
                    # 顯示PDF頁面預覽
                    if detail["matched_page"] != "未找到":
                        try:
                            page_num = int(detail["matched_page"])
                            
                            # 檢查是否有標記後的圖像
                            key = f"{page_num}_{detail['original_index']}"
                            if key in st.session_state.highlighted_images:
                                st.image(
                                    st.session_state.highlighted_images[key],
                                    caption=f"頁面 {page_num} (已標記差異)",
                                    use_column_width=True
                                )
                            # 否則顯示原始頁面
                            elif page_num in st.session_state.pdf_page_images:
                                st.image(
                                    st.session_state.pdf_page_images[page_num],
                                    caption=f"頁面 {page_num}",
                                    use_column_width=True
                                )
                        except:
                            pass
                
                st.markdown("---")
        
        # 表格比對結果標籤頁
        with tab2:
            # 顯示表格比對結果
            st.subheader("表格比對結果")
            
            # 過濾結果
            if "table_details" in st.session_state.comparison_report:
                if st.session_state.show_all_content:
                    table_details = st.session_state.comparison_report["table_details"]
                else:
                    table_details = [detail for detail in st.session_state.comparison_report["table_details"] if not detail["is_similar"]]
                
                # 排序結果，將相似度最低的放在前面
                table_details.sort(key=lambda x: x["similarity"])
                
                # 顯示表格比對結果
                for i, detail in enumerate(table_details):
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            st.markdown(f"**表格 {detail['original_index'] + 1}**")
                            st.markdown(f"相似度: {detail['similarity']:.2f}")
                            st.markdown(f"頁碼: {detail['matched_page']}")
                        try:    
                            with col2:
                                # 顯示原始表格和匹配表格
                                st.markdown("**原始表格:**")
                                if detail["original_table"]:
                                    df1 = pd.DataFrame(detail["original_table"])
                                    st.dataframe(df1)
                                else:
                                    st.markdown("無表格數據")
                                
                                st.markdown("**美編後表格:**")
                                if detail["matched_table"]:
                                    df2 = pd.DataFrame(detail["matched_table"])
                                    st.dataframe(df2)
                                else:
                                    st.warning(f"頁碼 {detail['matched_page']} 超出範圍")
                        except Exception as e:
                                st.error(f"無法顯示表格: {e}")
                        
                        # 顯示PDF頁面預覽
                        if detail["matched_page"] != "未找到":
                            try:
                                page_num = int(detail["matched_page"])
                                
                                # 顯示原始頁面
                                if page_num in st.session_state.pdf_page_images:
                                    st.image(
                                        st.session_state.pdf_page_images[page_num],
                                        caption=f"頁面 {page_num}",
                                        use_column_width=True
                                    )
                            except:
                                pass
                    
                    st.markdown("---")
            else:
                st.warning("未比對到有效表格，請檢查文件內容是否包含表格。")
    else:
        if not st.session_state.processing_complete:
            st.info("請上傳文件並點擊「開始比對」按鈕。")
        else:
            st.warning("未比對到有效段落，請檢查文件內容是否正確。")

# 主函數
def main():
    # 加載CSS
    load_css()
    
    # 初始化會話狀態
    init_session_state()
    
    # 側邊欄設置
    sidebar_settings()
    
    # 文件上傳區域
    file_upload_section()
    
    # 顯示比對結果
    display_comparison_results()
    
    # 清理臨時文件
    def cleanup():
        if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
            try:
                shutil.rmtree(st.session_state.temp_dir)
            except:
                pass
    
    # 註冊清理函數
    import atexit
    atexit.register(cleanup)

if __name__ == "__main__":
    main()
