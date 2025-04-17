import os
import re
import json
import tempfile
import shutil
import numpy as np
import pandas as pd
import difflib
import docx
import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
from io import BytesIO
import base64

# 檢查sentence-transformers是否可用
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# 導入上下文感知匹配模塊
try:
    from context_aware_matching import context_aware_matching, semantic_structure_matching
    CONTEXT_MATCHING_AVAILABLE = True
except ImportError:
    CONTEXT_MATCHING_AVAILABLE = False

# 設置頁面配置
st.set_page_config(
    page_title="期刊比對系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
st.markdown("""
<style>
    .main-header {
        color: #1E88E5;
        font-size: 2.5rem;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #0D47A1;
        font-size: 1.8rem;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .diff-char-removed {
        background-color: #FFCDD2;
        text-decoration: line-through;
    }
    .diff-char-added {
        background-color: #C8E6C9;
    }
    .diff-word-removed {
        background-color: #EF9A9A;
        text-decoration: line-through;
        padding: 2px;
        margin: 2px;
    }
    .diff-word-added {
        background-color: #A5D6A7;
        padding: 2px;
        margin: 2px;
    }
    .table-warning {
        background-color: #FFF9C4;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .api-key-input {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .ai-model-section {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 顯示標題
st.markdown('<h1 class="main-header">期刊比對系統</h1>', unsafe_allow_html=True)
st.markdown('本系統用於比對原始Word文件與美編後PDF文件的內容差異，幫助校對人員快速找出不一致之處。')

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
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key
        self.is_available = self._check_availability()
    
    def _check_availability(self):
        """檢查模型是否可用"""
        if self.model_name in ["BERT多語言模型", "MPNet中文模型", "RoBERTa中文模型"]:
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
            if self.model_name == "BERT多語言模型":
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
    
    # 其他模型的匹配方法實現...
    
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

# 顯示系統狀態
with st.expander("系統狀態", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if is_java_installed():
            st.success("Java已安裝")
        else:
            st.error("Java未安裝，表格提取功能可能受限")
    
    with col2:
        if is_easyocr_available():
            st.success("OpenOCR (EasyOCR) 已安裝")
        else:
            st.warning("OpenOCR未安裝，將使用Tesseract作為替代")
    
    with col3:
        if is_tabula_available():
            st.success("表格提取工具已安裝")
        else:
            st.error("表格提取工具未安裝或無法使用")
    
    with col4:
        if is_sentence_transformers_available():
            st.success("語義模型已安裝")
        else:
            st.warning("語義模型未安裝，語義比對功能將使用簡化版")
    
    with col5:
        if CONTEXT_MATCHING_AVAILABLE:
            st.success("上下文感知匹配可用")
        else:
            st.warning("上下文感知匹配不可用，將使用基本匹配")
    
    if not is_java_installed():
        st.info("安裝Java: 請執行 'sudo apt-get install -y default-jre' 或安裝適合您系統的Java運行環境")
    
    if not is_easyocr_available():
        st.info("安裝OpenOCR: 請執行 'pip install easyocr'")
    
    if not is_sentence_transformers_available():
        st.info("安裝語義模型: 請執行 'pip install sentence-transformers'")
        st.warning("注意：語義模型未安裝，系統將使用簡化版語義比對，精度可能較低")

# 側邊欄 - 比對設置
with st.sidebar:
    st.header("比對設置")
    
    # 比對模式
    comparison_mode = st.radio(
        "比對模式",
        ["精確比對", "語意比對", "混合比對", "生成式AI比對", "上下文感知比對"],
        index=4,
        help="選擇比對模式，上下文感知比對提供最高精度"
    )
    
    # 如果選擇生成式AI比對，顯示模型選擇
    if comparison_mode == "生成式AI比對":
        st.markdown('<div class="ai-model-section">', unsafe_allow_html=True)
        st.subheader("生成式AI設置")
        
        ai_model = st.selectbox(
            "選擇AI模型",
            ["BERT多語言模型", "MPNet中文模型", "RoBERTa中文模型", 
             "OpenAI API", "Anthropic API", "Gemini API", "Qwen API"],
            index=0,
            help="選擇用於段落匹配的生成式AI模型"
        )
        
        # 如果選擇API模型，顯示API Key輸入框
        if ai_model in ["OpenAI API", "Anthropic API", "Gemini API", "Qwen API"]:
            ai_api_key = st.text_input(f"{ai_model} Key", type="password", help=f"輸入您的{ai_model} Key")
            
            if ai_model == "OpenAI API":
                st.info("OpenAI API使用GPT-4模型，提供高精度段落匹配")
            elif ai_model == "Anthropic API":
                st.info("Anthropic API使用Claude模型，提供高精度段落匹配")
            elif ai_model == "Gemini API":
                st.info("Gemini API使用Google的Gemini Pro模型，提供高精度段落匹配")
            elif ai_model == "Qwen API":
                st.info("Qwen API使用阿里巴巴的Qwen Max模型，提供高精度段落匹配")
        else:
            ai_api_key = None
            st.info("本地模型無需API Key，但精度可能低於API模型")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        ai_model = None
        ai_api_key = None
    
    # 相似度閾值
    similarity_threshold = st.slider(
        "相似度閾值",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="相似度低於此閾值的段落將被標記為不一致"
    )
    
    # 忽略選項
    st.subheader("忽略選項")
    ignore_space = st.checkbox("忽略空格", value=True)
    ignore_punctuation = st.checkbox("忽略標點符號", value=True)
    ignore_case = st.checkbox("忽略大小寫", value=True)
    ignore_newline = st.checkbox("忽略換行", value=True)
    
    # OCR設置
    st.subheader("OCR設置")
    ocr_engine = st.radio(
        "OCR引擎",
        ["自動選擇", "Tesseract", "OpenOCR (EasyOCR)", "Qwen API"],
        index=0,
        help="選擇用於提取PDF文本的OCR引擎"
    )
    
    # 如果選擇Qwen API，顯示API Key輸入框
    if ocr_engine == "Qwen API":
        qwen_api_key = st.text_input("Qwen API Key", type="password", help="輸入您的Qwen API Key")
        st.info("Qwen API提供高精度OCR和表格識別，特別適合複雜排版的PDF")
    else:
        use_ocr = st.checkbox("使用OCR提取", value=True, help="啟用OCR可以提高文本提取質量，但會增加處理時間")
    
    # 表格處理設置
    st.subheader("表格處理")
    table_handling = st.radio(
        "表格處理方式",
        ["輔助人
(Content truncated due to size limit. Use line ranges to read in chunks)