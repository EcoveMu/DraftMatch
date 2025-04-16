import streamlit as st
import os
import tempfile
import docx
import re
import difflib
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import json
import shutil
import sys
from pathlib import Path
from enhanced_extraction import enhanced_pdf_extraction, improved_matching_algorithm
from qwen_api import QwenOCR

# 檢查sentence-transformers是否可用
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# 設置頁面配置
st.set_page_config(
    page_title="期刊比對系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS樣式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .highlight-diff {
        background-color: #FFECB3;
        padding: 2px;
        border-radius: 3px;
    }
    .diff-added {
        color: #000000;
        background-color: #C8E6C9;
        padding: 2px;
        border-radius: 3px;
    }
    .diff-removed {
        color: #000000;
        background-color: #FFCDD2;
        padding: 2px;
        border-radius: 3px;
    }
    .result-container {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .similarity-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .similarity-medium {
        color: #F57F17;
        font-weight: bold;
    }
    .similarity-low {
        color: #C62828;
        font-weight: bold;
    }
    .table-warning {
        background-color: #FFF3E0;
        padding: 10px;
        border-left: 4px solid #FF9800;
        margin-bottom: 10px;
    }
    .file-uploader-container {
        border: 1px dashed #BDBDBD;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .multi-file-uploader {
        margin-bottom: 20px;
    }
    .chapter-selector {
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .diff-char-removed {
        color: #000000;
        background-color: #FFCDD2;
        font-weight: bold;
        padding: 1px;
        border-radius: 2px;
    }
    .diff-char-added {
        color: #000000;
        background-color: #C8E6C9;
        font-weight: bold;
        padding: 1px;
        border-radius: 2px;
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
    .api-key-input {
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

from enhanced_extraction import enhanced_pdf_extraction
from comparison_algorithm_example import compare_documents
from custom_ai import CustomAI

st.header("📁 文件上傳")

col1, col2 = st.columns(2)
with col1:
    word_file = st.file_uploader("上傳原始 Word 文稿", type=["docx"])
with col2:
    pdf_file = st.file_uploader("上傳美編後 PDF 文件", type=["pdf"])

similarity_threshold = st.slider("相似度閾值", 0.0, 1.0, 0.6, 0.05)
use_ai = st.checkbox("使用生成式 AI 進行語意比對", value=False)
ai_key = st.text_input("🔑 請輸入你的 AI API 金鑰", type="password") if use_ai else None

if st.button("開始比對"):
    if word_file is None or pdf_file is None:
        st.warning("請先上傳 Word 與 PDF 檔案")
    else:
        st.info("🧠 開始比對中...")

        # 1. 保存上傳檔案至暫存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_word:
            tmp_word.write(word_file.read())
            word_path = tmp_word.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_file.read())
            pdf_path = tmp_pdf.name

        # 2. 進行文字抽取
        word_data, pdf_data = enhanced_pdf_extraction(word_path, pdf_path)

        # 3. 建立 AI 模型（如啟用）
        ai_instance = None
        if use_ai and ai_key:
            ai_instance = CustomAI(api_key=ai_key, model_name="Qwen")

        # 4. 執行比對演算法
        ignore_options = {
            "ignore_whitespace": True,
            "ignore_punctuation": True,
            "ignore_case": True,
            "ignore_linebreaks": True,
        }

        result = compare_documents(
            word_data,
            pdf_data,
            ignore_options=ignore_options,
            comparison_mode="hybrid" if use_ai else "exact",
            similarity_threshold=similarity_threshold,
            ai_instance=ai_instance
        )

        # 5. 顯示結果
        if result:
            st.success(f"比對完成，共處理 {len(result)} 組段落！")
            for item in result:
                st.markdown("### 📌 差異段落")
                st.markdown(f"- **原始：** {item['doc1_text']}")
                st.markdown(f"- **PDF：** {item['doc2_text']}")
                st.markdown(f"- **相似度：** {item['similarity']:.2f}")
                st.markdown("---")
        else:
            st.warning("未比對到有效段落，請檢查文件內容是否正確。")


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
                return False
        elif self.model_name in ["OpenAI API", "Anthropic API", "Gemini API", "Qwen API"]:
            # 檢查API模型
            return self.api_key is not None and len(self.api_key) > 0
        return False
    
    def match_paragraphs(self, source_paragraphs, target_paragraphs):
        """使用生成式AI匹配段落"""
        if not self.is_available:
            return None
        
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
        
        except Exception as e:
            st.error(f"BERT匹配失敗: {e}")
            return None
    
    def _match_with_mpnet(self, source_paragraphs, target_paragraphs):
        """使用MPNet中文模型匹配段落"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # 加載模型
            model_name = "shibing624/text2vec-base-chinese"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
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
        
        except Exception as e:
            st.error(f"MPNet匹配失敗: {e}")
            return None
    
    def _match_with_roberta(self, source_paragraphs, target_paragraphs):
        """使用RoBERTa中文模型匹配段落"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # 加載模型
            model_name = "hfl/chinese-roberta-wwm-ext"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
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
        
        except Exception as e:
            st.error(f"RoBERTa匹配失敗: {e}")
            return None
    
    def _match_with_openai(self, source_paragraphs, target_paragraphs):
        """使用OpenAI API匹配段落"""
        try:
            import openai
            
            # 設置API密鑰
            openai.api_key = self.api_key
            
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
            
            # 調用API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一個專業的文本匹配助手，擅長分析文本相似度和找出最佳匹配。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            # 解析結果
            result_text = response.choices[0].message.content
            
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
            st.error(f"OpenAI API匹配失敗: {e}")
            return None
    
def _match_with_anthropic(self, source_paragraphs, target_paragraphs):
    """使用Anthropic API匹配段落"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)

        # 準備資料
        source_texts = [p['content'] for p in source_paragraphs]
        target_texts = [p['content'] for p in target_paragraphs]

        # TODO: 加入Anthropic API調用邏輯
        return []  # 暫時返回空結果作為占位
    except Exception as e:
        st.error(f"Anthropic API 匹配失敗：{e}")
        return None

