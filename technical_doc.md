# 文稿比對系統技術文件

## 系統架構

文稿比對系統採用模組化設計，主要由以下幾個部分組成：

1. **文本提取模組**：負責從Word和PDF文件中提取文本內容
2. **比對演算法模組**：實現多種比對算法，包括精確比對、語意比對、混合比對等
3. **OCR處理模組**：整合多種OCR引擎，提高PDF文本提取質量
4. **生成式AI模組**：使用AI模型輔助段落匹配和結果分析
5. **UI介面模組**：提供使用者介面和視覺化展示

## 核心模組說明

### 文本提取模組

文本提取模組使用enhanced_extraction.py實現，主要功能包括：

- Word文件文本提取：使用python-docx庫提取段落和表格
- PDF文件文本提取：使用PyMuPDF和pdfplumber提取文本
- OCR文本提取：使用多種OCR引擎提取圖像化文本
- 表格提取：使用多種方法提取表格內容

### 比對演算法模組

比對演算法模組實現了多種比對算法，包括：

- **精確比對**：基於difflib.SequenceMatcher，計算字符級別的相似度
- **語意比對**：使用sentence-transformers計算語義相似度
- **混合比對**：結合精確比對和語意比對，平衡精確度和語義理解
- **上下文感知比對**：考慮段落的上下文關係，提供更準確的匹配結果
- **生成式AI比對**：使用AI模型進行段落匹配

### OCR處理模組

OCR處理模組整合了多種OCR引擎，包括：

- **Tesseract**：開源OCR引擎，支援多種語言
- **EasyOCR**：高精度OCR引擎，支援多種語言和複雜排版
- **Qwen API**：阿里巴巴的Qwen模型API，提供最高精度的OCR和表格識別

### 生成式AI模組

生成式AI模組支援多種AI模型，包括：

- **本地模型**：BERT、MPNet、RoBERTa等
- **API模型**：OpenAI、Anthropic、Gemini、Qwen等

主要功能包括：

- 段落匹配：使用AI模型找出最佳匹配的段落
- 結果分析：分析比對結果，提供專業見解
- 差異分類：對差異進行分類和評估

### UI介面模組

UI介面模組使用Streamlit實現，主要功能包括：

- 文件上傳：支援上傳Word和PDF文件
- 參數設置：設置比對模式、相似度閾值、忽略選項等
- 結果展示：顯示比對結果和差異
- PDF預覽：顯示PDF頁面並標記差異位置

## 資料流程

1. 使用者上傳Word和PDF文件
2. 系統從文件中提取文本內容
3. 系統根據選擇的比對模式進行比對
4. 系統生成比對報告
5. 系統顯示比對結果和差異
6. 如果啟用AI分析，系統使用AI模型分析結果
7. 系統生成PDF頁面預覽並標記差異

## 演算法說明

### 精確比對演算法

```python
def exact_matching(text1, text2, ignore_space=True, ignore_punctuation=True, ignore_case=True, ignore_newline=True):
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
```

### 語意比對演算法

```python
def semantic_matching(text1, text2, model=None):
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
```

### 上下文感知匹配演算法

```python
def context_aware_matching(text1, text2, context1=None, context2=None, ignore_options=None, model=None):
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
```

## 依賴庫

- streamlit：用於構建Web應用程式
- python-docx：用於處理Word文件
- PyMuPDF (fitz)：用於處理PDF文件
- pdfplumber：用於提取PDF表格
- pandas：用於數據處理
- numpy：用於數值計算
- PIL：用於圖像處理
- sentence-transformers：用於語義比對
- pytesseract：用於OCR文本提取
- easyocr：用於高精度OCR
- requests：用於API請求

## 擴展與優化

### 擴展方向

1. **多語言支持**：增加對更多語言的支持
2. **批量處理**：支援批量處理多個文件
3. **雲端存儲**：整合雲端存儲服務
4. **協作功能**：增加多人協作功能
5. **自動校正**：自動修正差異

### 優化方向

1. **性能優化**：提高大文件處理速度
2. **記憶體優化**：減少記憶體使用
3. **UI優化**：改進使用者介面
4. **算法優化**：提高比對精確度
5. **OCR優化**：提高OCR文本提取質量
