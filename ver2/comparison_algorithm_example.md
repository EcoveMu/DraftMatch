# 比對算法示例代碼

## 1. 精確比對示例 (使用difflib)

```python
def exact_matching(text1, text2, ignore_space=True, ignore_punctuation=True, ignore_case=True):
    """
    使用difflib進行精確比對
    
    參數:
        text1: 原始文本
        text2: 比對文本
        ignore_space: 是否忽略空格
        ignore_punctuation: 是否忽略標點符號
        ignore_case: 是否忽略大小寫
    
    返回:
        similarity: 相似度 (0.0-1.0)
        diff: 差異列表
    """
    import difflib
    import re
    
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
    
    # 計算相似度
    matcher = difflib.SequenceMatcher(None, text1, text2)
    similarity = matcher.ratio()
    
    # 生成差異
    diff = list(difflib.ndiff(text1.splitlines(), text2.splitlines()))
    
    return similarity, diff
```

## 2. 語意比對示例 (使用Sentence-BERT)

```python
def semantic_matching(text1, text2):
    """
    使用Sentence-BERT進行語意比對
    
    參數:
        text1: 原始文本
        text2: 比對文本
    
    返回:
        similarity: 語意相似度 (0.0-1.0)
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # 加載模型
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 計算句子嵌入
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    
    # 計算餘弦相似度
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    return similarity
```

## 3. 混合比對示例

```python
def hybrid_matching(text1, text2, exact_threshold=0.8, semantic_threshold=0.8):
    """
    結合精確比對和語意比對
    
    參數:
        text1: 原始文本
        text2: 比對文本
        exact_threshold: 精確比對閾值
        semantic_threshold: 語意比對閾值
    
    返回:
        is_similar: 是否相似
        exact_similarity: 精確比對相似度
        semantic_similarity: 語意比對相似度
        diff: 差異列表
    """
    # 精確比對
    exact_similarity, diff = exact_matching(text1, text2)
    
    # 如果精確比對相似度高，直接返回結果
    if exact_similarity >= exact_threshold:
        return True, exact_similarity, None, diff
    
    # 否則進行語意比對
    semantic_similarity = semantic_matching(text1, text2)
    
    # 根據語意相似度判斷是否相似
    is_similar = semantic_similarity >= semantic_threshold
    
    return is_similar, exact_similarity, semantic_similarity, diff
```

## 4. 分段比對示例

```python
def segment_matching(doc1, doc2, segment_type='paragraph'):
    """
    分段比對文檔
    
    參數:
        doc1: 原始文檔 (段落列表)
        doc2: 比對文檔 (段落列表)
        segment_type: 分段類型 ('paragraph', 'sentence')
    
    返回:
        results: 比對結果列表
    """
    results = []
    
    # 對每個段落進行比對
    for i, para1 in enumerate(doc1):
        best_match = None
        best_similarity = 0
        best_index = -1
        
        # 在doc2中尋找最佳匹配
        for j, para2 in enumerate(doc2):
            # 混合比對
            is_similar, exact_sim, semantic_sim, diff = hybrid_matching(para1, para2)
            
            # 計算綜合相似度
            similarity = max(exact_sim, semantic_sim) if semantic_sim else exact_sim
            
            # 更新最佳匹配
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = para2
                best_index = j
        
        # 記錄比對結果
        if best_similarity > 0:
            is_similar, exact_sim, semantic_sim, diff = hybrid_matching(para1, best_match)
            results.append({
                'original_index': i,
                'original_text': para1,
                'matched_index': best_index,
                'matched_text': best_match,
                'exact_similarity': exact_sim,
                'semantic_similarity': semantic_sim,
                'is_similar': is_similar,
                'diff': diff
            })
        else:
            # 未找到匹配
            results.append({
                'original_index': i,
                'original_text': para1,
                'matched_index': -1,
                'matched_text': None,
                'exact_similarity': 0,
                'semantic_similarity': 0,
                'is_similar': False,
                'diff': []
            })
    
    return results
```

## 5. OCR整合示例

```python
def extract_text_with_ocr(pdf_path, ocr_engine='openocr', lang='chi_tra+eng'):
    """
    使用OCR提取PDF文本
    
    參數:
        pdf_path: PDF文件路徑
        ocr_engine: OCR引擎 ('tesseract', 'openocr', 'azure', 'google')
        lang: 語言
    
    返回:
        text_results: 提取的文本
    """
    import os
    import fitz  # PyMuPDF
    from PIL import Image
    import tempfile
    
    # 創建臨時目錄
    temp_dir = tempfile.mkdtemp()
    
    # 使用PyMuPDF將PDF頁面轉換為圖像
    doc = fitz.open(pdf_path)
    text_results = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
        img_path = os.path.join(temp_dir, f'page_{page_num+1}.png')
        pix.save(img_path)
        
        # 根據選擇的OCR引擎提取文本
        if ocr_engine == 'tesseract':
            import pytesseract
            text = pytesseract.image_to_string(Image.open(img_path), lang=lang)
        
        elif ocr_engine == 'openocr':
            # 使用OpenOCR (需要安裝相應的庫)
            # 這裡是示例代碼，實際使用時需要根據OpenOCR的API進行調整
            try:
                import easyocr
                reader = easyocr.Reader(['ch_tra', 'en'])
                result = reader.readtext(img_path)
                text = '\n'.join([item[1] for item in result])
            except ImportError:
                # 如果OpenOCR不可用，回退到Tesseract
                import pytesseract
                text = pytesseract.image_to_string(Image.open(img_path), lang=lang)
        
        elif ocr_engine == 'azure':
            # 使用Azure Form Recognizer (需要API Key)
            # 這裡是示例代碼，實際使用時需要根據Azure API進行調整
            text = "Azure OCR integration placeholder"
        
        elif ocr_engine == 'google':
            # 使用Google Cloud Vision (需要API Key)
            # 這裡是示例代碼，實際使用時需要根據Google API進行調整
            text = "Google OCR integration placeholder"
        
        else:
            # 默認使用Tesseract
            import pytesseract
            text = pytesseract.image_to_string(Image.open(img_path), lang=lang)
        
        text_results.append({
            'page_num': page_num + 1,
            'text': text
        })
    
    return text_results
```
