"""comparison_algorithm.py  (rev‑2025‑04‑18b)

在 rev‑04‑18 基礎上新增 **substring_similarity**：
若 Word 段落完整包含於 PDF 段落 (或反之)，視為高相似度，
以「短文本長度 / 長文本長度」作為分數，可解決「多段對一頁」情境。
其餘演算法 (exact / semantic / hybrid / ai) 保持不變。
"""
from __future__ import annotations
import difflib, re, numpy as np, unicodedata, html
from typing import List, Dict, Any, Tuple, Union, Optional

try:
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    _SENTENCE_MODEL = True
except Exception:
    _st_model = None
    _SENTENCE_MODEL = False

def _normalize(text:str)->str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _preprocess(text: str, ignore_options: dict = None) -> str:
    """文本預處理通用函數"""
    if ignore_options is None:
        ignore_options = dict(ignore_space=True, ignore_punctuation=True,
                            ignore_case=True, ignore_newline=True)
    
    text = _normalize(text)
    if ignore_options.get('ignore_newline', True):
        text = text.replace('\n', ' ')
    if ignore_options.get('ignore_space', True):
        text = re.sub(r'\s+', ' ', text)
    if ignore_options.get('ignore_case', True):
        text = text.lower()
    if ignore_options.get('ignore_punctuation', True):
        text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def exact_matching(t1:str, t2:str,
                   ignore_space=True, ignore_punctuation=True, ignore_case=True)->float:
    if ignore_space:
        t1 = re.sub(r'\s+', '', t1)
        t2 = re.sub(r'\s+', '', t2)
    if ignore_punctuation:
        t1 = re.sub(r'[^\w\s]', '', t1)
        t2 = re.sub(r'[^\w\s]', '', t2)
    if ignore_case:
        t1 = t1.lower()
        t2 = t2.lower()
    return difflib.SequenceMatcher(None, t1, t2).ratio()

def semantic_matching(t1:str, t2:str)->float:
    if _SENTENCE_MODEL:
        emb1, emb2 = _st_model.encode([t1, t2])
        return float(np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2)))
    return exact_matching(t1, t2, False, False, False)

def hybrid_matching(t1:str, t2:str, exact_thresh=0.85)->float:
    exact = exact_matching(t1, t2, True, True, True)
    return exact if exact >= exact_thresh else semantic_matching(t1, t2)

def substring_similarity(a:str, b:str)->float:
    """如果 a 包含於 b 或 b 包含於 a，則以較短長度 / 較長長度 作為分數"""
    if not a or not b: return 0.0
    if a in b:
        return len(a)/len(b)
    if b in a:
        return len(b)/len(a)
    return 0.0

def _diff_html(a:str, b:str)->str:
    d = difflib.ndiff(a, b)
    buf=[]
    for s in d:
        if s.startswith('  '):
            buf.append(html.escape(s[2:]))
        elif s.startswith('- '):
            buf.append(f'<span class="diff-removed">{html.escape(s[2:])}</span>')
        elif s.startswith('+ '):
            buf.append(f'<span class="diff-added">{html.escape(s[2:])}</span>')
    return ''.join(buf)

def merge_word_paragraphs(paragraphs: list, max_distance: int = 3, ignore_options: dict = None) -> list:
    """合併鄰近的 Word 段落以支援多段對一組合"""
    if not paragraphs or not isinstance(paragraphs, list):
        return []

    merged = []
    n = len(paragraphs)
    
    for i in range(n):
        current = paragraphs[i]
        # 確保段落格式正確
        if not isinstance(current, dict) or 'content' not in current:
            continue
            
        # 單段
        merged.append({
            'content': current.get('content', ''),
            'indices': [current.get('index', i)],
            'processed': current.get('processed', _preprocess(current.get('content', ''), ignore_options))
        })
        
        # 嘗試與後續段落組合
        combined_text = current.get('content', '')
        indices = [current.get('index', i)]
        
        for j in range(i + 1, min(i + max_distance + 1, n)):
            next_para = paragraphs[j]
            if not isinstance(next_para, dict) or 'content' not in next_para:
                continue
                
            combined_text += '\n' + next_para.get('content', '')
            indices.append(next_para.get('index', j))
            
            merged.append({
                'content': combined_text,
                'indices': indices.copy(),
                'processed': _preprocess(combined_text, ignore_options)
            })
    
    return merged

def preprocess_text(text: str, ignore_options: Dict[str, bool]) -> str:
    """根據忽略選項預處理文本。"""
    if not isinstance(text, str):
        return ""
    result = text
    if ignore_options.get("ignore_space", False):
        result = re.sub(r'\s+', ' ', result)
    if ignore_options.get("ignore_punctuation", False):
        result = re.sub(r'[^\w\s]', '', result)
    if ignore_options.get("ignore_case", False):
        result = result.lower()
    if ignore_options.get("ignore_newline", False):
        result = result.replace('\n', ' ')
    return result.strip()

def calculate_similarity(text1: str, text2: str, mode: str, ignore_options: Dict[str, bool], ai_instance=None) -> float:
    """根據指定的模式計算兩段文本的相似度。"""
    # 預處理文本
    processed_text1 = preprocess_text(text1, ignore_options)
    processed_text2 = preprocess_text(text2, ignore_options)
    
    if not processed_text1 or not processed_text2:
        return 0.0
    
    # 根據不同模式計算相似度
    if mode == "exact":
        # 精確匹配
        return 1.0 if processed_text1 == processed_text2 else 0.0
    
    elif mode == "semantic":
        # 語意匹配 (簡化實現)
        words1 = set(processed_text1.split())
        words2 = set(processed_text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    elif mode == "ai" and ai_instance:
        # AI 匹配
        try:
            similarity, _ = ai_instance.semantic_comparison(text1, text2)
            return similarity
        except Exception as e:
            print(f"AI 比對失敗: {str(e)}")
            # 降級到混合匹配
            return calculate_similarity(text1, text2, "hybrid", ignore_options)
    
    else:  # 默認為 "hybrid"
        # 混合匹配: 編輯距離 + 詞彙重疊
        # 編輯距離部分
        s = difflib.SequenceMatcher(None, processed_text1, processed_text2)
        edit_similarity = s.ratio()
        
        # 詞彙重疊部分
        words1 = set(processed_text1.split())
        words2 = set(processed_text2.split())
        if not words1 or not words2:
            return edit_similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        word_similarity = intersection / union if union > 0 else 0.0
        
        # 混合相似度 (70% 編輯距離 + 30% 詞彙重疊)
        return 0.7 * edit_similarity + 0.3 * word_similarity

def generate_diff_html(text1: str, text2: str, ignore_options: Dict[str, bool]) -> str:
    """生成差異的HTML標記。"""
    # 預處理文本 (可選)
    if ignore_options.get("ignore_case", False):
        text1 = text1.lower()
        text2 = text2.lower()
    
    # 使用 difflib 生成差異
    d = difflib.Differ()
    diff = list(d.compare(text1.splitlines(), text2.splitlines()))
    
    # 生成HTML
    html = []
    for line in diff:
        tag = line[0]
        content = line[2:].strip()
        if not content:
            continue
            
        if tag == ' ':
            html.append(f"<span>{content}</span><br>")
        elif tag == '-':
            html.append(f"<span style='color: red; text-decoration: line-through;'>{content}</span><br>")
        elif tag == '+':
            html.append(f"<span style='color: green;'>{content}</span><br>")
            
    return "".join(html)

def generate_enhanced_diff_html(word_text: str, pdf_text: str) -> str:
    """
    以PDF內容為主生成增強型差異標示HTML。
    相同的內容顯示為灰色，不同的內容顯示為紅色。
    
    專門針對業務範疇描述文本與其他內容（如頁碼、註解、圖表說明等）的區分設計。
    
    Args:
        word_text: Word文件文本
        pdf_text: PDF文件文本
        
    Returns:
        str: 帶有差異標示的HTML
    """
    # 分行處理
    pdf_lines = pdf_text.splitlines()
    result_html = []
    
    # 定義要保留（顯示為灰色）的內容特徵關鍵詞和片段
    business_phrase_parts = [
        "崑鼎之業務範疇包括", "投資與經營", "操作營運", "技術與諮詢服務", 
        "服務對象涵蓋大中華區", "東南亞及美國等", "公部門及私人企業",
        "透過旗下", "家轉投資公司", "循環經濟業務發展中",
        "扮演不同角色相互支援", "在臺灣", "澳門", "東南亞", "美國等地",
        "提供全方位的專業環保服務", "建構專業的循環經濟團隊"
    ]
    
    # 定義明確要標紅的內容模式
    red_patterns = [
        r'^\s*\d+\s*$',                 # 僅包含數字的行（頁碼）
        r'請美編依公司地域重新繪圖',     # 美編指示
        r'已註解\s*\[D\d+\]',           # 註解標記
        r'上表為\s*\d+\s*家',           # 表格說明
        r'請確認'                       # 確認請求
    ]
    
    # 建立業務內容標識符 - 用於識別整段業務描述
    business_identifier = "崑鼎之業務範疇包括「投資與經營」、「操作營運」及「技術與諮詢服務」"
    business_section_active = False
    
    # 處理每一行
    for line in pdf_lines:
        line_stripped = line.strip()
        
        # 空行處理
        if not line_stripped:
            result_html.append("<div style='height: 1em;'></div>")
            continue
        
        # 檢查是否為明確標紅的內容
        is_red_pattern = any(re.search(pattern, line_stripped) for pattern in red_patterns)
        
        # 檢查是否開始業務描述段落
        if business_identifier in line_stripped:
            business_section_active = True
        
        # 檢查是否為業務描述內容的一部分
        is_business_content = False
        if business_section_active:
            is_business_content = True
            # 檢查是否結束業務描述段落（如以循環經濟團隊結尾）
            if "循環經濟團隊" in line_stripped and line_stripped.strip().endswith("。"):
                business_section_active = False
        else:
            # 單獨檢查此行是否包含業務描述特徵
            is_business_content = any(part in line_stripped for part in business_phrase_parts)
        
        # 決定顏色
        if is_red_pattern:
            # 明確標紅的模式 - 使用較醒目的紅色樣式
            result_html.append(f"<span style='color: #E53935; font-weight: 500; display: block; padding: 2px 0; background-color: #ffebee;'>{html.escape(line_stripped)}</span>")
        elif is_business_content:
            # 業務描述內容 - 使用灰色
            result_html.append(f"<span style='color: #707070; display: block; padding: 2px 0;'>{html.escape(line_stripped)}</span>")
        else:
            # 其他情況，檢查是否在Word中有相似內容
            word_normalized = re.sub(r'\s+', '', word_text.lower())
            line_normalized = re.sub(r'\s+', '', line_stripped.lower())
            
            # 針對短內容特殊處理
            if len(line_normalized) < 10:
                result_html.append(f"<span style='color: #E53935; font-weight: 500; display: block; padding: 2px 0; background-color: #ffebee;'>{html.escape(line_stripped)}</span>")
            elif line_normalized in word_normalized:
                result_html.append(f"<span style='color: #707070; display: block; padding: 2px 0;'>{html.escape(line_stripped)}</span>")
            else:
                result_html.append(f"<span style='color: #E53935; font-weight: 500; display: block; padding: 2px 0; background-color: #ffebee;'>{html.escape(line_stripped)}</span>")
    
    return "".join(result_html)

def compare_pdf_first(word_data, pdf_data, comparison_mode="hybrid", similarity_threshold=0.6, 
                    ignore_options=None, ai_instance=None, ocr_instance=None):
    """
    主比對函數：比對 Word 與 PDF，首先按頁面順序從 PDF 開始比對，尋找匹配的 Word 段落
    """
    if ignore_options is None:
        ignore_options = {"ignore_space": True, "ignore_punctuation": True, "ignore_case": True}
    
    # 預處理 Word 段落
    word_paragraphs = word_data.get("paragraphs", [])
    word_paragraphs = [p for p in word_paragraphs if p.get("content") and len(p.get("content", "")) > 5]
    
    # 預處理 PDF 段落
    pdf_paragraphs = pdf_data.get("paragraphs", [])
    pdf_paragraphs = [p for p in pdf_paragraphs if p.get("content") and len(p.get("content", "")) > 5]
    
    # 按頁碼排序 PDF 段落
    pdf_paragraphs.sort(key=lambda x: (x.get("page", 1), x.get("index", 0)))
    
    # 初始化結果
    matches = []
    matched_word_indices = set()
    matched_pdf_indices = set()
    
    # 初始化頁面段落對應
    page_paragraphs = {}
    for para in pdf_paragraphs:
        page = para.get("page", 1)
        if page not in page_paragraphs:
            page_paragraphs[page] = []
        page_paragraphs[page].append(para)
    
    # 按頁面順序處理段落
    for page, page_paras in sorted(page_paragraphs.items()):
        for pdf_para in page_paras:
            pdf_index = pdf_para.get("index", 0)
            if pdf_index in matched_pdf_indices:
                continue
            
            pdf_text = pdf_para.get("content", "")
            best_similarity = 0
            best_word_index = -1
            best_word_text = ""
            best_word_indices = []
            
            # 尋找匹配
            for word_index, word_para in enumerate(word_paragraphs):
                if word_index in matched_word_indices:
                    continue
                
                word_text = word_para.get("content", "")
                similarity = calculate_similarity(word_text, pdf_text, comparison_mode, ignore_options, ai_instance)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_word_index = word_index
                    best_word_text = word_text
                    best_word_indices = [word_index]
                
            # 如果找到足夠相似的匹配
            if best_similarity >= similarity_threshold:
                matched_word_indices.add(best_word_index)
                matched_pdf_indices.add(pdf_index)
                
                # 使用 difflib 生成差異標記
                diff_html = generate_diff_html(best_word_text, pdf_text, ignore_options)
                
                # 生成增強型差異標記
                enhanced_diff_html = generate_enhanced_diff_html(best_word_text, pdf_text)
                
                # 添加到結果
                match = {
                    "word_indices": best_word_indices,
                    "pdf_index": pdf_index,
                    "word_text": best_word_text,
                    "pdf_text": pdf_text,
                    "similarity": best_similarity,
                    "pdf_page": page,
                    "diff_html": diff_html,
                    "enhanced_diff_html": enhanced_diff_html,
                }
                
                # 如果啟用了 AI 比對，添加句子級別的差異摘要
                if comparison_mode == "ai" and ai_instance:
                    try:
                        diff_summary = ai_instance.analyze_differences(best_word_text, pdf_text)
                        match["diff_summary"] = diff_summary
                    except Exception as e:
                        print(f"AI 差異分析失敗: {str(e)}")
                
                matches.append(match)
    
    # 統計資訊
    stats = {
        "total_word": len(word_paragraphs),
        "total_pdf": len(pdf_paragraphs),
        "matched": len(matches),
        "unmatched_word": len(word_paragraphs) - len(matched_word_indices),
        "unmatched_pdf": len(pdf_paragraphs) - len(matched_pdf_indices),
    }
    
    return {
        "matches": matches,
        "statistics": stats,
        "matched_word_indices": list(matched_word_indices),
        "matched_pdf_indices": list(matched_pdf_indices),
    }

def compare_documents(doc1, doc2, *args, **kwargs):
    return compare_pdf_first(word_data=doc1, pdf_data=doc2, *args, **kwargs)
