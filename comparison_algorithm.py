"""comparison_algorithm.py  (rev‑2025‑04‑18b)

在 rev‑04‑18 基礎上新增 **substring_similarity**：
若 Word 段落完整包含於 PDF 段落 (或反之)，視為高相似度，
以「短文本長度 / 長文本長度」作為分數，可解決「多段對一頁」情境。
其餘演算法 (exact / semantic / hybrid / ai) 保持不變。
"""
from __future__ import annotations
import difflib, re, numpy as np, unicodedata, html
from typing import List, Dict, Any, Tuple, Union
import pandas as pd
import streamlit as st

_SENTENCE_MODEL = False
model = None

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    _SENTENCE_MODEL = True
except Exception as e:
    print(f"無法加載語義模型: {str(e)}，將使用精確匹配替代")
    model = None
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

def split_into_sentences(text: str) -> List[str]:
    """將文本拆分為句子列表"""
    # 處理常見中文標點符號和英文標點符號
    text = re.sub(r'([。！？\.!?][\'"」』]?)((?![\s\d]))', r'\1\n\2', text)
    # 如果沒有明確的句子邊界，則按50個字符左右的段落切分
    if '\n' not in text and len(text) > 50:
        sentences = []
        for i in range(0, len(text), 50):
            end = min(i + 50, len(text))
            if end < len(text) and not text[end].isspace():
                # 嘗試在空格處切分
                space_pos = text.rfind(' ', i, end)
                if space_pos > i:
                    end = space_pos + 1
            sentences.append(text[i:end])
        return sentences
    else:
        return [s.strip() for s in text.split('\n') if s.strip()]

def exact_matching(source_text: str, target_text: str) -> float:
    """精確匹配，返回相似度分數"""
    if not source_text or not target_text:
        return 0.0
    
    if source_text == target_text:
        return 1.0
    
    # 使用difflib計算相似度
    similarity = difflib.SequenceMatcher(None, source_text, target_text).ratio()
    return similarity

def semantic_matching(source_text: str, target_text: str) -> float:
    """語義匹配，返回相似度分數"""
    if not source_text or not target_text:
        return 0.0
    
    if model is None or not _SENTENCE_MODEL:
        # 如果模型加載失敗，退回到精確匹配
        return exact_matching(source_text, target_text)
    
    try:
        # 編碼文本
        source_embedding = model.encode(source_text, convert_to_tensor=True)
        target_embedding = model.encode(target_text, convert_to_tensor=True)
        
        # 計算餘弦相似度
        cosine_similarity = np.dot(source_embedding, target_embedding) / (
            np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding)
        )
        
        return float(cosine_similarity)
    except Exception as e:
        print(f"語義匹配時發生錯誤: {str(e)}")
        # 出錯時退回到精確匹配
        return exact_matching(source_text, target_text)

def substring_similarity(source_text: str, target_text: str) -> float:
    """計算子串包含關係的相似度"""
    if not source_text or not target_text:
        return 0.0
    
    source_lower = source_text.lower()
    target_lower = target_text.lower()
    
    # 檢查是否為子串關係
    if source_lower in target_lower:
        return len(source_lower) / len(target_lower)
    elif target_lower in source_lower:
        return len(target_lower) / len(source_lower)
    
    # 檢查最長共同子序列
    matcher = difflib.SequenceMatcher(None, source_lower, target_lower)
    common_blocks = matcher.get_matching_blocks()
    
    # 計算最長共同子序列的長度
    total_match_size = sum(block.size for block in common_blocks if block.size > 0)
    max_length = max(len(source_lower), len(target_lower))
    
    if max_length == 0:
        return 0.0
    
    return total_match_size / max_length

def hybrid_matching(source_text: str, target_text: str) -> Tuple[float, str]:
    """混合匹配策略，結合精確匹配和語義匹配"""
    exact_sim = exact_matching(source_text, target_text)
    substring_sim = substring_similarity(source_text, target_text)
    
    # 如果精確相似度或子串相似度較高，則不需要使用語義匹配
    if exact_sim > 0.8 or substring_sim > 0.8:
        if exact_sim >= substring_sim:
            return exact_sim, "exact"
        else:
            return substring_sim, "substring"
    
    # 否則使用語義匹配
    semantic_sim = semantic_matching(source_text, target_text)
    
    # 返回最高的相似度分數和匹配類型
    if exact_sim >= max(semantic_sim, substring_sim):
        return exact_sim, "exact"
    elif semantic_sim >= substring_sim:
        return semantic_sim, "semantic"
    else:
        return substring_sim, "substring"

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
    """合併鄰近的 Word 段落以支援多段對一組合，確保目錄項目也被納入比對範圍"""
    if not paragraphs or not isinstance(paragraphs, list):
        return []

    merged = []
    n = len(paragraphs)
    
    for i in range(n):
        current = paragraphs[i]
        # 確保段落格式正確
        if not isinstance(current, dict) or 'content' not in current:
            continue
            
        # 單段（包含所有類型段落，包括目錄項目）
        merged.append({
            'content': current.get('content', ''),
            'indices': [current.get('index', i)],
            'processed': current.get('processed', _preprocess(current.get('content', ''), ignore_options)),
            'paragraph_type': current.get('paragraph_type', 'normal')  # 保留段落類型信息
        })
        
        # 嘗試與後續段落組合
        combined_text = current.get('content', '')
        indices = [current.get('index', i)]
        paragraph_types = [current.get('paragraph_type', 'normal')]
        
        for j in range(i + 1, min(i + max_distance + 1, n)):
            next_para = paragraphs[j]
            if not isinstance(next_para, dict) or 'content' not in next_para:
                continue
                
            combined_text += '\n' + next_para.get('content', '')
            indices.append(next_para.get('index', j))
            paragraph_types.append(next_para.get('paragraph_type', 'normal'))
            
            merged.append({
                'content': combined_text,
                'indices': indices.copy(),
                'processed': _preprocess(combined_text, ignore_options),
                'paragraph_types': paragraph_types.copy()  # 保留所有段落類型信息
            })
    
    return merged

def compare_pdf_first(word_data: dict, pdf_data: dict,
                     comparison_mode: str = 'hybrid',
                     similarity_threshold: float = 0.6,
                     ignore_options: dict = None,
                     ai_instance = None) -> dict:
    """改進的比對算法，支援多段對一頁比對，確保目錄項目也參與比對"""
    # 輸入驗證
    if not isinstance(word_data, dict) or not isinstance(pdf_data, dict):
        raise ValueError("word_data 和 pdf_data 必須是字典格式")
    
    if 'paragraphs' not in word_data or 'paragraphs' not in pdf_data:
        raise ValueError("word_data 和 pdf_data 必須包含 'paragraphs' 鍵")
        
    if not isinstance(word_data['paragraphs'], list) or not isinstance(pdf_data['paragraphs'], list):
        raise ValueError("paragraphs 必須是列表格式")

    if ignore_options is None:
        ignore_options = dict(ignore_space=True, ignore_punctuation=True,
                            ignore_case=True, ignore_newline=True)

    # 預處理所有段落（包括目錄項目）
    for p in pdf_data['paragraphs']:
        if isinstance(p, dict) and 'content' in p:
            p['processed'] = _preprocess(p['content'], ignore_options)
    
    for p in word_data['paragraphs']:
        if isinstance(p, dict) and 'content' in p:
            p['processed'] = _preprocess(p['content'], ignore_options)

    # 生成 Word 段落的各種組合
    word_combinations = merge_word_paragraphs(word_data['paragraphs'], 
                                           ignore_options=ignore_options)
    
    matches = []
    used_word_indices = set()
    
    for pdf_para in pdf_data['paragraphs']:
        best_match = None
        best_sim = 0.0
        match_type = ""
        
        for word_combo in word_combinations:
            # 跳過已使用的段落組合
            if any(idx in used_word_indices for idx in word_combo['indices']):
                continue
                
            # 計算相似度
            if comparison_mode == 'hybrid':
                sim, match_type = hybrid_matching(pdf_para['processed'], word_combo['processed'])
            elif comparison_mode == 'semantic':
                sim = semantic_matching(pdf_para['processed'], word_combo['processed'])
                match_type = "semantic"
            elif comparison_mode == 'ai' and ai_instance:
                sim, match_type = ai_instance.semantic_comparison(pdf_para['content'], word_combo['content'])
            else:  # exact
                sim = exact_matching(pdf_para['processed'], word_combo['processed'])
                match_type = "exact"
                
            if sim > best_sim and sim >= similarity_threshold:
                best_sim = sim
                best_match = word_combo
        
        if best_match:
            # 檢查是否包含段落類型信息
            paragraph_type = best_match.get('paragraph_type', 'normal')
            paragraph_types = best_match.get('paragraph_types', [paragraph_type])
            
            matches.append({
                'pdf_index': pdf_para['index'],
                'pdf_page': pdf_para.get('page'),
                'pdf_text': pdf_para['content'],
                'pdf_paragraph_type': pdf_para.get('paragraph_type', 'normal'),
                'word_indices': best_match['indices'],
                'word_text': best_match['content'],
                'word_paragraph_types': paragraph_types,
                'similarity': best_sim,
                'diff_html': _diff_html(best_match['content'], pdf_para['content']),
                'match_type': match_type
            })
            used_word_indices.update(best_match['indices'])
    
    unmatched_pdf=[p for p in pdf_data['paragraphs']
                   if p['index'] not in {m['pdf_index'] for m in matches}]
    unmatched_word=[p for p in word_data['paragraphs']
                    if p['index'] not in used_word_indices]

    stats=dict(
        total_pdf=len(pdf_data['paragraphs']),
        total_word=len(word_data['paragraphs']),
        matched=len(matches),
        unmatched_pdf=len(unmatched_pdf),
        unmatched_word=len(unmatched_word),
        match_rate_pdf = len(matches)/len(pdf_data['paragraphs']) if pdf_data['paragraphs'] else 0,
        match_rate_word = len(used_word_indices)/len(word_data['paragraphs']) if word_data['paragraphs'] else 0
    )
    return dict(matches=matches,
                unmatched_pdf=unmatched_pdf,
                unmatched_word=unmatched_word,
                statistics=stats)

def compare_documents(word_paragraphs: List[Dict], 
                     pdf_paragraphs: List[Dict], 
                     similarity_threshold: float = 0.7,
                     matching_method: str = "hybrid") -> Dict:
    """
    比較Word和PDF文檔的段落，支持目錄項目的處理
    
    Args:
        word_paragraphs: Word文檔的段落列表，每個段落是包含'index'、'content'和可選'type'的字典
        pdf_paragraphs: PDF文檔的段落列表，每個段落是包含'index'、'content'、'page_num'和可選'type'的字典
        similarity_threshold: 相似度閾值，默認0.7
        matching_method: 比對方法，'hybrid'、'exact'或'semantic'
        
    Returns:
        包含匹配結果的字典
    """
    # 將輸入格式轉換為compare_pdf_first接受的格式
    word_data = {
        'paragraphs': [
            {
                'index': p.get('index', i),
                'content': p.get('content', ''),
                'paragraph_type': p.get('type', 'normal')
            } for i, p in enumerate(word_paragraphs)
        ]
    }
    
    pdf_data = {
        'paragraphs': [
            {
                'index': p.get('index', i),
                'content': p.get('content', ''),
                'page': p.get('page_num', -1),
                'paragraph_type': p.get('type', 'normal')
            } for i, p in enumerate(pdf_paragraphs)
        ]
    }
    
    # 使用現有的比對函數
    return compare_pdf_first(
        word_data=word_data, 
        pdf_data=pdf_data,
        comparison_mode=matching_method,
        similarity_threshold=similarity_threshold
    )

# 保留原來的compare_documents函數為了向後兼容
def compare_documents_legacy(doc1, doc2, *args, **kwargs):
    return compare_pdf_first(word_data=doc1, pdf_data=doc2, *args, **kwargs)

def compute_statistics(matches: List[Dict[str, Any]], source_count: int, target_count: int) -> Dict[str, Any]:
    """計算比對統計結果"""
    # 統計已匹配的段落
    matched_source_indices = set()
    matched_target_indices = set()
    total_similarity = 0.0
    match_count = 0
    
    if 'word_indices' in matches[0]:  # PDF優先模式
        for match in matches:
            if match['word_indices']:
                matched_source_indices.add(match['pdf_index'])
                for word_idx, similarity in zip(match['word_indices'], match['similarities']):
                    matched_target_indices.add(word_idx)
                    total_similarity += similarity
                    match_count += 1
    else:  # Word優先模式
        for match in matches:
            if match['pdf_indices']:
                matched_source_indices.add(match['word_index'])
                for pdf_idx, similarity in zip(match['pdf_indices'], match['similarities']):
                    matched_target_indices.add(pdf_idx)
                    total_similarity += similarity
                    match_count += 1
    
    # 計算未匹配的數量
    unmatched_source = source_count - len(matched_source_indices)
    unmatched_target = target_count - len(matched_target_indices)
    
    # 計算平均相似度
    avg_similarity = total_similarity / match_count if match_count > 0 else 0.0
    
    # 如果是PDF優先模式
    if 'word_indices' in matches[0]:
        return {
            'matched_count': match_count,
            'unmatched_pdf_count': unmatched_source,
            'unmatched_word_count': unmatched_target,
            'avg_similarity': avg_similarity
        }
    # 如果是Word優先模式
    else:
        return {
            'matched_count': match_count,
            'unmatched_word_count': unmatched_source,
            'unmatched_pdf_count': unmatched_target,
            'avg_similarity': avg_similarity
        }

"""比對結果顯示函數（增加目錄項目的顯示支持）"""
def display_match_results(comparison_results: Dict, 
                         word_paragraphs: List[Dict], 
                         pdf_paragraphs: List[Dict],
                         st = None):
    """
    在Streamlit界面中顯示比對結果，支持目錄項目的顯示
    
    Args:
        comparison_results: 從compare_documents返回的比對結果
        word_paragraphs: 原始Word段落列表
        pdf_paragraphs: 原始PDF段落列表
        st: Streamlit物件，默認使用全局st模塊
    """
    st = st or st_module
    
    # 顯示統計結果
    stats = comparison_results.get('statistics', {})
    st.write('### 比對統計')
    col1, col2, col3 = st.columns(3)
    col1.metric('Word文件段落數', stats.get('total_word', 0))
    col2.metric('PDF文件段落數', stats.get('total_pdf', 0))
    col3.metric('比對成功數', stats.get('matched', 0))
    
    col1, col2 = st.columns(2)
    col1.metric('Word未比對段落數', stats.get('unmatched_word', 0))
    col2.metric('PDF未比對段落數', stats.get('unmatched_pdf', 0))
    
    # 計算比對成功率
    word_match_rate = stats.get('match_rate_word', 0) * 100
    pdf_match_rate = stats.get('match_rate_pdf', 0) * 100
    st.progress(min(word_match_rate / 100, 1.0), f"Word匹配率: {word_match_rate:.1f}%")
    st.progress(min(pdf_match_rate / 100, 1.0), f"PDF匹配率: {pdf_match_rate:.1f}%")
    
    # 顯示成功匹配結果
    matches = comparison_results.get('matches', [])
    if matches:
        st.write('### 比對成功結果')
        for idx, match in enumerate(matches):
            with st.expander(f"匹配 {idx+1} (相似度: {match['similarity']:.2f}, 類型: {match.get('match_type', 'unknown')})"):
                col1, col2 = st.columns(2)
                
                # 處理Word內容顯示
                word_indices = match.get('word_indices', [])
                word_text = match.get('word_text', '')
                word_types = match.get('word_paragraph_types', [])
                
                # 特殊顯示目錄項目
                is_directory = 'directory' in word_types or match.get('word_paragraph_type') == 'directory'
                col1.markdown(f"**Word內容{' (目錄項目)' if is_directory else ''}:**")
                
                if is_directory:
                    col1.markdown(f"<div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px;'>{word_text}</div>", unsafe_allow_html=True)
                else:
                    col1.text_area("", word_text, height=150, key=f"word_text_{idx}")
                col1.markdown(f"Word索引: {word_indices}")
                
                # 處理PDF內容顯示
                pdf_text = match.get('pdf_text', '')
                pdf_page = match.get('pdf_page', '')
                
                # 處理頁碼為None的情況
                if pdf_page is None:
                    pdf_page = 'N/A'
                    
                is_pdf_directory = match.get('pdf_paragraph_type') == 'directory'
                
                col2.markdown(f"**PDF內容{' (目錄項目)' if is_pdf_directory else ''}:**")
                if is_pdf_directory:
                    col2.markdown(f"<div style='background-color: #f8f4e8; padding: 10px; border-radius: 5px;'>{pdf_text}</div>", unsafe_allow_html=True)
                else:
                    col2.text_area("", pdf_text, height=150, key=f"pdf_text_{idx}")
                col2.markdown(f"PDF頁碼: {pdf_page}")
                
                # 顯示差異
                st.markdown("**差異對比:**")
                st.markdown(f"""
                <style>
                .diff-removed {{background-color: #ffcccc; text-decoration: line-through;}}
                .diff-added {{background-color: #ccffcc;}}
                .diff-content {{background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace;}}
                </style>
                <div class='diff-content'>{match.get('diff_html', '')}</div>
                """, unsafe_allow_html=True)
    
    # 顯示未匹配的 Word 段落
    unmatched_word = comparison_results.get('unmatched_word', [])
    if unmatched_word:
        st.write('### 未匹配的 Word 段落')
        for p in unmatched_word:
            text = p.get('content', '')
            index = p.get('index', -1)
            paragraph_type = p.get('paragraph_type', 'normal')
            
            with st.expander(f"Word段落 {index}"):
                if paragraph_type == 'directory':
                    st.markdown(f"<div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px;'><strong>目錄項目:</strong> {text}</div>", unsafe_allow_html=True)
                else:
                    st.text_area("", text, height=100, key=f"unmatched_word_{index}")
    
    # 顯示未匹配的 PDF 段落
    unmatched_pdf = comparison_results.get('unmatched_pdf', [])
    if unmatched_pdf:
        st.write('### 未匹配的 PDF 段落')
        for p in unmatched_pdf:
            text = p.get('content', '')
            index = p.get('index', -1)
            page = p.get('page', -1)
            
            # 處理頁碼為None的情況
            if page is None:
                page = 'N/A'
                
            paragraph_type = p.get('paragraph_type', 'normal')
            
            with st.expander(f"PDF段落 {index} (頁碼: {page})"):
                if paragraph_type == 'directory':
                    st.markdown(f"<div style='background-color: #f8f4e8; padding: 10px; border-radius: 5px;'><strong>目錄項目:</strong> {text}</div>", unsafe_allow_html=True)
                else:
                    st.text_area("", text, height=100, key=f"unmatched_pdf_{index}")
