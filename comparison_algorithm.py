"""comparison_algorithm.py  (rev‑2025‑04‑18b)

優化文字比對演算法，加入更多智能比對功能：
1. 改進段落拆分邏輯
2. 加入上下文感知比對
3. 優化相似度計算
4. 加入差異摘要功能
"""
from __future__ import annotations
import difflib, re, numpy as np, unicodedata, html
from typing import List, Dict, Any, Tuple

try:
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    _SENTENCE_MODEL = True
except Exception:
    _st_model = None
    _SENTENCE_MODEL = False

def _normalize(text:str)->str:
    """文本標準化處理"""
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
    """將文本分割成句子"""
    # 使用正則表達式分割句子
    sentences = re.split(r'(?<=[。！？!?])', text)
    return [s.strip() for s in sentences if s.strip()]

def exact_matching(t1:str, t2:str,
                   ignore_space=True, ignore_punctuation=True, ignore_case=True)->float:
    """精確比對"""
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
    """語意比對"""
    if _SENTENCE_MODEL:
        emb1, emb2 = _st_model.encode([t1, t2])
        return float(np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2)))
    return exact_matching(t1, t2, False, False, False)

def hybrid_matching(t1:str, t2:str, exact_thresh=0.85)->float:
    """混合比對"""
    exact = exact_matching(t1, t2, True, True, True)
    return exact if exact >= exact_thresh else semantic_matching(t1, t2)

def substring_similarity(a:str, b:str)->float:
    """子字串相似度"""
    if not a or not b: return 0.0
    if a in b:
        return len(a)/len(b)
    if b in a:
        return len(b)/len(a)
    return 0.0

def _diff_html(a:str, b:str)->str:
    """生成差異 HTML"""
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

def compare_pdf_first(word_data: dict, pdf_data: dict,
                     comparison_mode: str = 'hybrid',
                     similarity_threshold: float = 0.6,
                     ignore_options: dict = None,
                     ai_instance = None) -> dict:
    """改進的比對算法，支援多段對一頁比對"""
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

    # 預處理所有段落
    for p in pdf_data['paragraphs']:
        if isinstance(p, dict) and 'content' in p:
            p['processed'] = _preprocess(p['content'], ignore_options)
            p['sentences'] = split_into_sentences(p['content'])
    
    for p in word_data['paragraphs']:
        if isinstance(p, dict) and 'content' in p:
            p['processed'] = _preprocess(p['content'], ignore_options)
            p['sentences'] = split_into_sentences(p['content'])

    # 生成 Word 段落的各種組合
    word_combinations = merge_word_paragraphs(word_data['paragraphs'], 
                                           ignore_options=ignore_options)
    
    matches = []
    used_word_indices = set()
    
    for pdf_para in pdf_data['paragraphs']:
        best_match = None
        best_sim = 0.0
        
        for word_combo in word_combinations:
            # 跳過已使用的段落組合
            if any(idx in used_word_indices for idx in word_combo['indices']):
                continue
                
            # 計算相似度
            if comparison_mode == 'hybrid':
                sim = hybrid_matching(pdf_para['processed'], word_combo['processed'])
            elif comparison_mode == 'semantic':
                sim = semantic_matching(pdf_para['processed'], word_combo['processed'])
            elif comparison_mode == 'ai' and ai_instance:
                sim, _ = ai_instance.semantic_comparison(pdf_para['content'], word_combo['content'])
            else:  # exact
                sim = exact_matching(pdf_para['processed'], word_combo['processed'])
                
            if sim > best_sim and sim >= similarity_threshold:
                best_sim = sim
                best_match = word_combo
        
        if best_match:
            # 生成差異摘要
            diff_summary = []
            for pdf_sent in pdf_para['sentences']:
                best_match_sent = None
                best_sent_sim = 0.0
                
                for word_sent in best_match['sentences']:
                    sent_sim = hybrid_matching(pdf_sent, word_sent)
                    if sent_sim > best_sent_sim:
                        best_sent_sim = sent_sim
                        best_match_sent = word_sent
                
                if best_match_sent and best_sent_sim < 0.9:
                    diff_summary.append({
                        'pdf_sentence': pdf_sent,
                        'word_sentence': best_match_sent,
                        'similarity': best_sent_sim
                    })
            
            matches.append({
                'pdf_index': pdf_para['index'],
                'pdf_page': pdf_para.get('page'),
                'pdf_text': pdf_para['content'],
                'word_indices': best_match['indices'],
                'word_text': best_match['content'],
                'similarity': best_sim,
                'diff_html': _diff_html(best_match['content'], pdf_para['content']),
                'diff_summary': diff_summary
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
        match_rate_word = len(matches)/len(word_data['paragraphs']) if word_data['paragraphs'] else 0
    )
    return dict(matches=matches,
                unmatched_pdf=unmatched_pdf,
                unmatched_word=unmatched_word,
                statistics=stats)

def compare_documents(doc1, doc2, *args, **kwargs):
    return compare_pdf_first(word_data=doc1, pdf_data=doc2, *args, **kwargs)
