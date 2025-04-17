
"""
comparison_algorithm.py  (rev‑2025‑04‑18)

核心比對演算法：
1.  exact_matching        ‒ difflib (字面)
2.  semantic_matching     ‒ sentence‑transformers (若可用)
3.  hybrid_matching       ‒ 先 exact，否則 semantic
4.  compare_pdf_first     ‒ 以 PDF 為主體逐段/逐頁比對 Word
-----------------------------------------------------------------
回傳：
    {
        "matches": [ {...}, ... ],
        "unmatched_pdf": [ {...}, ... ],
        "unmatched_word": [ {...}, ... ],
        "statistics": {...}
    }
-----------------------------------------------------------------
此版本與舊 compare_documents 向下相容，原函式仍保留
"""
from __future__ import annotations
import difflib, re, numpy as np, unicodedata, html

# ---------- 相似度基礎 ----------
try:
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    _SENTENCE_MODEL = True
except Exception:
    _st_model = None
    _SENTENCE_MODEL = False


def _normalize(text:str)->str:
    """全形→半形、簡繁統一、去多空白"""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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


def hybrid_matching(t1:str, t2:str,
                    exact_thresh = 0.85)->float:
    exact = exact_matching(t1, t2, True, True, True)
    return exact if exact >= exact_thresh else semantic_matching(t1, t2)


# ---------- Diff Helper ----------
def _diff_html(a:str, b:str)->str:
    d = difflib.ndiff(a, b)
    buff=[]
    for s in d:
        if s.startswith('  '):
            buff.append(html.escape(s[2:]))
        elif s.startswith('- '):
            buff.append(f'<span class="diff-removed">{html.escape(s[2:])}</span>')
        elif s.startswith('+ '):
            buff.append(f'<span class="diff-added">{html.escape(s[2:])}</span>')
    return ''.join(buff)


# ---------- PDF‑first 比對 ----------
def compare_pdf_first(
        word_data:dict, pdf_data:dict,
        comparison_mode:str='hybrid',
        similarity_threshold:float=0.6,
        ignore_options:dict|None=None,
        ai_instance=None)->dict:
    """
    以 pdf_data['paragraphs'] 為主迴圈，比對 word_data['paragraphs'].
    每個段落包含: index, content, page
    """
    if ignore_options is None:
        ignore_options = dict(ignore_space=True, ignore_punctuation=True,
                              ignore_case=True, ignore_newline=True)

    def preprocess(t:str)->str:
        t=_normalize(t)
        if ignore_options.get("ignore_newline", True):
            t = t.replace('\n', ' ')
        if ignore_options.get("ignore_space", True):
            t = re.sub(r'\s+', ' ', t)
        if ignore_options.get("ignore_case", True):
            t = t.lower()
        if ignore_options.get("ignore_punctuation", True):
            t = re.sub(r'[^\w\s]', '', t)
        return t.strip()

    # 預先處理文本
    for p in pdf_data['paragraphs']:
        p['processed']=preprocess(p['content'])
    for p in word_data['paragraphs']:
        p['processed']=preprocess(p['content'])

    matches=[]
    used_word=set()

    for pdf_para in pdf_data['paragraphs']:
        best=None; best_sim=0.0; best_idx=-1
        for w in word_data['paragraphs']:
            sim=0.0
            if comparison_mode=='exact':
                sim = exact_matching(pdf_para['processed'], w['processed'], False, False, False)
            elif comparison_mode=='semantic':
                sim = semantic_matching(pdf_para['processed'], w['processed'])
            elif comparison_mode=='hybrid':
                sim = hybrid_matching(pdf_para['processed'], w['processed'])
            elif comparison_mode=='ai' and ai_instance:
                sim,_ = ai_instance.semantic_comparison(pdf_para['content'], w['content'])
            if sim>best_sim:
                best_sim=sim; best=w; best_idx=w['index']
        if best and best_sim>=similarity_threshold:
            matches.append(dict(
                pdf_index=pdf_para['index'],
                pdf_page=pdf_para.get('page'),
                pdf_text=pdf_para['content'],
                word_index=best_idx,
                word_text=best['content'],
                similarity=best_sim,
                diff_html=_diff_html(best['content'], pdf_para['content'])
            ))
            used_word.add(best_idx)

    unmatched_pdf=[p for p in pdf_data['paragraphs']
                   if p['index'] not in {m['pdf_index'] for m in matches}]
    unmatched_word=[p for p in word_data['paragraphs']
                    if p['index'] not in used_word]

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


# ---------- 舊介面保持 ----------
def compare_documents(doc1, doc2, *args, **kwargs):
    """Deprecated wrapper – 保留給舊程式使用 (Word 為主)"""
    return compare_pdf_first(
        word_data=doc1, pdf_data=doc2, *args, **kwargs)
