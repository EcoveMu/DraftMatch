
"""comparison_algorithm.py  (rev‑2025‑04‑18b)

在 rev‑04‑18 基礎上新增 **substring_similarity**：
若 Word 段落完整包含於 PDF 段落 (或反之)，視為高相似度，
以「短文本長度 / 長文本長度」作為分數，可解決「多段對一頁」情境。
其餘演算法 (exact / semantic / hybrid / ai) 保持不變。
"""
from __future__ import annotations
import difflib, re, numpy as np, unicodedata, html

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

def compare_pdf_first(word_data:dict, pdf_data:dict,
                      comparison_mode:str='hybrid',
                      similarity_threshold:float=0.6,
                      ignore_options:dict|None=None,
                      ai_instance=None)->dict:
    if ignore_options is None:
        ignore_options = dict(ignore_space=True, ignore_punctuation=True,
                              ignore_case=True, ignore_newline=True)

    def preprocess(t:str)->str:
        t=_normalize(t)
        if ignore_options.get('ignore_newline', True):
            t=t.replace('\n',' ')
        if ignore_options.get('ignore_space', True):
            t=re.sub(r'\s+',' ', t)
        if ignore_options.get('ignore_case', True):
            t=t.lower()
        if ignore_options.get('ignore_punctuation', True):
            t=re.sub(r'[^\w\s]','', t)
        return t.strip()

    for p in pdf_data['paragraphs']:
        p['processed']=preprocess(p['content'])
    for p in word_data['paragraphs']:
        p['processed']=preprocess(p['content'])

    matches=[]
    used_word=set()

    for pdf_para in pdf_data['paragraphs']:
        best=None; best_sim=0.0; best_idx=-1
        for w in word_data['paragraphs']:
            # --- substring quick win ---
            sub_sim = substring_similarity(pdf_para['processed'], w['processed'])
            if sub_sim>=similarity_threshold and sub_sim>best_sim:
                best_sim=sub_sim; best=w; best_idx=w['index']; continue

            if comparison_mode=='exact':
                sim = exact_matching(pdf_para['processed'], w['processed'], False, False, False)
            elif comparison_mode=='semantic':
                sim = semantic_matching(pdf_para['processed'], w['processed'])
            elif comparison_mode=='hybrid':
                sim = hybrid_matching(pdf_para['processed'], w['processed'])
            elif comparison_mode=='ai' and ai_instance:
                sim,_ = ai_instance.semantic_comparison(pdf_para['content'], w['content'])
            else:
                sim=0.0
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

def compare_documents(doc1, doc2, *args, **kwargs):
    return compare_pdf_first(word_data=doc1, pdf_data=doc2, *args, **kwargs)
