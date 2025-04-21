import re
import difflib
import numpy as np
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

try:
    import torch
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    _SENTENCE_MODEL = True
except (ImportError, RuntimeError):
    _st_model = None
    _SENTENCE_MODEL = False

def split_into_sentences(text: str) -> list:
    """將文本拆分為句子列表。"""
    # 中英文標點符號的斷句模式
    pattern = r'(?<=[。．！!？?]|\.")\s*'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def _preprocess(text: str, ignore_options: Dict[str, bool]) -> str:
    """預處理文本，根據忽略選項進行調整。"""
    if ignore_options.get("ignore_space", True):
        text = re.sub(r'\s+', '', text)
    if ignore_options.get("ignore_punctuation", True):
        text = re.sub(r'[^\w\s]', '', text)
    if ignore_options.get("ignore_case", True):
        text = text.lower()
    if ignore_options.get("ignore_newline", True):
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
    return text.strip()

def substring_similarity(text1: str, text2: str) -> float:
    """計算子字串相似度。"""
    if not text1 or not text2:
        return 0.0
    min_len = min(len(text1), len(text2))
    max_len = max(len(text1), len(text2))
    return min_len / max_len if min_len > 0 else 0.0

def prepare_sentences(pdf_data: Dict[str, Any], word_data: Dict[str, Any], 
                     ignore_options: Dict[str, bool]) -> tuple:
    """準備 PDF 和 Word 的句子列表。"""
    pdf_sentences = []
    for para in pdf_data["paragraphs"]:
        page = para["page"]
        for sent in split_into_sentences(para["content"]):
            pdf_sentences.append({
                "page": page,
                "content": sent,
                "index": len(pdf_sentences),
                "processed": _preprocess(sent, ignore_options)
            })

    word_sentences = []
    for para in word_data["paragraphs"]:
        para_index = para["index"]
        para_page = para.get("page")
        for sent in split_into_sentences(para["content"]):
            word_sentences.append({
                "paragraph_index": para_index,
                "page": para_page,
                "content": sent,
                "processed": _preprocess(sent, ignore_options)
            })
            
    return pdf_sentences, word_sentences

def semantic_sentence_matching(text1: str, text2: str) -> float:
    """使用向量模型計算語意相似度"""
    if _SENTENCE_MODEL:
        embeddings = _st_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    return SequenceMatcher(None, text1, text2).ratio()

def compare_sentences(word_sents: List[Dict[str, Any]], 
                     pdf_sents: List[Dict[str, Any]], 
                     comparison_mode: str = 'semantic',
                     similarity_threshold: float = 0.6,
                     ignore_options: Optional[Dict[str, bool]] = None,
                     ai_instance: Any = None,
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """句子級別的比對，支援進度回調。"""
    if ignore_options is None:
        ignore_options = {
            "ignore_space": True,
            "ignore_punctuation": True,
            "ignore_case": True,
            "ignore_newline": True
        }

    matches = []
    used_word_idx = set()
    
    total = len(pdf_sents)
    for i, pdf_sent in enumerate(pdf_sents):
        if progress_callback:
            progress_callback(i / total)
            
        best_match = None
        best_sim = 0.0
        best_word_indices = []
        
        for ws in word_sents:
            if ws['paragraph_index'] in used_word_idx:
                continue
                
            # 計算相似度
            if comparison_mode == 'hybrid':
                sim = max(
                    SequenceMatcher(None, 
                                  pdf_sent['processed'],
                                  ws['processed']).ratio(),
                    substring_similarity(pdf_sent['processed'],
                                      ws['processed'])
                )
            elif comparison_mode == 'semantic':
                sim = semantic_sentence_matching(
                    pdf_sent['processed'],
                    ws['processed']
                )
            elif comparison_mode == 'ai' and ai_instance:
                sim, _ = ai_instance.semantic_comparison(
                    pdf_sent['content'],
                    ws['content']
                )
            else:  # exact
                sim = 1.0 if pdf_sent['processed'] == ws['processed'] else 0.0
                
            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                best_match = ws
                best_word_indices = [ws['paragraph_index']]
                
        if best_match:
            used_word_idx.update(best_word_indices)
            matches.append({
                "pdf_page": pdf_sent["page"],
                "pdf_text": pdf_sent["content"],
                "word_text": best_match["content"],
                "word_indices": best_word_indices,
                "word_page": best_match.get("page"),
                "similarity": best_sim,
                "diff_html": create_diff_html(best_match["content"],
                                            pdf_sent["content"])
            })
    
    if progress_callback:
        progress_callback(1.0)
    
    # 收集未匹配句子
    unmatched_pdf = [
        s for s in pdf_sents 
        if s["content"] and s["page"] not in {m["pdf_page"] for m in matches}
    ]
    
    return {
        "matches": matches,
        "unmatched_pdf": unmatched_pdf,
        "statistics": {
            "matched": len(matches),
            "total_pdf": len(pdf_sents),
            "total_word": len(word_sents)
        }
    }

def create_diff_html(text1: str, text2: str) -> str:
    """生成差異的 HTML 標記，支援中文字符。"""
    def split_text(text):
        return [c for c in text]
    
    differ = difflib.Differ()
    diff = list(differ.compare(split_text(text1), split_text(text2)))
    
    html_parts = []
    for char in diff:
        if char.startswith('+ '):
            html_parts.append(f'<span style="background-color:#e6ffe6">{char[2:]}</span>')
        elif char.startswith('- '):
            html_parts.append(f'<span style="background-color:#ffe6e6">{char[2:]}</span>')
        elif char.startswith('  '):
            html_parts.append(char[2:])
    
    return ''.join(html_parts)

def create_sentence_hash(sentence: str) -> str:
    """創建句子的簡單雜湊值用於快速比對"""
    words = set(sentence.lower().split())
    return f"{len(words)}:{sorted(words)[:3]}"

def quick_filter_candidates(pdf_sent: dict, 
                          word_sents: list, 
                          threshold: float = 0.3) -> list:
    """快速篩選可能的匹配候選"""
    pdf_hash = create_sentence_hash(pdf_sent['content'])
    pdf_len = len(pdf_sent['content'])
    
    candidates = []
    for ws in word_sents:
        word_hash = create_sentence_hash(ws['content'])
        word_len = len(ws['content'])
        
        # 長度比例檢查
        len_ratio = min(pdf_len, word_len) / max(pdf_len, word_len)
        if len_ratio < threshold:
            continue
            
        # 雜湊值比較
        if pdf_hash.split(':')[0] == word_hash.split(':')[0]:
            candidates.append(ws)
            
    return candidates