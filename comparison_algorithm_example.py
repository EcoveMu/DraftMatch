import difflib
import re
import html
import numpy as np
from sentence_transformers import SentenceTransformer

def exact_matching(text1, text2, ignore_space=True, ignore_punctuation=True, ignore_case=True):
    if ignore_space:
        text1 = re.sub(r'\s+', ' ', text1)
        text2 = re.sub(r'\s+', ' ', text2)
    if ignore_punctuation:
        text1 = re.sub(r'[.,;:!?，。；：！？]', '', text1)
        text2 = re.sub(r'[.,;:!?，。；：！？]', '', text2)
    if ignore_case:
        text1 = text1.lower()
        text2 = text2.lower()
    matcher = difflib.SequenceMatcher(None, text1, text2)
    similarity = matcher.ratio()
    diff = list(difflib.ndiff(text1.splitlines(), text2.splitlines()))
    return similarity, diff

def semantic_matching(text1, text2):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

def hybrid_matching(text1, text2, exact_threshold=0.8, semantic_threshold=0.8):
    exact_similarity, diff = exact_matching(text1, text2)
    if exact_similarity >= exact_threshold:
        return True, exact_similarity, None, diff
    semantic_similarity = semantic_matching(text1, text2)
    is_similar = semantic_similarity >= semantic_threshold
    return is_similar, exact_similarity, semantic_similarity, diff

def compare_documents(doc1, doc2, ignore_options=None, comparison_mode='hybrid', similarity_threshold=0.6, ai_instance=None):
    if ignore_options is None:
        ignore_options = {}

    results = []
    for i, para1 in enumerate(doc1):
        best_match = None
        best_similarity = 0
        best_index = -1

        for j, para2 in enumerate(doc2):
            if comparison_mode == 'exact':
                sim, _ = exact_matching(
                    para1['content'], para2['content'],
                    ignore_space=ignore_options.get("ignore_whitespace", True),
                    ignore_punctuation=ignore_options.get("ignore_punctuation", True),
                    ignore_case=ignore_options.get("ignore_case", True),
                )
            elif comparison_mode == 'semantic':
                sim = semantic_matching(para1['content'], para2['content'])
            elif comparison_mode == 'hybrid':
                is_similar, exact_sim, semantic_sim, _ = hybrid_matching(para1['content'], para2['content'])
                sim = max(exact_sim, semantic_sim) if semantic_sim else exact_sim
            elif comparison_mode == 'ai' and ai_instance:
                ai_matches = ai_instance.match_paragraphs(doc1, doc2)
                for m in ai_matches:
                    results.append({
                        "doc1_index": m["doc1_index"],
                        "doc2_index": m["doc2_index"],
                        "similarity": m["similarity"],
                        "doc1_text": doc1[m["doc1_index"]]["content"],
                        "doc2_text": doc2[m["doc2_index"]]["content"],
                    })
                return results
            else:
                sim = 0

            if sim > best_similarity:
                best_similarity = sim
                best_match = para2
                best_index = j

        if best_similarity >= similarity_threshold:
            results.append({
                "doc1_index": i,
                "doc2_index": best_index,
                "similarity": best_similarity,
                "doc1_text": para1["content"],
                "doc2_text": best_match["content"]
            })

    return results
def format_diff_html(a: str, b: str) -> str:
    """
    把兩段文字做字元級 diff，回傳帶 <span> 標記的 HTML。
    app.py 會用 .diff-removed / .diff-added 這兩個 class 去上色。
    """
    pieces = []
    for d in difflib.ndiff(a, b):
        tag, ch = d[0], html.escape(d[2:])
        if tag == "-":
            pieces.append(f'<span class="diff-removed">{ch}</span>')
        elif tag == "+":
            pieces.append(f'<span class="diff-added">{ch}</span>')
        else:     # ' ' 代表相同
            pieces.append(ch)
    return "".join(pieces)

def generate_comparison_report(results,
                               diff_display_mode="字符級別",
                               show_all_content=False,
                               similarity_cutoff: float = 0.8):
    """
    把 compare_documents() 的結果轉成 app.py 期望的 dict 結構：
    {
        "summary": {...},
        "paragraph_details": [...],
        # 這裡暫不處理表格，需要時再擴充
    }
    """
    total = len(results)
    similar = sum(1 for r in results if r["similarity"] >= similarity_cutoff)
    different = total - similar

    summary = {
        "total_paragraphs": total,
        "similar_paragraphs": similar,
        "different_paragraphs": different,
        "total_tables": 0,
        "similar_tables": 0,
        "different_tables": 0,
        "paragraph_similarity_percentage": (similar / total * 100) if total else 0,
        "table_similarity_percentage": 0,
    }

    paragraph_details = []
    for r in results:
        diff_html = format_diff_html(r["doc1_text"], r["doc2_text"])
        paragraph_details.append({
            "original_index": r["doc1_index"],
            "matched_page": r["doc2_index"],        # 目前只放索引，若要頁碼請自行再對應
            "matched_text": r["doc2_text"],
            "original_text": r["doc1_text"],
            "exact_similarity": r["similarity"],
            "is_similar": r["similarity"] >= similarity_cutoff,
            "diff_html": diff_html
        })

    report = {
        "summary": summary,
        "paragraph_details": paragraph_details,
        # 若未做表格比對，可不放 "table_details"
    }
    return report