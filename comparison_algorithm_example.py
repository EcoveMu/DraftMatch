import difflib
import re
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