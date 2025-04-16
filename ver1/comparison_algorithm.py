import difflib
import re
import numpy as np
from collections import defaultdict

def exact_matching(text1, text2, ignore_options=None):
    """精確比對兩段文本的相似度"""
    if ignore_options is None:
        ignore_options = {}
    
    # 預處理文本
    processed_text1 = preprocess_text(text1, ignore_options)
    processed_text2 = preprocess_text(text2, ignore_options)
    
    # 計算相似度
    if not processed_text1 or not processed_text2:
        return 0.0
    
    matcher = difflib.SequenceMatcher(None, processed_text1, processed_text2)
    similarity = matcher.ratio()
    
    return similarity

def semantic_matching(text1, text2, ai=None):
    """使用AI進行語義比對"""
    if ai is None:
        return 0.5, "未提供AI引擎"
    
    similarity, error = ai.semantic_comparison(text1, text2)
    
    if error:
        return 0.5, error
    
    return similarity, None

def hybrid_matching(text1, text2, ignore_options=None, ai=None):
    """混合比對，結合精確比對和語義比對"""
    exact_sim = exact_matching(text1, text2, ignore_options)
    
    # 如果精確比對相似度較高，直接返回
    if exact_sim > 0.8:
        return exact_sim, None, "exact"
    
    # 如果提供了AI引擎，則進行語義比對
    if ai is not None:
        semantic_sim, error = semantic_matching(text1, text2, ai)
        
        if error:
            return exact_sim, error, "exact"
        
        # 綜合考慮兩種相似度
        combined_sim = max(exact_sim, semantic_sim)
        
        return combined_sim, None, "hybrid"
    
    return exact_sim, None, "exact"

def preprocess_text(text, ignore_options):
    """根據忽略選項預處理文本"""
    if not text:
        return ""
    
    result = text
    
    if ignore_options.get("ignore_space", False):
        result = re.sub(r'\s+', '', result)
    
    if ignore_options.get("ignore_punctuation", False):
        result = re.sub(r'[^\w\s]', '', result)
    
    if ignore_options.get("ignore_case", False):
        result = result.lower()
    
    if ignore_options.get("ignore_newline", False):
        result = result.replace('\n', ' ')
        result = re.sub(r'\s+', ' ', result)
    
    return result

def find_matching_paragraph(original_text, pdf_paragraphs, ignore_options, comparison_mode, similarity_threshold, ai=None):
    """找到最匹配的段落"""
    best_match = None
    best_similarity = 0.0
    best_diff = None
    semantic_similarity = None
    
    # 按頁碼分組候選段落
    page_groups = defaultdict(list)
    for para in pdf_paragraphs:
        page = para.get("page", "未知")
        page_groups[page].append(para)
    
    # 在每個頁面內找最佳匹配
    page_best_matches = []
    
    for page, page_paras in page_groups.items():
        page_best_match = None
        page_best_similarity = 0.0
        page_best_diff = None
        page_semantic_similarity = None
        
        for para in page_paras:
            pdf_text = para.get("content", "")
            
            if comparison_mode == "精確比對":
                similarity = exact_matching(original_text, pdf_text, ignore_options)
                if similarity > page_best_similarity:
                    page_best_similarity = similarity
                    page_best_match = para
                    
                    # 生成差異
                    matcher = difflib.SequenceMatcher(None, original_text, pdf_text)
                    page_best_diff = list(matcher.get_opcodes())
            
            elif comparison_mode == "語意比對":
                if ai is not None:
                    similarity, error = semantic_matching(original_text, pdf_text, ai)
                    if error:
                        continue
                    
                    if similarity > page_best_similarity:
                        page_best_similarity = similarity
                        page_best_match = para
                        page_semantic_similarity = similarity
                else:
                    # 如果沒有提供AI引擎，則使用精確比對
                    similarity = exact_matching(original_text, pdf_text, ignore_options)
                    if similarity > page_best_similarity:
                        page_best_similarity = similarity
                        page_best_match = para
                        
                        # 生成差異
                        matcher = difflib.SequenceMatcher(None, original_text, pdf_text)
                        page_best_diff = list(matcher.get_opcodes())
            
            else:  # 混合比對
                similarity, error, match_type = hybrid_matching(original_text, pdf_text, ignore_options, ai)
                if similarity > page_best_similarity:
                    page_best_similarity = similarity
                    page_best_match = para
                    
                    if match_type == "hybrid" or match_type == "semantic":
                        page_semantic_similarity = similarity
                    
                    # 生成差異
                    matcher = difflib.SequenceMatcher(None, original_text, pdf_text)
                    page_best_diff = list(matcher.get_opcodes())
        
        if page_best_match and page_best_similarity >= similarity_threshold:
            page_best_matches.append({
                "match": page_best_match,
                "similarity": page_best_similarity,
                "diff": page_best_diff,
                "semantic_similarity": page_semantic_similarity
            })
    
    # 從所有頁面的最佳匹配中選擇全局最佳匹配
    if page_best_matches:
        best_page_match = max(page_best_matches, key=lambda x: x["similarity"])
        best_match = best_page_match["match"]
        best_similarity = best_page_match["similarity"]
        best_diff = best_page_match["diff"]
        semantic_similarity = best_page_match["semantic_similarity"]
    
    return best_match, best_similarity, best_diff, semantic_similarity

def compare_documents(word_data, pdf_data, ignore_options, comparison_mode, similarity_threshold, ai=None):
    """比對兩個文檔的內容"""
    results = {
        "paragraph_results": [],
        "table_results": [],
        "statistics": {
            "total_paragraphs": len(word_data["paragraphs"]),
            "similar_paragraphs": 0,
            "different_paragraphs": 0,
            "total_tables": len(word_data["tables"]),
            "similar_tables": 0,
            "different_tables": 0
        }
    }
    
    # 比對段落
    for i, para in enumerate(word_data["paragraphs"]):
        original_text = para["content"]
        
        # 找到最匹配的段落
        best_match, similarity, diff, semantic_similarity = find_matching_paragraph(
            original_text,
            pdf_data["paragraphs"],
            ignore_options,
            comparison_mode,
            similarity_threshold,
            ai
        )
        
        # 判斷是否相似
        is_similar = similarity >= similarity_threshold
        
        # 更新統計信息
        if is_similar:
            results["statistics"]["similar_paragraphs"] += 1
        else:
            results["statistics"]["different_paragraphs"] += 1
        
        # 添加結果
        result = {
            "original_index": i,
            "original_text": original_text,
            "matched_text": best_match["content"] if best_match else None,
            "matched_page": best_match["page"] if best_match else "未找到",
            "exact_similarity": similarity,
            "semantic_similarity": semantic_similarity,
            "is_similar": is_similar,
            "diff": diff
        }
        
        results["paragraph_results"].append(result)
    
    # 比對表格（暫時不實現）
    
    return results

def generate_comparison_report(comparison_results, diff_display_mode, show_all_content=False):
    """生成比對報告"""
    report = {
        "summary": {
            "total_paragraphs": comparison_results["statistics"]["total_paragraphs"],
            "similar_paragraphs": comparison_results["statistics"]["similar_paragraphs"],
            "different_paragraphs": comparison_results["statistics"]["different_paragraphs"],
            "similarity_percentage": comparison_results["statistics"]["similar_paragraphs"] / comparison_results["statistics"]["total_paragraphs"] * 100 if comparison_results["statistics"]["total_paragraphs"] > 0 else 0
        },
        "paragraph_details": []
    }
    
    # 添加段落詳情
    for result in comparison_results["paragraph_results"]:
        if show_all_content or not result["is_similar"]:
            detail = {
                "original_index": result["original_index"],
                "original_text": result["original_text"],
                "matched_text": result["matched_text"],
                "matched_page": result["matched_page"],
                "exact_similarity": result["exact_similarity"],
                "semantic_similarity": result["semantic_similarity"],
                "is_similar": result["is_similar"],
                "diff_html": format_diff_html(result["diff"], diff_display_mode) if result["diff"] else None
            }
            
            report["paragraph_details"].append(detail)
    
    return report

def format_diff_html(diff, mode):
    """格式化差異為HTML"""
    if not diff:
        return ""
    
    if mode == "字符級別":
        return format_char_level_diff_html(diff)
    elif mode == "詞語級別":
        return format_word_level_diff_html(diff)
    else:  # 行級別
        return format_line_level_diff_html(diff)

def format_char_level_diff_html(diff):
    """格式化字符級別的差異為HTML"""
    html = ""
    
    for tag, i1, i2, j1, j2 in diff:
        if tag == 'replace':
            html += f'<span class="diff-char-removed">{diff[0][1][i1:i2]}</span>'
            html += f'<span class="diff-char-added">{diff[0][2][j1:j2]}</span>'
        elif tag == 'delete':
            html += f'<span class="diff-char-removed">{diff[0][1][i1:i2]}</span>'
        elif tag == 'insert':
            html += f'<span class="diff-char-added">{diff[0][2][j1:j2]}</span>'
        elif tag == 'equal':
            html += diff[0][1][i1:i2]
    
    return html

def format_word_level_diff_html(diff):
    """格式化詞語級別的差異為HTML"""
    # 將字符級別的差異轉換為詞語級別
    text1 = diff[0][1]
    text2 = diff[0][2]
    
    words1 = re.findall(r'\w+|\W+', text1)
    words2 = re.findall(r'\w+|\W+', text2)
    
    matcher = difflib.SequenceMatcher(None, words1, words2)
    word_diff = list(matcher.get_opcodes())
    
    html = ""
    
    for tag, i1, i2, j1, j2 in word_diff:
        if tag == 'replace':
            html += f'<span class="diff-removed">{"".join(words1[i1:i2])}</span>'
            html += f'<span class="diff-added">{"".join(words2[j1:j2])}</span>'
        elif tag == 'delete':
            html += f'<span class="diff-removed">{"".join(words1[i1:i2])}</span>'
        elif tag == 'insert':
            html += f'<span class="diff-added">{"".join(words2[j1:j2])}</span>'
        elif tag == 'equal':
            html += "".join(words1[i1:i2])
    
    return html

def format_line_level_diff_html(diff):
    """格式化行級別的差異為HTML"""
    # 將字符級別的差異轉換為行級別
    text1 = diff[0][1]
    text2 = diff[0][2]
    
    lines1 = text1.split('\n')
    lines2 = text2.split('\n')
    
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    line_diff = list(matcher.get_opcodes())
    
    html = ""
    
    for tag, i1, i2, j1, j2 in line_diff:
        if tag == 'replace':
            html += f'<div class="diff-removed">{"<br>".join(lines1[i1:i2])}</div>'
            html += f'<div class="diff-added">{"<br>".join(lines2[j1:j2])}</div>'
        elif tag == 'delete':
            html += f'<div class="diff-removed">{"<br>".join(lines1[i1:i2])}</div>'
        elif tag == 'insert':
            html += f'<div class="diff-added">{"<br>".join(lines2[j1:j2])}</div>'
        elif tag == 'equal':
            html += f'<div>{"<br>".join(lines1[i1:i2])}</div>'
    
    return html
