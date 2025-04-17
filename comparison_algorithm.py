import difflib
import re
import numpy as np
from collections import defaultdict
import pandas as pd

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

def compare_table_cells(cell1, cell2, ignore_options):
    """比對表格單元格內容"""
    # 預處理單元格文本
    processed_cell1 = preprocess_text(str(cell1), ignore_options)
    processed_cell2 = preprocess_text(str(cell2), ignore_options)
    
    # 計算相似度
    if not processed_cell1 or not processed_cell2:
        # 如果兩個單元格都為空，視為完全相同
        if not processed_cell1 and not processed_cell2:
            return 1.0, None
        # 如果只有一個單元格為空，視為完全不同
        return 0.0, None
    
    matcher = difflib.SequenceMatcher(None, processed_cell1, processed_cell2)
    similarity = matcher.ratio()
    
    # 生成差異
    diff = list(matcher.get_opcodes())
    
    return similarity, diff

def find_matching_table(original_table, pdf_tables, ignore_options, similarity_threshold):
    """找到最匹配的表格"""
    best_match = None
    best_similarity = 0.0
    best_diff_matrix = None
    best_match_index = -1
    
    # 將原始表格轉換為DataFrame
    original_df = pd.DataFrame(original_table["content"])
    
    for i, pdf_table in enumerate(pdf_tables):
        # 將PDF表格轉換為DataFrame
        pdf_df = pd.DataFrame(pdf_table["content"])
        
        # 如果行數或列數差異太大，跳過此表格
        if abs(len(original_df) - len(pdf_df)) > max(len(original_df), len(pdf_df)) * 0.5 or \
           abs(len(original_df.columns) - len(pdf_df.columns)) > max(len(original_df.columns), len(pdf_df.columns)) * 0.5:
            continue
        
        # 計算表格相似度
        table_similarity, diff_matrix = compare_tables(original_df, pdf_df, ignore_options)
        
        if table_similarity > best_similarity:
            best_similarity = table_similarity
            best_match = pdf_table
            best_diff_matrix = diff_matrix
            best_match_index = i
    
    # 判斷是否找到匹配的表格
    is_similar = best_similarity >= similarity_threshold
    
    return best_match, best_similarity, best_diff_matrix, best_match_index, is_similar

def compare_tables(df1, df2, ignore_options):
    """比對兩個表格的內容"""
    # 調整表格大小，使其具有相同的行數和列數
    max_rows = max(len(df1), len(df2))
    max_cols = max(len(df1.columns), len(df2.columns))
    
    # 擴展df1
    df1_extended = pd.DataFrame(index=range(max_rows), columns=range(max_cols))
    for i in range(min(len(df1), max_rows)):
        for j in range(min(len(df1.columns), max_cols)):
            df1_extended.iloc[i, j] = df1.iloc[i, j] if i < len(df1) and j < len(df1.columns) else ""
    
    # 擴展df2
    df2_extended = pd.DataFrame(index=range(max_rows), columns=range(max_cols))
    for i in range(min(len(df2), max_rows)):
        for j in range(min(len(df2.columns), max_cols)):
            df2_extended.iloc[i, j] = df2.iloc[i, j] if i < len(df2) and j < len(df2.columns) else ""
    
    # 計算單元格相似度矩陣和差異矩陣
    similarity_matrix = np.zeros((max_rows, max_cols))
    diff_matrix = np.empty((max_rows, max_cols), dtype=object)
    
    for i in range(max_rows):
        for j in range(max_cols):
            cell1 = df1_extended.iloc[i, j] if i < len(df1_extended) and j < len(df1_extended.columns) else ""
            cell2 = df2_extended.iloc[i, j] if i < len(df2_extended) and j < len(df2_extended.columns) else ""
            
            similarity, diff = compare_table_cells(cell1, cell2, ignore_options)
            similarity_matrix[i, j] = similarity
            diff_matrix[i, j] = diff
    
    # 計算整體表格相似度
    if max_rows * max_cols > 0:
        table_similarity = np.sum(similarity_matrix) / (max_rows * max_cols)
    else:
        table_similarity = 0.0
    
    return table_similarity, diff_matrix

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
    
    # 比對表格
    for i, table in enumerate(word_data["tables"]):
        # 找到最匹配的表格
        best_match, similarity, diff_matrix, best_match_index, is_similar = find_matching_table(
            table,
            pdf_data["tables"],
            ignore_options,
            similarity_threshold
        )
        
        # 更新統計信息
        if is_similar:
            results["statistics"]["similar_tables"] += 1
        else:
            results["statistics"]["different_tables"] += 1
        
        # 添加結果
        result = {
            "original_index": i,
            "matched_index": best_match_index if best_match else -1,
            "original_table": table["content"],
            "matched_table": best_match["content"] if best_match else None,
            "matched_page": best_match["page"] if best_match else "未找到",
            "similarity": similarity,
            "is_similar": is_similar,
            "diff_matrix": diff_matrix
        }
        
        results["table_results"].append(result)
    
    return results

def generate_comparison_report(comparison_results, diff_display_mode, show_all_content=False):
    """生成比對報告"""
    report = {
        "summary": {
            "total_paragraphs": comparison_results["statistics"]["total_paragraphs"],
            "similar_paragraphs": comparison_results["statistics"]["similar_paragraphs"],
            "different_paragraphs": comparison_results["statistics"]["different_paragraphs"],
            "total_tables": comparison_results["statistics"]["total_tables"],
            "similar_tables": comparison_results["statistics"]["similar_tables"],
            "different_tables": comparison_results["statistics"]["different_tables"],
            "paragraph_similarity_percentage": comparison_results["statistics"]["similar_paragraphs"] / comparison_results["statistics"]["total_paragraphs"] * 100 if comparison_results["statistics"]["total_paragraphs"] > 0 else 0,
            "table_similarity_percentage": comparison_results["statistics"]["similar_tables"] / comparison_results["statistics"]["total_tables"] * 100 if comparison_results["statistics"]["total_tables"] > 0 else 0
        },
        "paragraph_details": [],
        "table_details": []
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
    
    # 添加表格詳情
    for result in comparison_results["table_results"]:
        if show_all_content or not result["is_similar"]:
            detail = {
                "original_index": result["original_index"],
                "matched_index": result["matched_index"],
                "original_table": result["original_table"],
                "matched_table": result["matched_table"],
                "matched_page": result["matched_page"],
                "similarity": result["similarity"],
                "is_similar": result["is_similar"],
                "diff_html": format_table_diff_html(result["original_table"], result["matched_table"], result["diff_matrix"]) if result["diff_matrix"] is not None else None
            }
            
            report["table_details"].append(detail)
    
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

def format_table_diff_html(original_table, matched_table, diff_matrix):
    """格式化表格差異為HTML"""
    if original_table is None or matched_table is None or diff_matrix is None:
        return ""
    
    # 將表格轉換為DataFrame
    df1 = pd.DataFrame(original_table)
    df2 = pd.DataFrame(matched_table)
    
    # 調整表格大小，使其具有相同的行數和列數
    max_rows = max(len(df1), len(df2))
    max_cols = max(len(df1.columns), len(df2.columns))
    
    # 生成HTML表格
    html = '<div class="table-container"><table>'
    
    # 添加表頭
    html += '<tr>'
    for j in range(max_cols):
        col_name = df1.columns[j] if j < len(df1.columns) else ""
        html += f'<th>{col_name}</th>'
    html += '</tr>'
    
    # 添加表格內容
    for i in range(max_rows):
        html += '<tr>'
        for j in range(max_cols):
            cell1 = df1.iloc[i, j] if i < len(df1) and j < len(df1.columns) else ""
            cell2 = df2.iloc[i, j] if i < len(df2) and j < len(df2.columns) else ""
            
            # 獲取單元格差異
            diff = diff_matrix[i, j] if i < diff_matrix.shape[0] and j < diff_matrix.shape[1] else None
            
            # 生成單元格HTML
            if diff is None:
                cell_html = f'<td>{cell1}</td>'
            else:
                # 計算相似度
                similarity = 0.0
                if diff:
                    equal_count = sum(1 for tag, _, _, _, _ in diff if tag == 'equal')
                    total_count = len(diff)
                    similarity = equal_count / total_count if total_count > 0 else 0.0
                
                # 根據相似度設置樣式
                if similarity >= 0.9:
                    cell_html = f'<td>{cell1}</td>'
                elif similarity >= 0.7:
                    cell_html = f'<td class="diff-warning">{cell1}<br><span class="diff-added">{cell2}</span></td>'
                else:
                    cell_html = f'<td class="diff-error"><span class="diff-removed">{cell1}</span><br><span class="diff-added">{cell2}</span></td>'
            
            html += cell_html
        html += '</tr>'
    
    html += '</table></div>'
    
    return html
