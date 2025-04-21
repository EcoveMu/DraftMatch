import re
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher

def normalize_number(s: str) -> str:
    """標準化數值格式，處理單位換算。"""
    if not isinstance(s, str):
        return str(s)
    
    match = re.match(r'^(\d+(?:\.\d+)?)(\s*)([A-Za-z%]+)?$', s.strip())
    if match:
        value = float(match.group(1))
        unit = match.group(3) or ''
        
        # 單位換算表
        unit_conversions = {
            'g': ('kg', 0.001),
            'gm': ('kg', 0.001),
            'mg': ('g', 0.001),
            'ml': ('l', 0.001),
            'cm': ('m', 0.01),
            'mm': ('m', 0.001)
        }
        
        if unit.lower() in unit_conversions:
            new_unit, factor = unit_conversions[unit.lower()]
            value *= factor
            unit = new_unit
            
        return f"{value}{unit}"
    return s

def find_best_match(target: str, candidates: List[str], threshold: float = 0.8) -> Optional[str]:
    """找出最佳匹配的字串。"""
    best_match = None
    best_ratio = 0
    
    for candidate in candidates:
        ratio = SequenceMatcher(None, target, candidate).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = candidate
            
    return best_match

def align_columns(pdf_table: Dict[str, Any], word_table: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """對齊表格欄位，返回欄位映射和差異資訊。"""
    pdf_headers = pdf_table["data"][0] if pdf_table["data"] else []
    word_headers = word_table["data"][0] if word_table["data"] else []
    
    mapping = {}
    col_name_diffs = []
    
    # 建立欄位映射
    for pdf_col in pdf_headers:
        # 先嘗試完全匹配
        if pdf_col in word_headers:
            mapping[pdf_col] = pdf_col
        else:
            # 嘗試找最相似的欄位名稱
            match = find_best_match(pdf_col, word_headers)
            mapping[pdf_col] = match
            if match and pdf_col != match:
                col_name_diffs.append((pdf_col, match))
    
    # 檢查欄位順序差異
    col_order_diff = pdf_headers != [mapping.get(pc) for pc in pdf_headers]
    
    return {
        "column_mapping": mapping,
        "name_differences": col_name_diffs,
        "order_different": col_order_diff
    }

def diff_table(word_table: Dict[str, Any], pdf_table: Dict[str, Any]) -> Dict[str, Any]:
    """計算兩個表格的詳細差異。"""
    alignment = align_columns(pdf_table, word_table)
    cell_differences = []
    row_similarities = []
    
    # 跳過表頭行
    pdf_data = pdf_table["data"][1:] if pdf_table["data"] else []
    word_data = word_table["data"][1:] if word_table["data"] else []
    
    # 比較每一行
    for row_idx, (pdf_row, word_row) in enumerate(zip(pdf_data, word_data)):
        matched_cells = 0
        row_diffs = []
        
        # 根據欄位映射比較每個儲存格
        for pdf_col_idx, pdf_cell in enumerate(pdf_row):
            pdf_header = pdf_table["data"][0][pdf_col_idx]
            word_col_name = alignment["column_mapping"].get(pdf_header)
            
            if word_col_name:
                word_col_idx = word_table["data"][0].index(word_col_name)
                word_cell = word_row[word_col_idx]
                
                # 標準化數值後比較
                pdf_value = normalize_number(pdf_cell)
                word_value = normalize_number(word_cell)
                
                if pdf_value != word_value:
                    row_diffs.append({
                        "row": row_idx + 2,  # +2 因為跳過表頭且從1開始計數
                        "col": pdf_header,
                        "pdf_value": pdf_cell,
                        "word_value": word_cell,
                        "diff_type": "內容差異"
                    })
                else:
                    matched_cells += 1
        
        # 計算該行相似度
        total_cols = len(pdf_row)
        row_sim = matched_cells / total_cols if total_cols > 0 else 0
        row_similarities.append({
            "row": row_idx + 2,
            "similarity": row_sim,
            "differences": row_diffs
        })
        cell_differences.extend(row_diffs)
    
    return {
        "column_alignment": alignment,
        "cell_differences": cell_differences,
        "row_similarities": row_similarities
    }

def table_similarity(word_table: Dict[str, Any], pdf_table: Dict[str, Any]) -> float:
    """計算兩個表格的整體相似度。"""
    # 如果表格結構完全不同，直接返回低相似度
    if not word_table["data"] or not pdf_table["data"]:
        return 0.0
        
    # 比較表格標題相似度
    title_sim = SequenceMatcher(None, 
                              word_table.get("title", ""), 
                              pdf_table.get("title", "")).ratio()
    
    # 計算儲存格匹配率
    total_cells = 0
    matched_cells = 0
    
    # 跳過表頭行
    word_data = word_table["data"][1:] if len(word_table["data"]) > 1 else []
    pdf_data = pdf_table["data"][1:] if len(pdf_table["data"]) > 1 else []
    
    # 取較短的行數
    min_rows = min(len(word_data), len(pdf_data))
    
    for i in range(min_rows):
        word_row = word_data[i]
        pdf_row = pdf_data[i]
        min_cols = min(len(word_row), len(pdf_row))
        
        for j in range(min_cols):
            total_cells += 1
            if normalize_number(word_row[j]) == normalize_number(pdf_row[j]):
                matched_cells += 1
                
    cell_sim = matched_cells / total_cells if total_cells > 0 else 0
    
    # 綜合考慮標題相似度和內容相似度
    return 0.3 * title_sim + 0.7 * cell_sim

def compare_tables(word_tables: List[Dict[str, Any]], 
                  pdf_tables: List[Dict[str, Any]], 
                  similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """比對 Word 與 PDF 的表格列表，傳回比對差異結果列表。"""
    table_matches = []
    used_word_indices = set()
    
    for pdf_table in pdf_tables:
        best_match = None
        best_sim = 0.0
        best_match_diff = None
        
        for word_table in word_tables:
            if word_table["index"] in used_word_indices:
                continue
                
            sim = table_similarity(word_table, pdf_table)
            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                best_match = word_table
                best_match_diff = diff_table(word_table, pdf_table)
                
        if best_match:
            used_word_indices.add(best_match["index"])
            table_matches.append({
                "pdf_page": pdf_table.get("page"),
                "pdf_index": pdf_table["index"],
                "pdf_title": pdf_table.get("title", ""),
                "word_index": best_match["index"],
                "word_page": best_match.get("page"),
                "word_title": best_match.get("title", ""),
                "similarity": best_sim,
                "diff": best_match_diff
            })
            
    return table_matches