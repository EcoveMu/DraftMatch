import os
import tempfile
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import re
import numpy as np
import pandas as pd
import streamlit as st

def enhanced_pdf_extraction(pdf_file, use_ocr=True):
    """
    增強的PDF文本提取函數，結合多種方法提取文本
    """
    # 創建臨時目錄
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp.pdf")
    
    # 保存上傳的文件
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getvalue())
    
    # 存儲所有提取的段落
    all_paragraphs = []
    all_tables = []
    
    # 方法1: 使用PyMuPDF直接提取
    try:
        doc = fitz.open(temp_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # 分割段落 - 使用更智能的段落分割
            lines = text.split('\n')
            current_para = ""
            
            for line in lines:
                line = line.strip()
                if not line:  # 空行表示段落分隔
                    if current_para:
                        all_paragraphs.append({
                            'content': current_para,
                            'page': page_num + 1,
                            'method': 'pymupdf',
                            'confidence': 0.9
                        })
                        current_para = ""
                else:
                    # 檢查是否是新段落的開始（例如縮進或標題）
                    if (not current_para or 
                        line[0].isupper() or 
                        re.match(r'^[0-9]+\.', line) or
                        len(line) < 20):  # 短行可能是標題
                        if current_para:
                            all_paragraphs.append({
                                'content': current_para,
                                'page': page_num + 1,
                                'method': 'pymupdf',
                                'confidence': 0.9
                            })
                            current_para = line
                    else:
                        current_para += " " + line
            
            # 添加最後一個段落
            if current_para:
                all_paragraphs.append({
                    'content': current_para,
                    'page': page_num + 1,
                    'method': 'pymupdf',
                    'confidence': 0.9
                })
    except Exception as e:
        st.warning(f"PyMuPDF提取失敗: {e}")
    
    # 方法2: 使用pdfplumber提取
    try:
        with pdfplumber.open(temp_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    # 使用與PyMuPDF相同的段落分割邏輯
                    lines = text.split('\n')
                    current_para = ""
                    
                    for line in lines:
                        line = line.strip()
                        if not line:  # 空行表示段落分隔
                            if current_para:
                                all_paragraphs.append({
                                    'content': current_para,
                                    'page': page_num + 1,
                                    'method': 'pdfplumber',
                                    'confidence': 0.85
                                })
                                current_para = ""
                        else:
                            # 檢查是否是新段落的開始
                            if (not current_para or 
                                line[0].isupper() or 
                                re.match(r'^[0-9]+\.', line) or
                                len(line) < 20):
                                if current_para:
                                    all_paragraphs.append({
                                        'content': current_para,
                                        'page': page_num + 1,
                                        'method': 'pdfplumber',
                                        'confidence': 0.85
                                    })
                                    current_para = line
                            else:
                                current_para += " " + line
                    
                    # 添加最後一個段落
                    if current_para:
                        all_paragraphs.append({
                            'content': current_para,
                            'page': page_num + 1,
                            'method': 'pdfplumber',
                            'confidence': 0.85
                        })
                
                # 提取表格
                tables = page.extract_tables()
                for table in tables:
                    # 過濾空表格
                    if table and any(any(cell for cell in row) for row in table):
                        # 獲取表格上下文（前後文本）
                        table_bbox = page.find_tables()[0].bbox if page.find_tables() else None
                        context = {
                            'previous_text': '',
                            'next_text': ''
                        }
                        
                        if table_bbox:
                            # 表格前的文本
                            above_table = page.crop((0, 0, page.width, table_bbox[1]))
                            context['previous_text'] = above_table.extract_text() or ''
                            
                            # 表格後的文本
                            below_table = page.crop((0, table_bbox[3], page.width, page.height))
                            context['next_text'] = below_table.extract_text() or ''
                        
                        all_tables.append({
                            'content': table,
                            'page': page_num + 1,
                            'method': 'pdfplumber',
                            'context': context,
                            'title': context['previous_text'].split('\n')[-1] if context['previous_text'] else ''
                        })
    except Exception as e:
        st.warning(f"pdfplumber提取失敗: {e}")
    
    # 方法3: 使用OCR提取 (如果啟用)
    if use_ocr:
        try:
            st.info("正在使用OCR提取文本，這可能需要一些時間...")
            
            # 使用PyMuPDF將PDF頁面轉換為圖像
            doc = fitz.open(temp_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                img_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                pix.save(img_path)
                
                # 使用OCR提取文本
                text = pytesseract.image_to_string(Image.open(img_path), lang="chi_tra+eng")
                
                # 使用與其他方法相同的段落分割邏輯
                lines = text.split('\n')
                current_para = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:  # 空行表示段落分隔
                        if current_para:
                            all_paragraphs.append({
                                'content': current_para,
                                'page': page_num + 1,
                                'method': 'ocr',
                                'confidence': 0.7
                            })
                            current_para = ""
                    else:
                        # 檢查是否是新段落的開始
                        if (not current_para or 
                            line[0].isupper() or 
                            re.match(r'^[0-9]+\.', line) or
                            len(line) < 20):
                            if current_para:
                                all_paragraphs.append({
                                    'content': current_para,
                                    'page': page_num + 1,
                                    'method': 'ocr',
                                    'confidence': 0.7
                                })
                                current_para = line
                        else:
                            current_para += " " + line
                
                # 添加最後一個段落
                if current_para:
                    all_paragraphs.append({
                        'content': current_para,
                        'page': page_num + 1,
                        'method': 'ocr',
                        'confidence': 0.7
                    })
        except Exception as e:
            st.warning(f"OCR提取失敗: {e}")
    
    # 去除重複段落
    unique_paragraphs = []
    seen_contents = set()
    
    for para in all_paragraphs:
        # 標準化內容以檢測重複
        normalized_content = re.sub(r'\s+', ' ', para['content']).strip().lower()
        
        # 如果內容太短或已存在，則跳過
        if len(normalized_content) < 5 or normalized_content in seen_contents:
            continue
        
        seen_contents.add(normalized_content)
        unique_paragraphs.append(para)
    
    # 按頁碼和段落位置排序
    unique_paragraphs.sort(key=lambda x: (x['page'], all_paragraphs.index(x)))
    
    # 清理臨時目錄
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass
    
    return {
        "paragraphs": unique_paragraphs,
        "tables": all_tables
    }

def improved_matching_algorithm(doc1_paragraphs, doc2_paragraphs):
    """
    改進的段落匹配算法，避免將不同的原始段落匹配到同一個PDF段落
    包含錯誤處理和降級機制，確保在scipy不可用時仍能提供基本功能
    """
    # 計算所有段落對之間的相似度
    similarity_matrix = np.zeros((len(doc1_paragraphs), len(doc2_paragraphs)))
    
    for i, para1 in enumerate(doc1_paragraphs):
        for j, para2 in enumerate(doc2_paragraphs):
            # 使用簡單的詞袋模型計算相似度
            words1 = set(re.sub(r'[^\w\s]', '', para1['content'].lower()).split())
            words2 = set(re.sub(r'[^\w\s]', '', para2['content'].lower()).split())
            
            if not words1 or not words2:
                similarity = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0.0
            
            similarity_matrix[i, j] = similarity
    
    # 嘗試使用匈牙利算法找到最佳匹配
    # 如果scipy不可用，則使用貪婪算法作為替代
    try:
        # 嘗試導入scipy
        from scipy.optimize import linear_sum_assignment
        
        # 將相似度轉換為成本（1-相似度）
        cost_matrix = 1 - similarity_matrix
        
        # 找到最佳匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 創建匹配結果
        matches = []
        for i, j in zip(row_ind, col_ind):
            matches.append({
                "doc1_index": i,
                "doc2_index": j,
                "similarity": similarity_matrix[i, j]
            })
        
        st.success("使用匈牙利算法成功匹配段落")
        return matches
    
    except (ImportError, ModuleNotFoundError) as e:
        st.warning(f"scipy模組不可用，使用貪婪算法作為替代: {str(e)}")
        
        # 使用貪婪算法作為替代
        matches = []
        used_doc2_indices = set()
        
        # 為每個doc1段落找到最佳匹配的doc2段落
        for i in range(len(doc1_paragraphs)):
            best_similarity = -1
            best_j = -1
            
            # 找到最相似的未使用的doc2段落
            for j in range(len(doc2_paragraphs)):
                if j not in used_doc2_indices and similarity_matrix[i, j] > best_similarity:
                    best_similarity = similarity_matrix[i, j]
                    best_j = j
            
            # 如果找到匹配，則添加到結果中
            if best_j != -1:
                matches.append({
                    "doc1_index": i,
                    "doc2_index": best_j,
                    "similarity": best_similarity
                })
                used_doc2_indices.add(best_j)
            else:
                # 如果沒有未使用的doc2段落，則選擇最相似的段落
                best_j = np.argmax(similarity_matrix[i])
                matches.append({
                    "doc1_index": i,
                    "doc2_index": best_j,
                    "similarity": similarity_matrix[i, best_j]
                })
        
        return matches
