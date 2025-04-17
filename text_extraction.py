import docx
import re
import fitz  # PyMuPDF
import os
import tempfile
import unicodedata

def extract_text_from_word(word_file):
    """從Word文件中提取文本"""
    doc = docx.Document(word_file)
    
    paragraphs = []
    tables = []
    
    # 提取段落
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            paragraphs.append({
                "index": i,
                "content": text,
                "type": "paragraph"
            })
    
    # 提取表格
    for i, table in enumerate(doc.tables):
        table_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                row_data.append(cell.text.strip())
            table_data.append(row_data)
        
        if any(any(cell for cell in row) for row in table_data):
            tables.append({
                "index": i,
                "content": table_data,
                "type": "table"
            })
    
    return {
        "paragraphs": paragraphs,
        "tables": tables
    }

def extract_text_from_pdf_with_page_info(pdf_file):
    """從PDF文件中提取文本，並保留頁碼信息"""
    # 保存上傳的PDF文件到臨時文件
    temp_dir = tempfile.mkdtemp()
    temp_pdf_path = os.path.join(temp_dir, "temp.pdf")
    
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.getvalue())
    
    # 打開PDF文件
    doc = fitz.open(temp_pdf_path)
    
    paragraphs = []
    
    # 提取每一頁的文本
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        # 分割文本為段落
        page_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # 添加頁碼信息
        for i, para_text in enumerate(page_paragraphs):
            paragraphs.append({
                "index": len(paragraphs),
                "content": para_text,
                "type": "paragraph",
                "page": page_num + 1
            })
    
    # 關閉PDF文件
    doc.close()
    
    return paragraphs

def normalize_chinese_text(text):
    """標準化中文文本，處理全半形、簡繁體等問題"""
    # 將全形字符轉換為半形
    text = unicodedata.normalize('NFKC', text)
    
    # 移除多餘的空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_extracted_text(text):
    """清理提取的文本，處理常見的編碼問題"""
    # 移除控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 標準化換行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 移除連續的換行符
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 標準化中文文本
    text = normalize_chinese_text(text)
    
    return text

def extract_page_numbers(text):
    """從文本中提取頁碼信息"""
    # 匹配常見的頁碼格式
    page_patterns = [
        r'第\s*(\d+)\s*頁',
        r'Page\s*(\d+)',
        r'P\.?\s*(\d+)',
        r'頁\s*(\d+)',
        r'-\s*(\d+)\s*-'
    ]
    
    for pattern in page_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
    
    return None

def preprocess_text(text, ignore_options):
    """根據忽略選項預處理文本"""
    if ignore_options.get("ignore_space", False):
        text = re.sub(r'\s+', '', text)
    
    if ignore_options.get("ignore_punctuation", False):
        text = re.sub(r'[^\w\s]', '', text)
    
    if ignore_options.get("ignore_case", False):
        text = text.lower()
    
    if ignore_options.get("ignore_newline", False):
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
    
    return text

def detect_encoding(text):
    """檢測文本編碼，處理亂碼問題"""
    # 檢查是否包含亂碼的特徵
    if re.search(r'[\uFFFD\u25A1\u2753]', text):
        return "亂碼"
    
    # 檢查是否包含中文字符
    if re.search(r'[\u4e00-\u9fff]', text):
        return "中文"
    
    # 檢查是否包含日文字符
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return "日文"
    
    # 檢查是否包含韓文字符
    if re.search(r'[\uAC00-\uD7AF]', text):
        return "韓文"
    
    # 默認為英文或其他西方語言
    return "英文"

def extract_and_process_documents(word_file, pdf_file, use_ocr=False, ocr_engine="tesseract", ocr=None):
    """提取並處理Word和PDF文件的內容"""
    # 提取Word文件內容
    word_data = extract_text_from_word(word_file)
    
    # 清理Word文件內容
    for i, para in enumerate(word_data["paragraphs"]):
        word_data["paragraphs"][i]["content"] = clean_extracted_text(para["content"])
    
    # 提取PDF文件內容
    pdf_paragraphs = extract_text_from_pdf_with_page_info(pdf_file)
    
    # 清理PDF文件內容
    for i, para in enumerate(pdf_paragraphs):
        pdf_paragraphs[i]["content"] = clean_extracted_text(para["content"])
    
    # 如果使用OCR，則使用OCR提取PDF文本
    if use_ocr and ocr_engine == "Qwen" and ocr:
        # 使用Qwen OCR提取文本
        ocr_results = ocr.extract_text_from_pdf(pdf_file)
        
        # 如果OCR成功，則使用OCR結果
        if ocr_results and not isinstance(ocr_results, str):
            pdf_paragraphs = []
            for page_num, page_text in ocr_results.items():
                # 分割文本為段落
                page_paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                
                # 添加頁碼信息
                for i, para_text in enumerate(page_paragraphs):
                    pdf_paragraphs.append({
                        "index": len(pdf_paragraphs),
                        "content": clean_extracted_text(para_text),
                        "type": "paragraph",
                        "page": page_num
                    })
    
    pdf_data = {
        "paragraphs": pdf_paragraphs,
        "tables": []  # 暫時不處理表格
    }
    
    return word_data, pdf_data
