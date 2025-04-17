import streamlit as st
import os
import tempfile
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import time
import json
import random
import numpy as np
import pandas as pd
import fitz  # PyMuPDF

# 設置頁面配置
st.set_page_config(
    page_title="期刊比對系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定義CSS
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #2C2C2C;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .stMarkdown p {
        color: white;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput input, .stNumberInput input {
        background-color: #2C2C2C;
        color: white;
    }
    .stSelectbox select {
        background-color: #2C2C2C;
        color: white;
    }
    .stRadio label {
        color: white;
    }
    .stCheckbox label {
        color: white;
    }
    .stExpander {
        background-color: #2C2C2C;
    }
    .stExpander summary {
        color: white;
    }
    .stDataFrame {
        background-color: #2C2C2C;
    }
    .stDataFrame th {
        background-color: #4CAF50;
        color: white;
    }
    .stDataFrame td {
        color: white;
    }
    .stSidebar {
        background-color: #2C2C2C;
    }
    .stSidebar .stMarkdown p, .stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 {
        color: white;
    }
    .stSidebar .stRadio label, .stSidebar .stCheckbox label {
        color: white;
    }
    .stSidebar .stSelectbox select {
        color: white;
    }
    .stSidebar [data-baseweb="select"] {
        color: white;
    }
    .stSidebar [data-baseweb="select"] > div {
        color: white;
    }
    .stSidebar [data-baseweb="select"] > div > div {
        color: white;
    }
    .stSidebar [data-baseweb="select"] svg {
        color: white;
    }
    .stSlider [data-baseweb="slider"] {
        background-color: #4CAF50;
    }
    .diff-highlight {
        background-color: rgba(255, 0, 0, 0.3);
        padding: 2px;
        border-radius: 3px;
        color: black;
    }
    .diff-highlight-green {
        background-color: rgba(0, 255, 0, 0.3);
        padding: 2px;
        border-radius: 3px;
        color: black;
    }
    .diff-section {
        border: 1px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .diff-title {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 10px;
        color: white;
    }
    .diff-content {
        display: flex;
        flex-direction: row;
        gap: 20px;
    }
    .diff-original, .diff-edited {
        flex: 1;
        padding: 10px;
        background-color: #2C2C2C;
        border-radius: 5px;
        color: white;
    }
    .diff-original-title, .diff-edited-title {
        font-weight: bold;
        margin-bottom: 5px;
        color: white;
    }
    .similarity-score {
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 0px 5px 5px 5px;
    }
    /* 確保選擇後的選項文字為白色 */
    .stSelectbox [data-baseweb="select"] [data-testid="stMarkdown"] p {
        color: white !important;
    }
    /* 確保下拉選單中的選項文字為白色 */
    [data-baseweb="menu"] [data-testid="stMarkdown"] p {
        color: white !important;
    }
    /* 確保選擇框中的文字為白色 */
    [data-baseweb="select"] [data-testid="stMarkdown"] p {
        color: white !important;
    }
    /* 確保所有文字在深色背景下都是白色 */
    p, h1, h2, h3, h4, h5, h6, li, span, div {
        color: white !important;
    }
    /* 確保表格中的文字為白色 */
    table, th, td {
        color: white !important;
    }
    /* 移除Streamlit警告訊息 */
    .stWarning {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 模擬從Word文件提取文本
def extract_text_from_word(file_path):
    # 這裡是模擬數據，實際應用中應該使用python-docx庫提取真實數據
    paragraphs = [
        "公司治理評鑑-連續10屆金管會公司治理評鑑TOP 5%",
        "ESG評比- S&P Global 標普全球評級TOP 5%",
        "12+項ESG Global 評鑑全球評級TOP 5%",
        "國際肯定-上櫃+臺灣企業永續獎+天下CSR企業公民獎",
        "產業領導人-入選永續指數成分股+企業永續報告書白金獎",
        "回收再利用-回收率達99.9%",
        "循環經濟-台灣高科技產業的廢溶劑回收再利用",
        "節能減碳-2050淨零排放",
        "資源循環-資源循環零廢棄物",
        "表格 1: 2023年度財務數據",
        "項目,金額",
        "營收,1,234,567",
        "淨利,456,789"
    ]
    return paragraphs

# 模擬從PDF文件提取文本
def extract_text_from_pdf(file_path):
    # 這裡是模擬數據，實際應用中應該使用PyMuPDF或pdfplumber庫提取真實數據
    pages = {
        1: ["公司治理", "ESG評比", "國際肯定"],
        2: ["產業領導", "回收再利用", "循環經濟"],
        3: ["節能減碳", "資源循環"],
        4: ["公司治理評鑑-連續10屆金管會公司治理評鑑TOP 5%"],
        5: ["ESG評比- S&P Global 標普全球評級TOP 5%"],
        6: ["12+項ESG Global 評鑑全球評級TOP 5%"],
        7: ["國際肯定-上櫃+臺灣企業永續獎+天下CSR企業公民獎"],
        8: ["產業領導人-入選永續指數成分股+企業永續報告書白金獎"],
        9: ["回收再利用-回收率達99.9%"],
        10: ["循環經濟-台灣高科技產業的廢溶劑回收再利用"],
        11: ["節能減碳-2050淨零排放"],
        12: ["資源循環-資源循環零廢棄物"]
    }
    return pages

# 模擬比對算法
def compare_documents(word_text, pdf_pages, similarity_threshold=0.8):
    # 這裡是模擬比對結果，實際應用中應該使用更複雜的算法
    total_paragraphs = len(word_text)
    matched_paragraphs = 0
    not_found_paragraphs = 0
    
    comparison_results = []
    table_results = []
    
    for i, paragraph in enumerate(word_text):
        if "表格" in paragraph:
            # 表格比對邏輯
            table_results.append({
                "paragraph_id": i + 1,
                "original_text": paragraph,
                "matched_text": "未找到匹配內容",
                "page_number": "未找到",
                "similarity_score": 0.0,
                "is_matched": False,
                "is_table": True
            })
            not_found_paragraphs += 1
        else:
            # 文本段落比對邏輯
            best_match = None
            best_score = 0
            best_page = None
            
            for page_num, page_content in pdf_pages.items():
                for content in page_content:
                    # 簡單的相似度計算，實際應用中應該使用更複雜的算法
                    similarity = calculate_similarity(paragraph, content)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = content
                        best_page = page_num
            
            if best_score >= similarity_threshold:
                comparison_results.append({
                    "paragraph_id": i + 1,
                    "original_text": paragraph,
                    "matched_text": best_match,
                    "page_number": best_page,
                    "similarity_score": best_score,
                    "is_matched": True,
                    "is_table": False
                })
                matched_paragraphs += 1
            else:
                comparison_results.append({
                    "paragraph_id": i + 1,
                    "original_text": paragraph,
                    "matched_text": "未找到匹配內容",
                    "page_number": "未找到",
                    "similarity_score": best_score,
                    "is_matched": False,
                    "is_table": False
                })
                not_found_paragraphs += 1
    
    summary = {
        "total_paragraphs": total_paragraphs,
        "matched_paragraphs": matched_paragraphs,
        "not_found_paragraphs": not_found_paragraphs,
        "match_rate": matched_paragraphs / total_paragraphs if total_paragraphs > 0 else 0
    }
    
    return comparison_results, table_results, summary

# 計算文本相似度
def calculate_similarity(text1, text2):
    # 這是一個簡單的相似度計算函數，實際應用中應該使用更複雜的算法
    # 例如餘弦相似度、Jaccard相似度等
    common_words = set(text1.split()) & set(text2.split())
    total_words = set(text1.split()) | set(text2.split())
    return len(common_words) / len(total_words) if total_words else 0

# 生成PDF頁面預覽並標記差異
def generate_pdf_preview_with_diff(pdf_path, page_number, diff_text):
    # 創建一個模擬的PDF頁面預覽圖像
    width, height = 800, 1000
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # 嘗試加載字體，如果失敗則使用默認字體
    try:
        font = ImageFont.truetype("Arial", 14)
        title_font = ImageFont.truetype("Arial", 18)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # 繪製頁面標題
    draw.text((20, 20), f"頁面 {page_number}", fill=(0, 0, 0), font=title_font)
    
    # 模擬PDF內容
    if page_number == 4:
        # 第4頁 - 公司治理評鑑
        draw.text((50, 100), "公司治理評鑑", fill=(0, 0, 0), font=title_font)
        text = "連續10屆金管會公司治理評鑑TOP 5%"
        draw.text((50, 150), text, fill=(0, 0, 0), font=font)
        
        # 標記差異
        if "公司治理評鑑" in diff_text:
            # 使用紅色矩形框標記差異
            draw.rectangle([(45, 145), (400, 175)], outline=(255, 0, 0), width=2)
            # 添加箭頭指向差異
            draw.line([(420, 160), (405, 160)], fill=(255, 0, 0), width=2)
            draw.polygon([(405, 155), (405, 165), (395, 160)], fill=(255, 0, 0))
            # 添加差異說明
            draw.text((430, 150), "差異: 原文為「公司治理評鑑-連續10屆金管會公司治理評鑑TOP 5%」", fill=(255, 0, 0), font=font)
    
    elif page_number == 5:
        # 第5頁 - ESG評比
        draw.text((50, 100), "ESG評比", fill=(0, 0, 0), font=title_font)
        text = "S&P Global 標普全球評級TOP 5%"
        draw.text((50, 150), text, fill=(0, 0, 0), font=font)
        
        # 標記差異
        if "ESG評比" in diff_text:
            # 使用紅色矩形框標記差異
            draw.rectangle([(45, 145), (350, 175)], outline=(255, 0, 0), width=2)
            # 添加箭頭指向差異
            draw.line([(370, 160), (355, 160)], fill=(255, 0, 0), width=2)
            draw.polygon([(355, 155), (355, 165), (345, 160)], fill=(255, 0, 0))
            # 添加差異說明
            draw.text((380, 150), "差異: 原文為「ESG評比- S&P Global 標普全球評級TOP 5%」", fill=(255, 0, 0), font=font)
    
    elif page_number == 8:
        # 第8頁 - 產業領導
        draw.text((50, 100), "產業領導", fill=(0, 0, 0), font=title_font)
        text = "入選永續指數成分股+企業永續報告書白金獎"
        draw.text((50, 150), text, fill=(0, 0, 0), font=font)
        
        # 標記差異
        if "產業領導" in diff_text:
            # 使用紅色矩形框標記差異
            draw.rectangle([(45, 145), (450, 175)], outline=(255, 0, 0), width=2)
            # 添加箭頭指向差異
            draw.line([(470, 160), (455, 160)], fill=(255, 0, 0), width=2)
            draw.polygon([(455, 155), (455, 165), (445, 160)], fill=(255, 0, 0))
            # 添加差異說明
            draw.text((480, 150), "差異: 原文為「產業領導人-入選永續指數成分股+企業永續報告書白金獎」", fill=(255, 0, 0), font=font)
    
    # 將圖像轉換為base64編碼
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

# 生成真實PDF頁面預覽
def get_pdf_page_image(page_number):
    """
    返回預先準備好的PDF頁面圖像
    """
    # 這裡我們使用預先準備好的圖像，實際應用中應該從真實PDF中提取
    if page_number == 4:
        image_path = "page4.png"
    elif page_number == 5:
        # 第5頁 - ESG評比
        width, height = 800, 1000
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("Arial", 14)
            title_font = ImageFont.truetype("Arial", 18)
        except IOError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # 繪製頁面內容
        draw.text((50, 50), "ECOVE 環保事業", fill=(0, 0, 0), font=title_font)
        draw.text((50, 100), "ESG評比", fill=(0, 0, 0), font=title_font)
        draw.text((50, 150), "S&P Global 標普全球評級TOP 5%", fill=(0, 0, 0), font=font)
        draw.text((50, 200), "永續經營", fill=(0, 0, 0), font=font)
        draw.text((50, 250), "環境保護", fill=(0, 0, 0), font=font)
        draw.text((50, 300), "社會責任", fill=(0, 0, 0), font=font)
        
        # 繪製紅色框標記差異
        draw.rectangle([(45, 145), (350, 175)], outline=(255, 0, 0), width=2)
        # 添加箭頭指向差異
        draw.line([(370, 160), (355, 160)], fill=(255, 0, 0), width=2)
        draw.polygon([(355, 155), (355, 165), (345, 160)], fill=(255, 0, 0))
        # 添加差異說明
        draw.text((380, 150), "差異: 原文有連字符「-」", fill=(255, 0, 0), font=font)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    elif page_number == 8:
        # 第8頁 - 產業領導
        width, height = 800, 1000
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("Arial", 14)
            title_font = ImageFont.truetype("Arial", 18)
        except IOError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # 繪製頁面內容
        draw.text((50, 50), "ECOVE 環保事業", fill=(0, 0, 0), font=title_font)
        draw.text((50, 100), "產業領導", fill=(0, 0, 0), font=title_font)
        draw.text((50, 150), "入選永續指數成分股+企業永續報告書白金獎", fill=(0, 0, 0), font=font)
        draw.text((50, 200), "永續經營策略", fill=(0, 0, 0), font=font)
        draw.text((50, 250), "環境保護措施", fill=(0, 0, 0), font=font)
        draw.text((50, 300), "社會責任實踐", fill=(0, 0, 0), font=font)
        
        # 繪製紅色框標記差異
        draw.rectangle([(45, 145), (450, 175)], outline=(255, 0, 0), width=2)
        # 添加箭頭指向差異
        draw.line([(470, 160), (455, 160)], fill=(255, 0, 0), width=2)
        draw.polygon([(455, 155), (455, 165), (445, 160)], fill=(255, 0, 0))
        # 添加差異說明
        draw.text((480, 150), "差異: 原文為「產業領導人」", fill=(255, 0, 0), font=font)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    else:
        # 默認頁面
        width, height = 800, 1000
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("Arial", 14)
            title_font = ImageFont.truetype("Arial", 18)
        except IOError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # 繪製頁面內容
        draw.text((50, 50), f"ECOVE 環保事業 - 頁面 {page_number}", fill=(0, 0, 0), font=title_font)
        draw.text((50, 100), "永續經營報告", fill=(0, 0, 0), font=title_font)
        draw.text((50, 150), "環境保護", fill=(0, 0, 0), font=font)
        draw.text((50, 200), "社會責任", fill=(0, 0, 0), font=font)
        draw.text((50, 250), "公司治理", fill=(0, 0, 0), font=font)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

# 主應用程序
def main():
    st.title("期刊比對系統")
    
    # 系統介紹
    with st.expander("系統介紹", expanded=False):
        st.markdown("""
        本系統用於比對原始Word文件與美編後PDF文件的內容差異，幫助校對人員快速找出不一致之處。
        
        ### 主要功能
        - **文本提取**：從Word和PDF文件中提取文本
        - **比對算法**：使用多種算法比對文本差異
        - **視覺化差異**：直觀顯示差異並提供PDF頁面預覽
        - **生成式AI**：使用AI增強比對結果分析
        
        ### 使用方法
        1. 上傳原始Word文件和美編後PDF文件
        2. 設置比對參數
        3. 點擊"開始比對"按鈕
        4. 查看比對結果
        """)
    
    # 側邊欄設置
    st.sidebar.title("系統設置")
    
    # 比對設置
    st.sidebar.header("比對設置")
    comparison_mode = st.sidebar.selectbox(
        "比對模式",
        ["精確比對", "語意比對", "混合比對"],
        index=2
    )
    
    si
(Content truncated due to size limit. Use line ranges to read in chunks)