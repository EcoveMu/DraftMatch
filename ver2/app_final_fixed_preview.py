
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
import streamlit as st

# 設置頁面配置
st.set_page_config(
    page_title="期刊比對系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定義CSS樣式
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
</style>
""", unsafe_allow_html=True)

# 頁面標題與檔案上傳區
st.title("📄 期刊內容比對系統")
st.markdown("請上傳原始 Word 檔與美編後 PDF 進行比對。")

word_file = st.file_uploader("⬆️ 上傳 Word 原始檔", type=["docx"])
pdf_file = st.file_uploader("⬆️ 上傳 PDF 美編報告", type=["pdf"])

if word_file and pdf_file:
    st.success("📁 檔案上傳成功，請點選左側選項進行比對！")

# TODO: 加入比對邏輯與比對結果視覺化
