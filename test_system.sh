#!/bin/bash

# 文稿比對系統測試腳本

echo "開始測試文稿比對系統..."

# 檢查Python環境
echo "檢查Python環境..."
python3 --version
if [ $? -ne 0 ]; then
    echo "錯誤: Python3 未安裝"
    exit 1
fi

# 檢查依賴庫
echo "檢查依賴庫..."
pip3 list | grep streamlit
pip3 list | grep PyMuPDF
pip3 list | grep python-docx
pip3 list | grep pandas
pip3 list | grep numpy
pip3 list | grep Pillow

# 檢查文件是否存在
echo "檢查文件是否存在..."
if [ ! -f "app_updated.py" ]; then
    echo "錯誤: app_updated.py 不存在"
    exit 1
fi

if [ ! -f "enhanced_extraction.py" ]; then
    echo "錯誤: enhanced_extraction.py 不存在"
    exit 1
fi

if [ ! -f "qwen_api.py" ]; then
    echo "錯誤: qwen_api.py 不存在"
    exit 1
fi

# 檢查文件語法
echo "檢查Python語法..."
python3 -m py_compile app_updated.py
if [ $? -ne 0 ]; then
    echo "錯誤: app_updated.py 語法錯誤"
    exit 1
fi

# 檢查文檔
echo "檢查文檔..."
if [ ! -f "user_guide.md" ]; then
    echo "錯誤: user_guide.md 不存在"
    exit 1
fi

if [ ! -f "technical_doc.md" ]; then
    echo "錯誤: technical_doc.md 不存在"
    exit 1
fi

# 測試完成
echo "測試完成，系統檢查通過！"
echo "請使用以下命令啟動應用程式："
echo "streamlit run app_updated.py"
