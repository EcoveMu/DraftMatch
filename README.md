# 文稿比對系統

這個系統用於比對美編前、美編後的文稿，幫助校對人員快速找出不一致之處。

## 主要功能

- 從Word和PDF文件中提取文本內容
- 多種比對模式：精確比對、語意比對、混合比對、上下文感知比對和生成式AI比對
- 高級OCR處理：支援多種OCR引擎，包括Tesseract、EasyOCR和Qwen API
- 生成式AI輔助：支援多種AI模型，提供更準確的段落匹配和智能分析
- 視覺化差異展示：直觀顯示文本差異，支援字符級別、詞語級別和行級別的差異展示
- PDF頁面預覽：顯示PDF頁面並標記差異位置
- 表格比對：支援表格內容的比對和差異展示

## 系統需求

- Python 3.7+
- Streamlit
- PyMuPDF (fitz)
- python-docx
- pandas
- numpy
- PIL
- 其他依賴庫（詳見requirements.txt）

## 快速開始

1. 安裝依賴庫：
   ```
   pip install -r requirements.txt
   ```

2. 啟動應用程式：
   ```
   streamlit run app.py
   ```

3. 在瀏覽器中打開顯示的URL（通常是http://localhost:8501）

4. 上傳原始Word文件和美編後PDF文件，設置比對參數，點擊「開始比對」按鈕

## 文件說明

- `app.py`：主程式文件
- `enhanced_extraction.py`：增強的文本提取模組
- `qwen_api.py`：Qwen API整合模組
- `user_guide.md`：使用者指南
- `technical_doc.md`：技術文件
- `test_system.sh`：測試腳本

## 使用指南

詳細的使用方法請參考 [使用者指南](user_guide.md)。

## 技術文件

系統的技術細節請參考 [技術文件](technical_doc.md)。

## 測試

運行測試腳本檢查系統環境和依賴：

```
bash test_system.sh
```

## 版本歷史

- v1.0.0：基礎版本，實現基本比對功能
- v2.0.0：增加上下文感知匹配功能
- v3.0.0：增加高級OCR處理功能
- v4.0.0：增加生成式AI輔助功能
- v5.0.0：改進UI和使用者體驗
- v6.0.0：整合所有功能，優化比對演算法

## 授權

本項目採用 MIT 授權。
