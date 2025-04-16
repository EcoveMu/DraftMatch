# 期刊比對系統使用說明

## 系統簡介

期刊比對系統是一個專為出版社和編輯團隊設計的工具，用於比對原始Word文件與美編後PDF文件的內容差異，幫助校對人員快速找出不一致之處，大幅提高校對效率。

本系統具有以下特點：
- 高效的文本提取：從Word和PDF文件中準確提取文本和表格
- 多種比對算法：支持精確比對、語意比對和混合比對
- 視覺化差異顯示：直觀展示文本差異，並在PDF頁面上標記問題區域
- 生成式AI增強：利用AI技術提高比對準確性，提供智能分析和報告
- 自定義選項：支持自定義OCR和AI模型，滿足不同需求

## 系統需求

- Python 3.8或更高版本
- 以下Python套件：
  - streamlit
  - python-docx
  - PyMuPDF
  - pdfplumber
  - pytesseract
  - difflib
  - numpy
  - pandas
  - Pillow
  - requests
  - sentence-transformers (可選，用於語意比對)

## 安裝指南

1. 克隆或下載系統代碼到本地目錄
2. 安裝所需的Python套件：
   ```
   pip install -r requirements.txt
   ```
3. 安裝Tesseract OCR（可選，如果使用Tesseract作為OCR引擎）：
   - Windows: 下載並安裝[Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-chi-tra`
   - macOS: `brew install tesseract`

## 啟動系統

在系統目錄下執行以下命令啟動應用：
```
streamlit run app_final.py
```

系統將在本地啟動，並在瀏覽器中打開（通常是http://localhost:8501）。

## 使用指南

### 1. 上傳文件

1. 在左側的「原始Word文件」區域上傳原始Word文檔
2. 在右側的「美編後PDF文件」區域上傳美編後的PDF文檔
3. 點擊「開始比對」按鈕開始處理

### 2. 系統設置

在左側邊欄可以調整以下設置：

#### 比對設置
- **比對模式**：選擇精確比對、語意比對或混合比對
  - 精確比對：基於字符級別的精確比對，適合要求高度一致的場景
  - 語意比對：基於語義的比對，能夠識別表達相同意思但用詞不同的內容
  - 混合比對：結合精確比對和語意比對，平衡準確性和靈活性
- **相似度閾值**：設置判定為相似的最低閾值（0-1之間）

#### 忽略選項
- **忽略空格**：比對時忽略空格差異
- **忽略標點符號**：比對時忽略標點符號差異
- **忽略大小寫**：比對時忽略大小寫差異
- **忽略換行**：比對時忽略換行差異

#### OCR設置
- **使用OCR提取PDF文本**：啟用/禁用OCR功能
- **OCR引擎**：選擇OCR引擎
  - tesseract：使用開源的Tesseract OCR引擎
  - Qwen：使用阿里雲的Qwen模型進行OCR（推薦）
  - 自定義API：使用自定義的OCR API
- **OCR API密鑰**：如果使用Qwen或自定義API，需要提供API密鑰
- **自定義OCR API URL**：如果使用自定義API，需要提供API URL

#### 生成式AI設置
- **使用生成式AI增強功能**：啟用/禁用AI功能
- **AI模型**：選擇AI模型
  - Qwen (免費)：使用免費的Qwen API
  - OpenAI：使用OpenAI的API
  - Anthropic：使用Anthropic的API
  - 自定義API：使用自定義的AI API
- **AI API密鑰**：如果使用OpenAI、Anthropic或自定義API，需要提供API密鑰
- **AI API URL**：如果使用自定義API，需要提供API URL
- **模型名稱**：選擇具體的模型名稱

#### 顯示設置
- **差異顯示模式**：選擇字符級別、詞語級別或行級別的差異顯示
- **顯示所有內容**：是否顯示所有內容（包括相似的段落和表格）

#### 視覺化設置
- **使用視覺化差異顯示**：啟用/禁用視覺化差異顯示

### 3. 查看比對結果

比對完成後，系統將顯示以下內容：

#### 比對結果摘要
- 總段落數、相似段落數、不同段落數
- 總表格數、相似表格數、不同表格數
- 段落相似度、表格相似度、整體相似度
- AI分析結果（如果啟用了AI功能）
- AI摘要報告（如果啟用了AI功能）

#### 段落比對
- 顯示原始文本和美編後文本的對比
- 視覺化標記差異
- 顯示PDF頁面預覽，並標記問題區域
- 顯示精確相似度和語意相似度

#### 表格比對
- 顯示原始表格和美編後表格的對比
- 顯示PDF頁面預覽
- 顯示表格相似度

## 常見問題

### 1. OCR提取的文本不準確怎麼辦？
- 嘗試使用Qwen OCR引擎，它通常比Tesseract有更好的效果
- 調整忽略選項，例如忽略空格、標點符號等
- 如果有自己的OCR API，可以使用自定義API選項

### 2. 比對結果有誤判怎麼辦？
- 調整相似度閾值，提高閾值可以減少誤判
- 嘗試不同的比對模式，例如從精確比對切換到混合比對
- 啟用生成式AI功能，利用AI提高比對準確性

### 3. 系統運行緩慢怎麼辦？
- 減少PDF頁數或分批處理
- 禁用視覺化差異顯示
- 如果不需要OCR，可以禁用OCR功能
- 如果不需要AI功能，可以禁用AI功能

### 4. 如何獲取API密鑰？
- Qwen API：訪問[阿里雲官網](https://www.aliyun.com/)註冊並申請
- OpenAI API：訪問[OpenAI官網](https://openai.com/)註冊並申請
- Anthropic API：訪問[Anthropic官網](https://www.anthropic.com/)註冊並申請

## 技術支持

如有任何問題或建議，請聯繫技術支持團隊。
