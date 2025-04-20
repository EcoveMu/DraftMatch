.
├── app.py                   # Streamlit 入口點 / 前端 UI 與整體流程控制
├── requirements.txt         # Python 依賴清單
├── custom.css               # 自訂頁面樣式（顏色、佈局、動畫）
├── comparison_algorithm.py  # 段落比對核心（精確 / 語意 / 混合 / AI）
├── text_extraction.py       # 基礎 Word / PDF 文字抽取與前處理
├── enhanced_extraction.py   # 進階抽取（pdfplumber、PyMuPDF、OCR、表格）
├── generative_ai.py         # QwenAI 包裝：OCR、語意比對、結果分析與摘要
├── custom_ai.py             # 通用 AI 包裝（OpenAI / Anthropic / Qwen / 免費端點）
├── qwen_ocr.py              # 專用 Qwen OCR／表格抽取與影像標註工具
├── qwen_api.py              # 精簡版 Qwen OCR（舊版，相容用途）
└── README.md                # 專案說明（本檔）

## 原作業流程
針對企業報告書，常常會需要比對美編前原稿、美編後完稿兩個檔案之間的差異，找出錯字、漏字、錯誤的數據等
但往往需要耗費極大量的時間
因此本系統旨在協助比對作業，比對邏輯大致上為：
1.以美編後完稿的每一頁作為比對的主體
2.將美編前原稿的內容，與美編後完稿的內容作比對
3.美編後完稿的1頁內容，可能會與將美編前原稿的好幾個段落、不同檔案做比對
4.畫面應顯示美編後完稿的每一句話，可能是比對至美編後完稿的哪幾句



## 文件說明

- **app.py**: 主應用程序文件，集成了所有功能
- **qwen_ocr.py**: 實現了無API key的OCR功能
- **generative_ai.py**: 實現了無API key的生成式AI功能
- **comparison_algorithm.py**: 實現了表格比對算法
- **text_extraction.py**: 文本提取和處理功能

## 使用說明

1. **安裝依賴**
   ```
   pip install streamlit pandas numpy requests sentence-transformers easyocr tabula-py pymupdf pillow
   ```

2. **運行應用**
   ```
   streamlit run app.py
   ```

3. **使用免費API**
   - 默認使用免費API，無需輸入API key
   - 在側邊欄選擇"Qwen (免費)"選項

4. **使用付費API**
   - 在側邊欄選擇"Qwen (API)"選項
   - 輸入您的API key

5. **表格比對功能**
   - 上傳包含表格的文件後，系統會自動進行表格比對
   - 在"表格比對結果"標籤頁查看比對結果
   - 表格差異會以顏色標記顯示

## 注意事項

1. 免費API可能有使用限制，如果遇到問題，可以切換到付費API
2. 表格比對功能對於複雜表格可能需要更長的處理時間
3. 對於大型文件，建議增加處理超時時間

## 技術細節

1. **表格比對算法**
   - 使用單元格對單元格的比對方式
   - 計算表格整體相似度
   - 生成差異矩陣，用於可視化顯示

2. **免費API實現**
   - 使用公開可用的API端點
   - 自動處理API請求格式
   - 解析API響應，提取所需信息

3. **差異可視化**
   - 使用顏色區分添加、刪除和修改的內容
   - 提供字符級別、詞語級別和行級別的差異顯示
   - 表格差異使用顏色標記，直觀顯示不同的單元格
