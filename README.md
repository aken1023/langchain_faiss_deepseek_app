# LangChain + FAISS + DeepSeek Flask 應用

這是一個基於Flask的知識庫問答應用，使用LangChain框架、FAISS向量數據庫和DeepSeek大語言模型來實現文檔檢索和智能問答功能。

## 功能特點

- 使用DeepSeek大語言模型進行自然語言理解和生成
- 使用FAISS向量數據庫進行高效的相似度搜索
- 支持文本文檔的加載、處理和索引
- 提供簡潔的Web界面進行問答交互
- 支持重新加載索引功能

## 系統要求

- Python 3.8+
- Flask
- LangChain
- FAISS-CPU 或 FAISS-GPU
- HuggingFace Transformers
- DeepInfra API 密鑰

## 安裝步驟

1. 克隆或下載本項目代碼

2. 安裝依賴包

```bash
pip install -r requirements.txt
```

3. 創建`requirements.txt`文件，內容如下：

```
flask==2.0.1
langchain==0.0.267
faiss-cpu==1.7.4  # 如果有GPU，可以安裝faiss-gpu
huggingface-hub==0.16.4
transformers==4.33.2
sentence-transformers==2.2.2
python-dotenv==1.0.0
```

## 配置

1. 創建`.env`文件，設置以下環境變量：

```
DEEPINFRA_API_TOKEN=your_deepinfra_api_token
DOCS_DIR=./docs
INDEX_PATH=./faiss_index
```

2. 在`docs`目錄中放入你的文本文檔（.txt格式）

## 使用方法

1. 啟動Flask應用：

```bash
python langchain_faiss_deepseek_app.py
```

2. 在瀏覽器中訪問：`http://localhost:5000`

3. 在界面中輸入問題並提交

4. 如需重新加載索引（例如添加新文檔後），點擊「重新加載索引」按鈕

## 目錄結構

```
.
├── langchain_faiss_deepseek_app.py  # 主應用程序
├── templates/                       # HTML模板
│   └── index.html                   # 前端界面
├── docs/                            # 文檔目錄（放置.txt文件）
├── faiss_index/                     # FAISS索引存儲目錄
├── requirements.txt                 # 依賴包列表
└── .env                             # 環境變量配置
```

## 自定義配置

### 修改嵌入模型

可以在代碼中修改嵌入模型：

```python
# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
```

可以替換為其他支持中文的嵌入模型，如`paraphrase-multilingual-MiniLM-L12-v2`等。

### 修改DeepSeek模型

可以在代碼中修改DeepSeek模型配置：

```python
# 初始化DeepSeek LLM
llm = DeepInfra(
    model_id="deepseek-ai/deepseek-coder-33b-instruct",
    api_key=DEEPSEEK_API_TOKEN,
    model_kwargs={
        "temperature": 0.1,
        "max_tokens": 1024,
        "top_p": 0.9
    }
)
```

## 注意事項

- 首次運行時，系統會自動加載文檔並創建索引，這可能需要一些時間
- 確保DeepInfra API密鑰有效且有足夠的配額
- 對於大量文檔，建議增加系統內存或使用GPU加速

## 故障排除

1. 如果遇到「QA鏈未初始化」錯誤，請檢查文檔目錄是否包含有效的文本文件

2. 如果遇到模型加載錯誤，請檢查網絡連接和API密鑰

3. 如果索引創建失敗，可能是內存不足，嘗試減少文檔數量或增加系統內存