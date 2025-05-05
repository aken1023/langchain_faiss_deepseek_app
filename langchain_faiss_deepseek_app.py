import os
import sys
from typing import List, Dict, Any
import dotenv
import torch

# 加載.env文件
dotenv.load_dotenv()

# 檢測GPU資源
def check_gpu_availability():
    """檢測是否有可用的GPU資源"""
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0)
        }
        print(f"檢測到GPU資源: {gpu_info['device_name']}")
        return gpu_info
    else:
        print("未檢測到GPU資源，將使用CPU運算")
        return {"available": False}

# 檢測GPU可用性
GPU_INFO = check_gpu_availability()

# 檢測FAISS是否支持GPU
def check_faiss_gpu():
    """檢測FAISS是否支持GPU"""
    try:
        import faiss
        has_gpu_support = hasattr(faiss, 'GpuIndexFlatL2')
        if has_gpu_support and GPU_INFO['available']:
            print(f"FAISS GPU支持已啟用，將使用GPU加速向量檢索")
            return True
        else:
            if not has_gpu_support:
                print("FAISS未檢測到GPU支持，將使用CPU模式")
            return False
    except ImportError:
        print("未能導入FAISS庫，請確保已正確安裝")
        return False

# 檢測FAISS GPU支持
FAISS_GPU_AVAILABLE = check_faiss_gpu()

# 檢查環境變量
if not os.environ.get("OPENAI_API_KEY"):
    print("警告: OPENAI_API_KEY 環境變量未設置")
    # 如果有直接設置的API密鑰，可以在這裡設置
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, session
import werkzeug.utils
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # 使用OpenAI替代DeepSeek
import tiktoken  # 用於計算token數量

app = Flask(__name__)

# 設置密鑰用於session
app.secret_key = 'your_secret_key_here'

# 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_openai_api_key_here")
DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
INDEX_PATH = os.environ.get("INDEX_PATH", "./faiss_index")

# Token費用配置 (美元/1000 tokens)
TOKEN_PRICES = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03}
}

# 累計費用跟踪
CUMULATIVE_COST = {
    "usd": 0.0,
    "twd": 0.0,
    "total_tokens": 0
}

# 初始化嵌入模型
if GPU_INFO["available"]:
    # 使用GPU進行嵌入計算
    embeddings = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-base",
        model_kwargs={"device": f"cuda:{GPU_INFO['current_device']}"}
    )
    print(f"嵌入模型將使用GPU: {GPU_INFO['device_name']}")
else:
    # 使用CPU進行嵌入計算
    embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    print("嵌入模型將使用CPU")

# 初始化OpenAI LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,  # 使用OpenAI API密鑰
    model_name="gpt-3.5-turbo",  # 使用OpenAI的模型
    temperature=0.1,
    max_tokens=1024
)

# 文檔加載和處理
def load_documents(directory: str) -> List[Any]:
    """從指定目錄加載文檔"""
    try:
        # 導入PDF加載器
        from langchain_community.document_loaders import PyPDFLoader
        
        # 加載TXT文件
        txt_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
        txt_docs = txt_loader.load()
        
        # 加載PDF文件
        pdf_loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        
        # 合併文檔
        documents = txt_docs + pdf_docs
        print(f"成功加載 {len(documents)} 個文檔 (TXT: {len(txt_docs)}, PDF: {len(pdf_docs)})")
        return documents
    except Exception as e:
        print(f"加載文檔時出錯: {e}")
        return []

def process_documents(documents: List[Any]) -> List[Any]:
    """處理文檔，分割成較小的塊"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文檔已分割為 {len(chunks)} 個塊")
    return chunks

# 全局變量用於跟踪索引進度
indexing_status = {
    "in_progress": False,
    "total_documents": 0,
    "processed_documents": 0,
    "total_chunks": 0,
    "current_stage": "",
    "error": None
}

# 創建或加載向量存儲
def get_vector_store():
    """創建或加載FAISS向量存儲"""
    global indexing_status
    
    if os.path.exists(INDEX_PATH):
        try:
            vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print(f"成功從 {INDEX_PATH} 加載向量存儲")
            indexing_status = {
                "in_progress": False,
                "total_documents": 0,
                "processed_documents": 0,
                "total_chunks": 0,
                "current_stage": "完成",
                "error": None
            }
            return vector_store
        except Exception as e:
            error_msg = f"加載向量存儲時出錯: {e}"
            print(error_msg)
            indexing_status["error"] = error_msg
    
    # 如果加載失敗或索引不存在，創建新的向量存儲
    indexing_status = {
        "in_progress": True,
        "total_documents": 0,
        "processed_documents": 0,
        "total_chunks": 0,
        "current_stage": "開始加載文檔",
        "error": None
    }
    print("創建新的向量存儲...")
    
    try:
        # 加載文檔
        documents = load_documents(DOCS_DIR)
        if not documents:
            error_msg = "沒有找到文檔，無法創建向量存儲"
            print(error_msg)
            indexing_status = {
                "in_progress": False,
                "current_stage": "失敗",
                "error": error_msg
            }
            return None
        
        indexing_status["total_documents"] = len(documents)
        indexing_status["processed_documents"] = len(documents)
        indexing_status["current_stage"] = "文檔分割中"
        
        # 處理文檔
        chunks = process_documents(documents)
        indexing_status["total_chunks"] = len(chunks)
        indexing_status["current_stage"] = "創建向量索引中"
        
        # 創建向量存儲
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # 保存向量存儲
        indexing_status["current_stage"] = "保存索引中"
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        vector_store.save_local(INDEX_PATH)
        print(f"向量存儲已保存到 {INDEX_PATH}")
        
        indexing_status["in_progress"] = False
        indexing_status["current_stage"] = "完成"
        
        return vector_store
    except Exception as e:
        error_msg = f"創建向量存儲時出錯: {e}"
        print(error_msg)
        indexing_status = {
            "in_progress": False,
            "current_stage": "失敗",
            "error": error_msg
        }
        return None

# 初始化QA鏈
def initialize_qa_chain():
    """初始化問答鏈"""
    vector_store = get_vector_store()
    if vector_store is None:
        return None
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

# 全局QA鏈
qa_chain = None

# 在Flask 2.0+中，@app.before_first_request已被棄用
# 改為使用with app.app_context():
def setup_qa_chain():
    """設置QA鏈"""
    global qa_chain
    if qa_chain is None:
        qa_chain = initialize_qa_chain()

@app.route('/login', methods=['GET'])
def login():
    """登入頁面"""
    if session.get('logged_in'):
        return redirect('/')
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    """處理登入請求"""
    password = request.json.get('password')
    if password == '1111':
        session['logged_in'] = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': '密碼錯誤'}), 401

@app.route('/logout')
def logout():
    """登出"""
    session.pop('logged_in', None)
    return redirect('/login')

@app.route('/')
def index():
    """主頁"""
    if not session.get('logged_in'):
        return redirect('/login')
    setup_qa_chain()
    return render_template('index.html')

@app.route('/admin')
def admin():
    """管理員頁面"""
    if not session.get('logged_in'):
        return redirect('/login')
    return render_template('admin.html')

@app.route('/api/admin/upload', methods=['POST'])
def upload_files():
    """處理文件上傳"""
    if 'files' not in request.files:
        return jsonify({
            "success": False,
            "error": "沒有找到文件"
        })
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({
            "success": False,
            "error": "沒有選擇文件"
        })
    
    try:
        for file in files:
            filename = werkzeug.utils.secure_filename(file.filename)
            file_path = os.path.join(DOCS_DIR, filename)
            file.save(file_path)
            print(f"文件已保存: {file_path}")
        
        return jsonify({
            "success": True,
            "message": "文件上傳成功"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"上傳文件時出錯: {str(e)}"
        })

@app.route('/api/admin/files', methods=['GET'])
def list_files():
    """獲取文件列表"""
    try:
        files = []
        for filename in os.listdir(DOCS_DIR):
            if os.path.isfile(os.path.join(DOCS_DIR, filename)):
                files.append({
                    "name": filename,
                    "path": os.path.join(DOCS_DIR, filename)
                })
        
        return jsonify({
            "success": True,
            "files": files
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"獲取文件列表時出錯: {str(e)}"
        })

@app.route('/api/admin/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """刪除文件"""
    try:
        file_path = os.path.join(DOCS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({
                "success": True,
                "message": f"文件 {filename} 已刪除"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"文件 {filename} 不存在"
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"刪除文件時出錯: {str(e)}"
        })

@app.route('/api/query', methods=['POST'])
def query():
    """處理查詢API"""
    global qa_chain
    setup_qa_chain()
    
    if qa_chain is None:
        return jsonify({
            "error": "QA鏈未初始化，請確保文檔已正確加載"
        }), 500
    
    data = request.json
    if not data or 'query' not in data:
        return jsonify({
            "error": "請提供查詢內容"
        }), 400
    
    query_text = data['query']
    
    try:
        # 執行查詢
        result = qa_chain.invoke({"query": query_text})
        
        # 提取來源文檔
        source_documents = []
        if 'source_documents' in result:
            for doc in result['source_documents']:
                source_documents.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "未知")
                })
        
        # 計算token使用量和費用
        answer = result.get("result", "")
        token_info = calculate_token_usage(query_text, answer, source_documents)
        
        return jsonify({
            "answer": answer,
            "source_documents": source_documents,
            "token_info": token_info
        })
    except Exception as e:
        return jsonify({
            "error": f"處理查詢時出錯: {str(e)}"
        }), 500

@app.route('/api/indexing-status', methods=['GET'])
def get_indexing_status():
    """獲取索引進度"""
    global indexing_status
    return jsonify(indexing_status)

@app.route('/api/cost-stats', methods=['GET'])
def get_cost_stats():
    """獲取累計費用統計"""
    global CUMULATIVE_COST
    return jsonify({
        "cumulative_tokens": CUMULATIVE_COST["total_tokens"],
        "cumulative_cost_usd": round(CUMULATIVE_COST["usd"], 6),
        "cumulative_cost_twd": round(CUMULATIVE_COST["twd"], 2)
    })

@app.route('/api/reset-cost', methods=['POST'])
def reset_cost_stats():
    """重置累計費用統計"""
    global CUMULATIVE_COST
    CUMULATIVE_COST = {
        "usd": 0.0,
        "twd": 0.0,
        "total_tokens": 0
    }
    return jsonify({
        "success": True,
        "message": "累計費用已重置"
    })

@app.route('/api/reload', methods=['POST'])
def reload_index():
    """重新加載索引"""
    global qa_chain
    global indexing_status
    setup_qa_chain()
    
    try:
        # 刪除現有索引
        if os.path.exists(INDEX_PATH):
            import shutil
            shutil.rmtree(INDEX_PATH)
            print(f"已刪除現有索引: {INDEX_PATH}")
        
        # 重新初始化QA鏈
        qa_chain = initialize_qa_chain()
        
        if qa_chain is None:
            return jsonify({
                "error": "重新加載索引失敗，請檢查文檔目錄"
            }), 500
        
        return jsonify({
            "message": "索引已成功重新加載"
        })
    except Exception as e:
        return jsonify({
            "error": f"重新加載索引時出錯: {str(e)}"
        }), 500

# 計算token使用量和費用
def calculate_token_usage(query, answer, source_documents):
    """計算token使用量和費用"""
    global CUMULATIVE_COST
    try:
        # 獲取當前使用的模型
        model_name = llm.model_name
        if model_name not in TOKEN_PRICES:
            model_name = "gpt-3.5-turbo"  # 默認使用gpt-3.5-turbo的價格
        
        # 獲取編碼器
        encoding = tiktoken.encoding_for_model(model_name)
        
        # 計算輸入tokens (查詢 + 文檔內容)
        input_text = query
        for doc in source_documents:
            input_text += doc["content"]
        
        input_tokens = len(encoding.encode(input_text))
        
        # 計算輸出tokens (回答)
        output_tokens = len(encoding.encode(answer))
        
        # 計算費用 (美元)
        input_cost = (input_tokens / 1000) * TOKEN_PRICES[model_name]["input"]
        output_cost = (output_tokens / 1000) * TOKEN_PRICES[model_name]["output"]
        total_cost = input_cost + output_cost
        
        # 轉換為台幣 (假設1美元=30台幣)
        twd_cost = total_cost * 30
        
        # 更新累計費用
        total_tokens = input_tokens + output_tokens
        CUMULATIVE_COST["usd"] += total_cost
        CUMULATIVE_COST["twd"] += twd_cost
        CUMULATIVE_COST["total_tokens"] += total_tokens
        
        # 獲取GPU信息
        gpu_info = None
        if GPU_INFO["available"]:
            gpu_info = {
                "name": GPU_INFO["device_name"],
                "count": GPU_INFO["device_count"]
            }
        
        return {
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(total_cost, 6),
            "cost_twd": round(twd_cost, 2),
            "cumulative_cost_usd": round(CUMULATIVE_COST["usd"], 6),
            "cumulative_cost_twd": round(CUMULATIVE_COST["twd"], 2),
            "cumulative_tokens": CUMULATIVE_COST["total_tokens"],
            "gpu_info": gpu_info
        }
    except Exception as e:
        print(f"計算token使用量時出錯: {e}")
        return {
            "model": llm.model_name,
            "error": str(e),
            "gpu_info": GPU_INFO["available"] and {
                "name": GPU_INFO["device_name"],
                "count": GPU_INFO["device_count"]
            } or None
        }

if __name__ == '__main__':
    # 確保目錄存在
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    # 初始化QA鏈
    with app.app_context():
        qa_chain = initialize_qa_chain()
    
    # 啟動Flask應用
    app.run(debug=True, host='0.0.0.0', port=5007)