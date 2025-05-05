import requests

base_url = "http://122.100.99.161:8080"  # 原始基础URL
api_key = "app-D6U65NfvxvbeACL2aLQXJNb0"  # 按照要求使用指定的API密钥

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def test_api_endpoint(endpoint, method='GET', payload=None, description=""):
    """通用API端點測試函數"""
    url = f"{base_url}{endpoint}"
    print(f"\n===== 測試 {description} ({url}) =====")
    
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=payload)
        elif method.upper() == 'OPTIONS':
            response = requests.options(url, headers=headers)
        else:
            print(f"不支持的HTTP方法: {method}")
            return
            
        print(f"狀態碼: {response.status_code}")
        
        print("\n響應摘要:")
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"響應類型: {type(data)}")
                print(f"響應包含的鍵: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
                
                # 顯示回答內容（如果存在）
                if 'answer' in data:
                    answer = data['answer']
                    print(f"\nAI回答: {answer[:100]}..." if len(answer) > 100 else f"\nAI回答: {answer}")
                elif 'text' in data:
                    text = data['text']
                    print(f"\n生成文本: {text[:100]}..." if len(text) > 100 else f"\n生成文本: {text}")
                
                # 顯示其他有用的元數據
                if 'metadata' in data and data['metadata']:
                    print(f"\n元數據: {data['metadata']}")
            except ValueError:
                print(f"響應不是JSON格式: {response.text[:200]}..." if len(response.text) > 200 else f"響應不是JSON格式: {response.text}")
        else:
            print(f"錯誤響應: {response.text[:200]}..." if len(response.text) > 200 else f"錯誤響應: {response.text}")
    except Exception as e:
        print(f"發生錯誤: {str(e)}")

def test_chat_message():
    """測試聊天型App對話呼叫"""
    payload = {
        'inputs': {},
        'query': '你好',
        'response_mode': 'blocking',
        'conversation_id': '',
        'user': 'test_user'
    }
    test_api_endpoint('/v1/chat-messages', 'POST', payload, "聊天型App對話呼叫")

def test_completion_message():
    """測試文字生成型App使用"""
    payload = {
        'inputs': {},
        'query': '你好',
        'response_mode': 'blocking',
        'user': 'test_user'
    }
    test_api_endpoint('/v1/completion-messages', 'POST', payload, "文字生成型App使用")

def test_models():
    """測試查詢目前支援的LLM模型"""
    test_api_endpoint('/v1/models', 'GET', None, "查詢目前支援的LLM模型")

def test_parameters():
    """測試查詢自訂欄位設定"""
    test_api_endpoint('/v1/parameters', 'GET', None, "查詢自訂欄位設定")

if __name__ == "__main__":
    print(f"\n======== 使用API密鑰: {api_key} 進行測試 ========\n")
    
    # 測試所有端點
    test_chat_message()
    test_completion_message()
    test_models()
    test_parameters()
    
    # 文件上傳端點需要額外的處理，這裡只是示範
    print("\n===== 測試 上傳文件給知識庫使用 (/v1/files/upload) =====\n")
    print("注意: 文件上傳需要額外的multipart/form-data處理，此處僅作說明")
    print("如需實際測試，請參考Dify API文檔實現文件上傳功能")
