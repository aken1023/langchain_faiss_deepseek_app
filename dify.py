import requests
import json

class DifyAPI:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def test_connection(self):
        try:
            # 測試基本的消息發送功能
            endpoint = f'{self.base_url}/v1/chat-messages'
            
            # 測試命令列表
            test_commands = [
                '你好',
                '如何申請補助？',
                '請說明申請流程'
            ]
            
            for command in test_commands:
                print(f'\n測試命令：{command}')
                payload = {
                    'inputs': {},
                    'query': command,
                    'response_mode': 'streaming',
                    'conversation_id': '',
                    'user': 'test_user'
                }
                
                response = requests.post(endpoint, headers=self.headers, json=payload, stream=True)
                response.raise_for_status()
                
                # 讀取回應
                got_response = False
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str.strip() != '[DONE]':
                                try:
                                    data = json.loads(data_str)
                                    if 'answer' in data:
                                        print('回答：', data['answer'])
                                        got_response = True
                                        break
                                except json.JSONDecodeError:
                                    continue
                
                if not got_response:
                    print(f'測試命令 "{command}" 未收到回應')
                    return False
            
            print('\n所有連接測試成功!')
            return True
            
        except requests.exceptions.RequestException as e:
            print('連接測試失敗!')
            print('錯誤信息:', str(e))
            return False

def main():
    # 設置 API 參數
    api_key = 'app-fzs50rbMxZtDJSx5M56fwlz2'
    base_url = 'http://122.100.99.161:8080'
    
    # 創建 DifyAPI 實例
    dify = DifyAPI(api_key, base_url)
    
    # 執行連接測試
    dify.test_connection()

if __name__ == '__main__':
    main()