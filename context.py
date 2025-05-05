import requests

api_key = 'app-D6U65NfvxvbeACL2aLQXJNb0'
base_url = 'http://122.100.99.161:8080'

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "inputs": {},
    "query": "請問補助案的條件有哪些？",
    "response_mode": "blocking",
    "user": "user-001"
}

response = requests.post(f"{base_url}/v1/completion-messages", json=payload, headers=headers)
print(response.status_code)
print(response.json())
