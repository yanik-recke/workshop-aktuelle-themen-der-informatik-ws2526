import requests
try:
    r = requests.get('http://localhost:1234/v1/models', timeout=5)
    data = r.json()
    print('Available models in LM Studio:')
    for m in data.get('data', []):
        print(f"  - {m['id']}")
except Exception as e:
    print(f"Error: {e}")
