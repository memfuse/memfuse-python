import requests

try:
    response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Connection error: {e}") 