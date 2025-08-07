import requests

url = "http://127.0.0.1:8000/hackrx/run"

headers = {
    "Authorization": "Bearer hackrx2025",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "Does this policy cover maternity?",
        "What is the waiting period for cataract surgery?"
    ]
}

response = requests.post(url, headers=headers, json=data)

print("✅ Status Code:", response.status_code)

answers = response.json().get("answers", [])
print("🧾 Answers:")
for i, ans in enumerate(answers, 1):
    print(f"{i}. {ans}")
