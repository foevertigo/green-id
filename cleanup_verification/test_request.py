import requests

url = "http://127.0.0.1:5000/cleanup-check"  


files = {
    "before": open("test_images/before.png", "rb"),
    "after": open("test_images/after.png", "rb")
}

response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())
