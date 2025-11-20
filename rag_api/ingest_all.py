import os
import requests

DATA_DIR = "data"
API_URL = "http://127.0.0.1:8000/api/ingest-upload/"

for file_name in os.listdir(DATA_DIR):
    if file_name.lower().endswith(".pdf"):
        file_path = os.path.join(DATA_DIR, file_name)
        print(f"Uploading {file_name}...")
        with open(file_path, "rb") as f:
            r = requests.post(API_URL, files={"file": (file_name, f)})
            print(r.json())
    else:
        print(f"Skipping {file_name}, not a PDF.")