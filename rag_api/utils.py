import os
import json
import PyPDF2

def read_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def split_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def save_metadata(metadata, vector_dir):
    with open(os.path.join(vector_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

def load_metadata(vector_dir):
    path = os.path.join(vector_dir, "metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []
def allowed_file(filename: str, allowed_extensions: set) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions