import os
import shutil
import pdfplumber
from docx import Document

BASE_TMP = "/tmp"


def create_session_folder(session_id: str) -> str:
    folder = os.path.join(BASE_TMP, session_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def save_uploaded_file(upload_file, folder: str) -> str:
    path = os.path.join(folder, upload_file.filename)
    with open(path, "wb") as f:
        f.write(upload_file.file.read())
    return path


def cleanup_session(session_path: str):
    if os.path.exists(session_path):
        shutil.rmtree(session_path, ignore_errors=True)


def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
