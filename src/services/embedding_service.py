import re
import string
from sentence_transformers import SentenceTransformer

# Load once at startup
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text.strip()


def embed_text(text: str):
    return embed_model.encode(text, normalize_embeddings=True)
