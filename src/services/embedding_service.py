from fastembed.embedding import FlagEmbedding

# Load embedding model once
embedder = FlagEmbedding("BAAI/bge-small-en")

def embed_text(text):
    """
    Returns embedding vector for a given text
    """
    embedding = embedder.embed([text])
    return list(embedding)[0]
