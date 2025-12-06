import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in stop_words]
    return " ".join(tokens)
