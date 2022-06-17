import re
import string


def preprocess_text(text: str) -> str:
    text = re.sub(r'\n', ' ', text)
    text = text.strip()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text
