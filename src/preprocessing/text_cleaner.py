import re

def clean_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text