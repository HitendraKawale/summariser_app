import re
import pdfplumber
import requests
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    # collapse whitespace and remove obviously noisy runs
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_pdf(file_obj) -> str:
    text = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text.append(t)
    return clean_text(" ".join(text))


def extract_text_from_url(url: str) -> str:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    # remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return clean_text(soup.get_text(separator=" "))


def chunk_text_words(text: str, max_words: int = 600):
    """
    Naive word-based chunking so we don't overflow model token limits.
    BART can handle ~1024 tokens; ~600-700 words per chunk is a safe default.
    """
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i : i + max_words])
