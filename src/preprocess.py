import os
import re

import pdfplumber
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


def extract_text(filepath: str) -> str:
    """
    Extract raw text from PDF or DOCX file.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        text = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
        return "\n".join(text)

    elif ext == '.docx':
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        raise ValueError("Unsupported file type")


def strip_identity_info(text: str) -> str:
    """
    Remove obvious identity clues before scoring:
    - Emails
    - Phone numbers
    - Header lines like 'Name:', 'Email:'
    - All-caps name-like lines (rough heuristic)

    Goal: reduce bias by not exposing direct personal identifiers to the model
    during first-pass screening. [web:149][web:152]
    """
    t = text

    # Remove email addresses
    t = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', t)

    # Remove phone numbers (international + local patterns)
    t = re.sub(r'\+?\d[\d\-\s\(\)]{7,}\d', ' ', t)

    # Remove common header labels and following content on the same line
    t = re.sub(r'(?i)(name|address|phone|email)\s*[:\-].*', ' ', t)

    # Remove lines that are likely just a full name in ALL CAPS (very rough)
    cleaned_lines = []
    for line in t.splitlines():
        stripped = line.strip()
        if stripped.isupper() and 2 <= len(stripped.split()) <= 4:
            # Skip this line (likely a name header)
            continue
        cleaned_lines.append(line)
    t = "\n".join(cleaned_lines)

    return t


def preprocess_text(text: str) -> str:
    """
    Basic NLP preprocessing:
    - Lowercase
    - Remove non-alphanumeric characters
    - Tokenize
    - Remove stopwords
    """
    # Lowercase
    text = text.lower()

    # Replace non-alphanumeric with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    return " ".join(tokens)
