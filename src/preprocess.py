import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pdfplumber
from docx import Document

# -------------------------------------------------
# Ensure ALL required NLTK resources (IMPORTANT)
# -------------------------------------------------
def ensure_nltk_resources():
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }

    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

ensure_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))

# -------------------------------------------------
# Resume Text Extraction
# -------------------------------------------------
def extract_text(file_path: str) -> str:
    ext = file_path.split(".")[-1].lower()
    text = ""

    if ext == "pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + " "

    elif ext == "docx":
        doc = Document(file_path)
        text = " ".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX")

    return text.strip()


# -------------------------------------------------
# Text Preprocessing (SAFE)
# -------------------------------------------------
def preprocess_text(text: str) -> str:
    if not text:
        return ""

    # Normalize
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenize (requires punkt + punkt_tab)
    tokens = word_tokenize(text)

    # Clean tokens
    tokens = [
        token for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    return " ".join(tokens)
