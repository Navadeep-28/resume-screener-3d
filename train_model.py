#!/usr/bin/env python3
"""
✅ ResumeScreener Training (FIXED)
"""

import requests
import pandas as pd
import numpy as np
import re, os, json, joblib
import spacy
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("models", exist_ok=True)
nlp = spacy.load("en_core_web_sm")

print("🚀 Downloading dataset...")
url = "https://raw.githubusercontent.com/anukalp-mishra/Resume-Screening/main/resume_dataset.csv"
df = pd.read_csv(StringIO(requests.get(url).text))

job_descs = {
    "Data Science": "python pandas machine learning data science nlp",
    "HR": "recruitment hr talent acquisition screening",
    "Software": "python flask javascript backend frontend developer"
}
df["jd"] = df["Category"].map(job_descs).fillna("software developer")

def clean_text(text):
    text = re.sub(r"\n+", " ", text.lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    doc = nlp(text)
    return " ".join(t.lemma_ for t in doc if not t.is_stop)

df["clean_resume"] = df["Resume"].apply(clean_text)
df["clean_jd"] = df["jd"].apply(clean_text)

# ---- Quality Model (simulated but consistent) ----
np.random.seed(42)
df["quality"] = np.clip(0.7 + 0.2 * np.random.rand(len(df)), 0.5, 1.0)

tfidf_general = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
Xg = tfidf_general.fit_transform(df["clean_resume"])
yg = df["quality"]

Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(Xg, yg, test_size=0.2, random_state=42)
model_general = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
model_general.fit(Xg_tr, yg_tr)

print("✅ Quality R²:", r2_score(yg_te, model_general.predict(Xg_te)))

joblib.dump(tfidf_general, "models/tfidf_general.pkl")
joblib.dump(model_general, "models/model_general.pkl")

# ---- Job Match Model (FIXED) ----
tfidf_match = TfidfVectorizer(max_features=2000)
tfidf_match.fit(pd.concat([df["clean_resume"], df["clean_jd"]]))

match_scores = [
    cosine_similarity(
        tfidf_match.transform([r]),
        tfidf_match.transform([j])
    )[0][0]
    for r, j in zip(df["clean_resume"], df["clean_jd"])
]


joblib.dump(tfidf_match, "models/tfidf_match.pkl")


# ---- Skills List (Keyword-safe) ----
skills = sorted({
    "python","flask","django","machine learning","nlp","sql",
    "pandas","numpy","scikit-learn","data analysis","javascript",
    "react","api","deep learning","hr","recruitment"
})

with open("models/skills.json", "w") as f:
    json.dump(skills, f)

print("🎉 TRAINING COMPLETE")

