#!/usr/bin/env python3
from flask import (
    Flask, request, render_template,
    send_file, jsonify, redirect, url_for, session
)
import os, re, json, joblib, sqlite3
import numpy as np
import PyPDF2
import spacy
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import spacy
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ================= APP SETUP =================
app = Flask(__name__)
app.secret_key = "change-this-secret-key"

UPLOAD = "uploads"
DB = "users.db"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs("models", exist_ok=True)

nlp = spacy.load("en_core_web_sm")

# ================= JOB DESCRIPTION TEMPLATES =================
JOB_TEMPLATES = {
    "python_dev": "Python Developer with Flask, Django, REST APIs and SQL.",
    "data_scientist": "Data Scientist skilled in Python, Pandas, NumPy, ML, NLP.",
    "ml_engineer": "ML Engineer with model training, deployment, NLP.",
    "frontend_dev": "Frontend Developer with HTML, CSS, JavaScript, React, UI/UX.",
    "backend_dev": "Backend Developer with APIs, databases, authentication."
}

# ================= DATABASE =================
def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'hr'
        )
        """)

init_db()

# ================= HELPERS =================
def clean_text(text):
    text = re.sub(r"\n+", " ", text.lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    doc = nlp(text)
    return " ".join(t.lemma_ for t in doc if not t.is_stop)

def extract_keywords(text):
    doc = nlp(text.lower())
    return set(t.text for t in doc if t.is_alpha and not t.is_stop)

def register_user(username, password, role=None):
    role = role if role in ["admin", "hr"] else "hr"
    hashed = generate_password_hash(password)
    try:
        with sqlite3.connect(DB) as conn:
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed, role)
            )
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username, password):
    with sqlite3.connect(DB) as conn:
        row = conn.execute(
            "SELECT password, role FROM users WHERE username=?",
            (username,)
        ).fetchone()
    if row and check_password_hash(row[0], password):
        return row[1]
    return None

def login_required():
    return "user" in session

# ================= ATS DECISION LOGIC =================

PASS_THRESHOLD = 0.70      # overall
MATCH_THRESHOLD = 0.60     # job relevance
QUALITY_THRESHOLD = 0.65   # resume quality

def ats_decision(final, quality, match):
    if final >= PASS_THRESHOLD and match >= MATCH_THRESHOLD:
        return "PASS"
    return "FAIL"

def decision_reason(decision, quality, match):
    """Generate human-readable reason for the decision"""
    if decision == "PASS":
        return "The candidate's resume meets the required standards for this position."
    else:
        reasons = []
        if quality < QUALITY_THRESHOLD:
            reasons.append("resume quality")
        if match < MATCH_THRESHOLD:
            reasons.append("job relevance")
        
        if reasons:
            return f"The candidate did not meet requirements in: {', '.join(reasons)}."
        return "The overall score did not meet the hiring benchmark."

def calculate_confidence(final, quality, match):
    """Calculate confidence score based on how far from thresholds"""
    # Higher confidence when scores are far from decision boundaries
    quality_distance = abs(quality - QUALITY_THRESHOLD)
    match_distance = abs(match - MATCH_THRESHOLD)
    final_distance = abs(final - PASS_THRESHOLD)
    
    # Average distance from thresholds (normalized)
    confidence = (quality_distance + match_distance + final_distance) / 3
    # Scale to 0.5-1.0 range (minimum 50% confidence)
    confidence = min(0.5 + confidence, 1.0)
    
    return confidence

def failure_reasons(quality, match):
    reasons = []

    if quality < QUALITY_THRESHOLD:
        reasons.append("Resume quality is below the expected standard")

    if match < MATCH_THRESHOLD:
        reasons.append("Job description relevance is low")

    if not reasons:
        reasons.append("Overall score did not meet hiring benchmark")

    return reasons

def skill_gap_analysis(resume_text, job_desc):
    resume_skills = extract_keywords(resume_text)
    job_skills = extract_keywords(job_desc)

    missing = sorted(list(job_skills - resume_skills))
    matched = sorted(list(job_skills & resume_skills))

    return matched[:10], missing[:10]

def improvement_tips(missing_skills):
    tips = []
    for skill in missing_skills[:5]:
        tips.append(f"Add hands-on experience or projects related to '{skill}'")

    if not tips:
        tips.append("Resume is strong — consider optimizing formatting and clarity")

    return tips

def ai_explanation(final, quality, match, decision):
    if decision == "PASS":
        return (
            "The resume demonstrates strong alignment with the job role, "
            "showing both relevant skills and acceptable resume quality."
        )
    return (
        "The resume does not sufficiently match the job requirements. "
        "Improving relevance and skill alignment can significantly increase chances."
    )


# ================= LOAD MODELS =================
try:
    tfidf_general = joblib.load("models/tfidf_general.pkl")
    model_general = joblib.load("models/model_general.pkl")
    tfidf_match = joblib.load("models/tfidf_match.pkl")
    model_match = joblib.load("models/model_match.pkl")
    skills_list = json.load(open("models/skills.json"))
    MODELS = True
except Exception as e:
    print("⚠️ Models not loaded:", e)
    MODELS = False

# ================= AUTH ROUTES =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = authenticate_user(
            request.form["username"],
            request.form["password"]
        )
        if role:
            session["user"] = request.form["username"]
            session["role"] = role
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        if register_user(
            request.form["username"],
            request.form["password"],
            request.form.get("role")
        ):
            return redirect(url_for("login"))
        return render_template("register.html", error="User already exists")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/clear")
def clear():
    """Clear results and start new analysis"""
    if 'last_result' in session:
        del session['last_result']
    return redirect(url_for("index"))

# ================= RESULTS ROUTE =================
@app.route("/results")
def results():
    if not login_required():
        return redirect(url_for("login"))
    
    result_data = session.get('last_result')
    if not result_data:
        return redirect(url_for("index"))
    
    return render_template(
        "index.html",
        job_templates=JOB_TEMPLATES,
        **result_data
    )

# ================= MAIN APP =================
@app.route("/", methods=["GET", "POST"])
def index():
    if not login_required():
        return redirect(url_for("login"))

    if request.method == "POST":
        job_desc = request.form.get("job_desc", "")
        job_role = request.form.get("job_role")

        if job_role in JOB_TEMPLATES and not job_desc.strip():
            job_desc = JOB_TEMPLATES[job_role]

        file = request.files.get("resume")
        if not file or not file.filename.endswith(".pdf"):
            return render_template(
                "index.html",
                error="Upload PDF only",
                job_templates=JOB_TEMPLATES
            )

        path = os.path.join(UPLOAD, file.filename)
        file.save(path)

        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                if p.extract_text():
                    text += p.extract_text()

        if not text.strip():
            return render_template(
                "index.html",
                error="Scanned PDF not supported",
                job_templates=JOB_TEMPLATES
            )

        clean_resume = clean_text(text)
        clean_jd = clean_text(job_desc)

        # 🔒 SAFETY CHECK
        if not MODELS:
            return render_template(
                "index.html",
                error="⚠️ ML models not trained. Run train_model.py first.",
                job_templates=JOB_TEMPLATES
            )

        # ================= ML SCORING =================
        q = model_general.predict(
            tfidf_general.transform([clean_resume])
        )[0]

        m = model_match.predict(
            tfidf_match.transform([clean_resume + " " + clean_jd])
        )[0]

        final = 0.6 * q + 0.4 * m
        
        # ================= ENHANCED FEATURES =================
        # Calculate keyword-based metrics
        resume_keywords = extract_keywords(text)
        jd_keywords = extract_keywords(job_desc)
        common_skills = resume_keywords.intersection(jd_keywords)
        jd_coverage = round(len(common_skills) / max(len(jd_keywords), 1) * 100, 2)
        
        # Determine status with configurable threshold
        PASS_THRESHOLD_CONFIG = 0.65
        status = "PASS" if final >= PASS_THRESHOLD_CONFIG else "FAIL"
        
        # Enhanced failure reasons with detailed criteria
        enhanced_failure_reasons = []
        if q < 0.5:
            enhanced_failure_reasons.append("Low resume quality (format, clarity, or structure issues)")
        if m < 0.6:
            enhanced_failure_reasons.append("Poor alignment with the job description")
        if jd_coverage < 50:
            enhanced_failure_reasons.append("More than half of the required skills are missing")
        if len(common_skills) < 5:
            enhanced_failure_reasons.append("Insufficient relevant skills detected for this role")
        if status == "PASS":
            enhanced_failure_reasons = []
        
        # Calculate hiring risk
        if final >= 0.8 and jd_coverage >= 70:
            hiring_risk = "LOW"
        elif final >= 0.6 and jd_coverage >= 50:
            hiring_risk = "MEDIUM"
        else:
            hiring_risk = "HIGH"
        
        # Explainability module for AI transparency
        explainability = {
            "quality_explanation": (
                "Resume shows strong structure and clarity"
                if q >= 0.7 else
                "Resume structure or clarity needs improvement"
            ),
            "match_explanation": (
                "Resume aligns well with the job role"
                if m >= 0.7 else
                "Resume does not sufficiently match the job requirements"
            ),
            "coverage_explanation": f"{jd_coverage}% of job description skills were found in the resume",
            "decision_explanation": (
                "Candidate meets the minimum criteria for this role"
                if status == "PASS"
                else "Candidate does not meet the minimum criteria for this role"
            )
        }
        
        # Keep existing features
        decision = ats_decision(final, q, m)
        reasons = failure_reasons(q, m)
        matched_skills, missing_skills = skill_gap_analysis(text, job_desc)
        recommendations = improvement_tips(missing_skills)
        explanation = ai_explanation(final, q, m, decision)
        confidence = calculate_confidence(final, q, m)
        decision_text = decision_reason(decision, q, m)

        resume_vec = tfidf_general.transform([clean_resume])
        skills_vec = tfidf_general.transform(skills_list)
        sims = cosine_similarity(resume_vec, skills_vec)[0]

        top_skills = [
            {"name": skills_list[i], "score": float(sims[i])}
            for i in np.argsort(sims)[-8:][::-1]
        ]

        # ================= HEATMAP =================
        heatmap_words = list(
            extract_keywords(text)
            .intersection(extract_keywords(job_desc))
        )

        # ================= PDF REPORT =================
        report = os.path.join(UPLOAD, "report.pdf")
        doc = SimpleDocTemplate(report, pagesize=letter)
        styles = getSampleStyleSheet()
        doc.build([
            Paragraph("AI Resume Screening Report", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"Decision: {decision}", styles["Normal"]),
            Paragraph(f"Status: {status}", styles["Normal"]),
            Paragraph(f"Confidence: {confidence:.1%}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph(f"Final Score: {final:.1%}", styles["Normal"]),
            Paragraph(f"Quality Score: {q:.1%}", styles["Normal"]),
            Paragraph(f"Job Match Score: {m:.1%}", styles["Normal"]),
            Paragraph(f"JD Coverage: {jd_coverage}%", styles["Normal"]),
            Paragraph(f"Hiring Risk: {hiring_risk}", styles["Normal"]),
        ])

        os.remove(path)

        # Store results in session to enable proper redirect
        session['last_result'] = {
            'success': True,
            'score': final,
            'quality': q,
            'match': m,
            'decision': decision,
            'decision_reason': decision_text,
            'confidence': confidence,
            'reasons': reasons,
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'recommendations': recommendations,
            'explanation': explanation,
            'top_skills': top_skills,
            'report': "report.pdf",
            'preview': text[:500] + "...",
            'heatmap': heatmap_words,
            # Enhanced features
            'status': status,
            'jd_coverage': jd_coverage,
            'hiring_risk': hiring_risk,
            'failure_reasons_enhanced': enhanced_failure_reasons,
            'explainability': explainability
        }
        
        return redirect(url_for('results'))

    return render_template(
        "index.html",
        job_templates=JOB_TEMPLATES
    )

# ================= UTIL =================
@app.route("/download/<f>")
def download(f):
    if not login_required():
        return redirect(url_for("login"))
    return send_file(os.path.join(UPLOAD, f), as_attachment=True)

@app.route("/api/health")
def health():
    return jsonify({
        "models_loaded": MODELS,
        "auth": "enabled"
    })

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


