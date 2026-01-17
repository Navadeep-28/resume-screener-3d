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
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from spacy.cli import download
from math import ceil

# ================= NLP =================
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

# =========================================================
# 🔥 CORE RESUME SCORING FUNCTION (REUSABLE)
# =========================================================
def score_resume(resume_text, job_desc):
    """Reusable function to score a resume against a job description"""
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_desc)

    q = model_general.predict(
        tfidf_general.transform([clean_resume])
    )[0]

    resume_vec = tfidf_match.transform([clean_resume])
    jd_vec = tfidf_match.transform([clean_jd])
    m = cosine_similarity(resume_vec, jd_vec)[0][0]

    final = 0.6 * q + 0.4 * m

    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_desc)

    matched = resume_keywords & jd_keywords
    coverage = round(len(matched) / max(len(jd_keywords), 1) * 100, 2)

    return {
        "final": float(final),
        "quality": float(q),
        "match": float(m),
        "coverage": coverage,
        "matched_skills": sorted(list(matched))[:10]
    }

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

        resume_vec = tfidf_match.transform([clean_resume])
        jd_vec = tfidf_match.transform([clean_jd])
        m = cosine_similarity(resume_vec, jd_vec)[0][0]


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

# =========================================================
# 🔥 BATCH RESUME SCREENING
# =========================================================
@app.route("/batch-screen", methods=["POST"])
def batch_screen():
    try:
        if not login_required():
            return redirect(url_for("login"))

        if not MODELS:
            return render_template("batch_results.html", error="ML models not loaded")

        files = request.files.getlist("resumes")
        job_desc = request.form.get("job_desc", "")
        job_role = request.form.get("job_role")

        if job_role in JOB_TEMPLATES and not job_desc.strip():
            job_desc = JOB_TEMPLATES[job_role]

        results = []

        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue

            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() or "" for page in reader.pages)

            if not text.strip():
                continue

            scores = score_resume(text, job_desc)

            results.append({
                "filename": file.filename,
                "final": round(scores["final"] * 100, 1),
                "quality": round(scores["quality"] * 100, 1),
                "match": round(scores["match"] * 100, 1),
                "coverage": scores["coverage"]
            })

        if not results:
            return render_template("batch_results.html", error="No valid resumes found")

        # 🔥 SORT
        results.sort(key=lambda x: x["final"], reverse=True)

        # 🔥 PAGINATION
        page = int(request.args.get("page", 1))
        per_page = 5
        total = len(results)
        total_pages = ceil(total / per_page)

        start = (page - 1) * per_page
        end = start + per_page
        paginated_results = results[start:end]

        session["last_batch_results"] = results
        session["last_job_role"] = job_role


        return render_template(
            "batch_results.html",
            ranked_results=paginated_results,
            page=page,
            total_pages=total_pages
        )

    except Exception as e:
        print("BATCH ERROR:", e)
        return render_template("batch_results.html", error="Internal server error")

@app.route("/export-batch-pdf")
def export_batch_pdf():
    if not login_required():
        return redirect("/login")

    results = session.get("last_batch_results")
    job_role = session.get("last_job_role", "N/A")

    if not results:
        return "No batch data to export", 400

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(
        f"<b>Batch Resume Screening Report</b><br/>Job Role: {job_role}",
        styles["Title"]
    ))

    table_data = [["Rank", "Filename", "Final %", "Quality %", "Match %", "Coverage %"]]

    for i, r in enumerate(results, start=1):
        table_data.append([
            i, r["filename"], r["final"], r["quality"], r["match"], r["coverage"]
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.grey),
        ("ALIGN", (2,1), (-1,-1), "CENTER"),
    ]))

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="batch_screening_results.pdf",
        mimetype="application/pdf"
    )






# =========================================================
# 🔥 COMPARE TWO RESUMES
# =========================================================
@app.route("/compare-resumes", methods=["POST"])
def compare_resumes():
    if not login_required():
        return redirect(url_for("login"))

    if not MODELS:
        return render_template(
            "compare_results.html",
            error="⚠️ ML models not loaded."
        )

    r1 = request.files.get("resume1")
    r2 = request.files.get("resume2")
    job_desc = request.form.get("job_desc", "")
    job_role = request.form.get("job_role")

    if job_role in JOB_TEMPLATES and not job_desc.strip():
        job_desc = JOB_TEMPLATES[job_role]

    def extract_text(file):
        reader = PyPDF2.PdfReader(file)
        return " ".join(
            p.extract_text() for p in reader.pages if p.extract_text()
        )

    t1 = extract_text(r1)
    t2 = extract_text(r2)

    s1 = score_resume(t1, job_desc)
    s2 = score_resume(t2, job_desc)

    result = {
        "resume_1": {
            "final": round(s1["final"] * 100, 2),
            "match": round(s1["match"] * 100, 2),
            "coverage": s1["coverage"]
        },
        "resume_2": {
            "final": round(s2["final"] * 100, 2),
            "match": round(s2["match"] * 100, 2),
            "coverage": s2["coverage"]
        },
        "winner": "resume_1" if s1["final"] > s2["final"] else "resume_2"
    }

    # 🔐 store for PDF export
    session["last_compare_result"] = result

    return render_template("compare_results.html", **result)


@app.route("/export-compare-pdf")
def export_compare_pdf():
    if not login_required():
        return redirect("/login")

    data = session.get("last_compare_result")
    if not data:
        return "No comparison data", 400

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Resume Comparison Report</b>", styles["Title"]))
    elements.append(Paragraph(
        f"Winner: <b>{data['winner'].replace('_', ' ').upper()}</b>",
        styles["Heading2"]
    ))

    table_data = [
        ["Metric", "Resume 1", "Resume 2"],
        ["Final Score", data["r1"]["final"], data["r2"]["final"]],
        ["Job Match", data["r1"]["match"], data["r2"]["match"]],
        ["JD Coverage", data["r1"]["coverage"], data["r2"]["coverage"]],
    ]

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.black),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (1,1), (-1,-1), "CENTER")
    ]))

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="resume_comparison.pdf",
        mimetype="application/pdf"
    )

@app.route("/compare-from-batch", methods=["POST"])
def compare_from_batch():
    if not login_required():
        return jsonify({"error": "Unauthorized"}), 401

    selected = request.form.getlist("compare_ids")

    if len(selected) != 2:
        return "Please select exactly 2 resumes to compare", 400

    batch = session.get("last_batch_results")
    if not batch:
        return "Batch results expired. Please re-run batch screening.", 400

    r1 = batch[int(selected[0])]
    r2 = batch[int(selected[1])]

    session["last_compare_results"] = {
        "resume_1": r1,
        "resume_2": r2,
        "winner": "resume_1" if r1["final"] > r2["final"] else "resume_2"
    }

    return render_template(
        "compare_results.html",
        resume_1=r1,
        resume_2=r2,
        winner="resume_1" if r1["final"] > r2["final"] else "resume_2",
        from_batch=True
    )






# ================= UTIL =================
@app.route("/download/<f>")
def download_file(f):
    if not login_required():
        return redirect(url_for("login"))
    return send_file(os.path.join(UPLOAD, f), as_attachment=True)

@app.route("/api/health")
def health():
    return jsonify({
        "models_loaded": MODELS,
        "auth": "enabled",
        "batch_screening": True,
        "resume_comparison": True,
        "features": [
            "single_resume_analysis",
            "batch_screening",
            "resume_comparison",
            "pdf_report_generation",
            "skill_gap_analysis",
            "ats_decision_logic"
        ]
    })

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)















