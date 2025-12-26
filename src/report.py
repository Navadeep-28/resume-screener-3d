from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import os

os.makedirs("outputs", exist_ok=True)

def _get_status(score):
    """Get status icon for score (standalone function, no self)."""
    if score >= 80:
        return "🟢 Excellent"
    elif score >= 60:
        return "🟡 Good"
    else:
        return "🔴 Poor"

def generate_pdf_report(filename, results, resume_text, job_category, job_description=None):
    """Generate comprehensive PDF report (with optional JD section)."""
    doc = SimpleDocTemplate(f"outputs/{filename}.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # ---------- TITLE ----------
    title = Paragraph(f"AI Resume Screening Report - {results.get('job_category', job_category)}",
                      styles['Title'])
    story.append(title)
    story.append(Spacer(1, 20))

    # ---------- OVERALL SCORE ----------
    overall = results.get('overall_score') or results.get('overall', 0)
    decision = "✅ ACCEPTED" if overall >= 70 else "❌ NEEDS IMPROVEMENT"
    overall_text = (
        f"<b>Overall Score: {overall:.2f}%</b><br/>"
        f"<font size=12>{decision}</font>"
    )
    story.append(Paragraph(overall_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # ---------- SCORES TABLE ----------
    tfidf = results.get('tfidf_similarity') or results.get('tfidf', 0)
    skills = results.get('skills_match') or results.get('skills', 0)
    exp = results.get('experience', 0)
    jd_kw = results.get('jd_keyword_match', 0)
    jd_mode = results.get('jd_mode', False)

    data = [
        ['Metric', 'Score', 'Status'],
        ['TF-IDF Similarity', f"{tfidf:.2f}%", _get_status(tfidf)],
        ['Skills Match', f"{skills:.2f}%", _get_status(skills)],
        ['Experience', f"{exp:.2f}%", _get_status(exp)],
    ]

    # Add JD row only when JD mode is used
    if jd_mode:
        data.append(['JD Keyword Match', f"{jd_kw:.2f}%", _get_status(jd_kw)])

    table = Table(data, colWidths=[2*inch, inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white])
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # ---------- MATCHED SKILLS ----------
    story.append(Paragraph("✅ MATCHED SKILLS", styles['Heading2']))
    matched = ', '.join(results.get('matched_skills', [])) or 'None'
    story.append(Paragraph(matched, styles['Normal']))
    story.append(Spacer(1, 12))

    # ---------- MISSING SKILLS ----------
    story.append(Paragraph("❌ MISSING SKILLS", styles['Heading2']))
    missing = ', '.join(results.get('missing_skills', [])) or 'None'
    story.append(Paragraph(missing, styles['Normal']))
    story.append(Spacer(1, 20))

    # ---------- JD SECTION (IDEA 6) ----------
    if job_description:
        story.append(Paragraph("📄 JOB DESCRIPTION USED", styles['Heading2']))
        jd_para = job_description.replace('\n', '<br/>')
        story.append(Paragraph(jd_para, styles['Normal']))
        story.append(Spacer(1, 12))

        jd_used = results.get('jd_used_keywords', []) or []
        jd_missing = results.get('jd_missing_keywords', []) or []

        if jd_used:
            story.append(Paragraph("✅ JD KEYWORDS FOUND IN RESUME", styles['Heading3']))
            story.append(Paragraph(', '.join(jd_used), styles['Normal']))
            story.append(Spacer(1, 8))

        if jd_missing:
            story.append(Paragraph("⚠️ IMPORTANT JD KEYWORDS MISSING", styles['Heading3']))
            story.append(Paragraph(', '.join(jd_missing), styles['Normal']))
            story.append(Spacer(1, 12))

    # ---------- OPTIONAL: EXPLANATION BLOCK ----------
    explanation = results.get('explanation', {})
    top_phrases = explanation.get('top_matching_phrases', [])
    missing_critical = explanation.get('missing_critical_skills', [])
    exp_expl = explanation.get('experience_explanation', "")

    if top_phrases or missing_critical or exp_expl:
        story.append(Paragraph("🧠 WHY THIS SCORE?", styles['Heading2']))

        if top_phrases:
            story.append(Paragraph(
                "Top matching phrases: " + ', '.join(top_phrases),
                styles['Normal']
            ))
            story.append(Spacer(1, 6))

        if missing_critical:
            story.append(Paragraph(
                "Missing critical skills: " + ', '.join(missing_critical),
                styles['Normal']
            ))
            story.append(Spacer(1, 6))

        if exp_expl:
            story.append(Paragraph(exp_expl, styles['Normal']))
            story.append(Spacer(1, 12))

    # ---------- BUILD PDF ----------
    doc.build(story)
