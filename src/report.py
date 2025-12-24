from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

def _get_status(score):
    """Get status icon for score (standalone function, no self)."""
    if score >= 80:
        return "🟢 Excellent"
    elif score >= 60:
        return "🟡 Good"
    else:
        return "🔴 Poor"

def generate_pdf_report(filename, results, resume_text, job_category):
    """Generate comprehensive PDF report."""
    doc = SimpleDocTemplate(f"outputs/{filename}.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph(f"AI Resume Screening Report - {results['job_category']}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 20))

    # Overall Score
    overall_text = (
        f"<b>Overall Score: {results['overall']}%</b><br/>"
        f"<font size=12>{'✅ ACCEPTED' if results['overall'] >= 70 else '❌ NEEDS IMPROVEMENT'}</font>"
    )
    story.append(Paragraph(overall_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Scores Table
    data = [
        ['Metric', 'Score', 'Status'],
        ['TF-IDF Similarity', f"{results['tfidf']}%", _get_status(results['tfidf'])],
        ['Skills Match', f"{results['skills']}%", _get_status(results['skills'])],
        ['Experience', f"{results['experience']}%", _get_status(results['experience'])]
    ]

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

    # Matched Skills
    story.append(Paragraph("✅ MATCHED SKILLS", styles['Heading2']))
    matched = ', '.join(results['matched_skills']) if results['matched_skills'] else 'None'
    story.append(Paragraph(matched, styles['Normal']))
    story.append(Spacer(1, 12))

    # Missing Skills
    story.append(Paragraph("❌ MISSING SKILLS", styles['Heading2']))
    missing = ', '.join(results['missing_skills']) if results['missing_skills'] else 'None'
    story.append(Paragraph(missing, styles['Normal']))

    doc.build(story)
