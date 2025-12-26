import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re
from collections import Counter


class ResumeScreener:
    def __init__(self):
        self.skills_db = self._create_skills_db()
        self.jds = self._create_job_descriptions()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model_path = 'models/tfidf_vectorizer.pkl'
        self._load_or_train_model()

        # Critical skills (subset) for explanation
        self.critical_skills = {
            'data_science': ['python', 'sql', 'machine learning', 'statistics'],
            'software_engineer': ['python', 'java', 'javascript', 'rest api', 'docker'],
            'product_manager': ['roadmap', 'stakeholder', 'metrics', 'sql']
        }

    def _create_skills_db(self):
        """Default skills database"""
        return {
            'data_science': [
                'python', 'r', 'sql', 'pandas', 'numpy', 'scikit-learn', 'tensorflow',
                'pytorch', 'machine learning', 'deep learning', 'nlp', 'computer vision',
                'data analysis', 'statistics', 'tableau', 'power bi', 'aws', 'azure'
            ],
            'software_engineer': [
                'python', 'java', 'javascript', 'react', 'node.js', 'docker', 'kubernetes',
                'aws', 'microservices', 'rest api', 'sql', 'nosql', 'git', 'jenkins',
                'agile', 'scrum', 'mongodb', 'postgresql'
            ],
            'product_manager': [
                'agile', 'scrum', 'jira', 'confluence', 'stakeholder', 'roadmap',
                'user stories', 'a/b testing', 'metrics', 'kpi', 'customer success',
                'product lifecycle', 'market research', 'sql', 'analytics'
            ]
        }

    def _create_job_descriptions(self):
        """Sample job descriptions"""
        return {
            'data_science': """
            Looking for Data Scientist with 3+ years experience in Python, SQL, Machine Learning. 
            Must have experience with pandas, scikit-learn, AWS. Knowledge of deep learning preferred.
            """,
            'software_engineer': """
            Software Engineer role requiring Python/Java, REST APIs, Docker, AWS. Experience with 
            microservices, SQL databases, and CI/CD pipelines required. Agile methodology experience.
            """,
            'product_manager': """
            Product Manager with experience in Agile/Scrum, JIRA, stakeholder management, and metrics 
            analysis. SQL knowledge and experience with A/B testing preferred.
            """
        }

    def _load_or_train_model(self):
        """Load existing model or train new one"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            # Train on sample data
            all_texts = list(self.jds.values())
            self.vectorizer.fit(all_texts)
            os.makedirs('models', exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)

    def score_resume(self, resume_text, job_category, job_description=None):
        """
        **IDEA 6**: Score resume against job category OR custom job description.
        
        Args:
            resume_text: Processed resume text
            job_category: Job role ('data_science', etc.)
            job_description: Optional custom JD text
        """
        # **IDEA 6: Use custom JD if provided**
        if job_description:
            jd_text = job_description
            jd_mode = True
        else:
            jd_text = self.jds.get(job_category, "")
            jd_mode = False

        skills = self.skills_db.get(job_category, [])

        # ---------- TF-IDF similarity ----------
        texts = [resume_text, jd_text]
        tfidf_matrix = self.vectorizer.transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # ---------- Skill matching ----------
        resume_words = set(resume_text.lower().split())
        matched_skills = [s for s in skills if s.lower() in resume_words]
        skill_score = len(matched_skills) / max(len(skills), 1)

        # ---------- Experience extraction ----------
        exp_score, years = self._extract_experience(resume_text)

        # ---------- **IDEA 6: JD-specific keyword matching** ----------
        jd_keywords = self._extract_keywords(jd_text)
        resume_keywords = self._extract_keywords(resume_text)
        jd_keyword_match = len(set(jd_keywords) & set(resume_keywords)) / max(len(jd_keywords), 1)

        # ---------- Final weighted score ----------
        if jd_mode:
            # **JD Mode**: 60% TF-IDF + 20% skills + 10% JD keywords + 10% experience
            final_score = (similarity * 0.6 + skill_score * 0.2 + jd_keyword_match * 0.1 + exp_score * 0.1) * 100
        else:
            # **Category Mode**: Original weighting
            final_score = (similarity * 0.5 + skill_score * 0.3 + exp_score * 0.2) * 100

        # ---------- Heatmap data ----------
        heatmap_data = self._generate_heatmap_data(resume_text, jd_text)

        # ---------- Explainability ----------
        top_phrases = self._get_top_matching_phrases(resume_text, jd_text)
        missing_critical = self._get_missing_critical_skills(matched_skills, job_category)
        experience_explanation = self._build_experience_explanation(exp_score, years)

        explanation = {
            "top_matching_phrases": top_phrases[:5],
            "missing_critical_skills": missing_critical[:10],
            "experience_explanation": experience_explanation
        }

        result = {
            'overall_score': round(final_score, 2),  # Renamed for frontend consistency
            'overall': round(final_score, 2),       # Keep backward compatibility
            'tfidf_similarity': round(similarity * 100, 2),
            'tfidf': round(similarity * 100, 2),    # Backward compatibility
            'skills_match': round(skill_score * 100, 2),
            'skills': round(skill_score * 100, 2),  # Backward compatibility
            'experience': round(exp_score * 100, 2),
            'jd_keyword_match': round(jd_keyword_match * 100, 2) if jd_mode else 0,
            'jd_mode': jd_mode,
            'matched_skills': matched_skills[:10],
            'missing_skills': [s for s in skills[:10] if s.lower() not in resume_words],
            'jd_used_keywords': list(set(jd_keywords) & set(resume_keywords))[:10] if jd_mode else [],
            'jd_missing_keywords': list(set(jd_keywords) - set(resume_keywords))[:10] if jd_mode else [],
            'heatmap_data': heatmap_data,
            'job_category': job_category.title().replace('_', ' '),
            'explanation': explanation
        }

        return result

    def _extract_keywords(self, text, top_n=20):
        """Extract key skills/keywords from text."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Filter common stop words and short terms
        filtered = [w for w in words if len(w) > 3 and w not in {
            'with', 'from', 'have', 'been', 'that', 'this', 'your', 'will', 'they'
        }]
        return list(set(filtered))[:top_n]

    def _extract_experience(self, text):
        """Extract experience years heuristically"""
        years = re.findall(r'(\d+)\s*(?:years?|yrs?|experience)', text.lower())
        total_years = sum(int(y) for y in years)
        years_val = total_years if years else 0

        # Map years to 0–1 score
        if years_val >= 10:
            score = 1.0
        elif years_val >= 5:
            score = 0.8
        elif years_val >= 2:
            score = 0.6
        elif years_val > 0:
            score = 0.4
        else:
            score = 0.3  # Default 30% if no explicit years

        return score, years_val

    def _generate_heatmap_data(self, resume, jd):
        """Generate data for word overlap heatmap"""
        resume_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', resume.lower()))
        jd_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', jd.lower()))
        common = resume_words.intersection(jd_words)

        return {
            'common_words': len(common),
            'resume_unique': len(resume_words - jd_words),
            'jd_unique': len(jd_words - resume_words),
            'common_percentage': round(len(common) / len(jd_words) * 100, 2) if jd_words else 0
        }

    # ---------- explainability helpers (unchanged) ----------

    def _get_top_matching_phrases(self, resume_text, jd_text):
        """Most frequent overlapping tokens between resume and JD."""
        resume_tokens = [t.lower() for t in re.findall(r'\w+', resume_text)]
        jd_tokens = set([t.lower() for t in re.findall(r'\w+', jd_text)])

        freq = Counter()
        for tok in resume_tokens:
            if tok in jd_tokens:
                freq[tok] += 1

        sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [t[0] for t in sorted_tokens]

    def _get_missing_critical_skills(self, matched_skills, job_category):
        matched_lower = {s.lower() for s in matched_skills}
        critical = self.critical_skills.get(job_category, [])
        return [s for s in critical if s.lower() not in matched_lower]

    def _build_experience_explanation(self, exp_score, years_experience):
        if years_experience >= 8:
            return f"Strong experience (~{years_experience} years) for this role."
        elif years_experience >= 4:
            return f"Moderate experience (~{years_experience} years); fits most requirements."
        elif years_experience >= 1:
            return f"Some experience (~{years_experience} years); may need mentoring."
        else:
            return "No explicit years of experience detected; score based on other signals."
