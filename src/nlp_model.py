import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re

class ResumeScreener:
    def __init__(self):
        self.skills_db = self._create_skills_db()
        self.jds = self._create_job_descriptions()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model_path = 'models/tfidf_vectorizer.pkl'
        self._load_or_train_model()
    
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
                'agile', ' scrum', 'mongodb', 'postgresql'
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
    
    def score_resume(self, resume_text, job_category):
        jd_text = self.jds.get(job_category, "")
        skills = self.skills_db.get(job_category, [])
        
        # TF-IDF similarity
        texts = [resume_text, jd_text]
        tfidf_matrix = self.vectorizer.transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Skill matching
        resume_words = set(resume_text.lower().split())
        matched_skills = [s for s in skills if s.lower() in resume_words]
        skill_score = len(matched_skills) / max(len(skills), 1)
        
        # Experience extraction (heuristic)
        exp_score = self._extract_experience(resume_text)
        
        # Final weighted score
        final_score = (similarity * 0.5 + skill_score * 0.3 + exp_score * 0.2) * 100
        
        # Heatmap data for visualization
        heatmap_data = self._generate_heatmap_data(resume_text, jd_text)
        
        return {
            'overall': round(final_score, 2),
            'tfidf': round(similarity * 100, 2),
            'skills': round(skill_score * 100, 2),
            'experience': round(exp_score * 100, 2),
            'matched_skills': matched_skills[:10],  # Top 10
            'missing_skills': [s for s in skills[:10] if s.lower() not in resume_words],
            'heatmap_data': heatmap_data,
            'job_category': job_category.title().replace('_', ' ')
        }
    
    def _extract_experience(self, text):
        """Extract experience years heuristically"""
        years = re.findall(r'(\d+)\s*(?:years?|yrs?|experience)', text.lower())
        total_years = sum(int(y) for y in years)
        return min(total_years / 10, 1.0) if years else 0.3  # Default 30%
    
    def _generate_heatmap_data(self, resume, jd):
        """Generate data for word overlap heatmap"""
        resume_words = set(resume.lower().split())
        jd_words = set(jd.lower().split())
        common = resume_words.intersection(jd_words)
        
        return {
            'common_words': len(common),
            'resume_unique': len(resume_words - jd_words),
            'jd_unique': len(jd_words - resume_words),
            'common_percentage': round(len(common) / len(jd_words) * 100, 2) if jd_words else 0
        }
