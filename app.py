import os
import json
import hashlib
import zipfile
from functools import wraps
from datetime import datetime

from flask import (
    Flask, render_template, request,
    jsonify, send_file, redirect, url_for, session
)
from werkzeug.utils import secure_filename

from config import Config
from src.preprocess import extract_text, preprocess_text, strip_identity_info
from src.nlp_model import ResumeScreener
from src.report import generate_pdf_report

# ---------- APP SETUP ----------

app = Flask(__name__)
app.config.from_object(Config)
app.static_folder = 'static'
app.secret_key = app.config['SECRET_KEY']

# ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

screener = ResumeScreener()
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# ---------- UTILITIES ----------

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _users_file_path() -> str:
    return app.config.get('USERS_FILE', 'data/users.json')

def _load_users() -> dict:
    """Load users from JSON file."""
    path = _users_file_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def _save_users(users: dict) -> None:
    """Save users dict to JSON file."""
    path = _users_file_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)

def _hash_password(password: str) -> str:
    """Very simple hash for demo purposes."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def login_required(role: str | None = None):
    """Decorator: require login, and optionally a specific role."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'user' not in session:
                return redirect(url_for('login'))
            if role and session.get('role') != role:
                return "Forbidden", 403
            return f(*args, **kwargs)
        return wrapper
    return decorator

# ---------- AUTH ROUTES ----------

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Sign in existing HR/Admin user."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        users = _load_users()
        user = users.get(username)

        if user and user.get('password_hash') == _hash_password(password):
            session['user'] = username
            session['role'] = user.get('role', 'hr')
            return redirect(url_for('index'))

        return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register a new HR or Admin user."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm = request.form.get('confirm', '').strip()
        role = request.form.get('role', '').strip()  # 'hr' or 'admin'

        if not username or not password or not confirm or not role:
            return render_template('register.html', error="All fields are required")

        if password != confirm:
            return render_template('register.html', error="Passwords do not match")

        if role not in ('hr', 'admin'):
            return render_template('register.html', error="Invalid role selected")

        users = _load_users()
        if username in users:
            return render_template('register.html', error="Username already exists")

        users[username] = {
            "password_hash": _hash_password(password),
            "role": role
        }
        _save_users(users)

        # auto-login after registration
        session['user'] = username
        session['role'] = role
        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    """Log out current user."""
    session.clear()
    return redirect(url_for('login'))

@app.route('/candidate', methods=['GET', 'POST'])
def candidate():
    """Public candidate feedback page. No login required."""
    if request.method == 'GET':
        return render_template('candidate.html')

    # POST: handle candidate upload
    try:
        if 'resume' not in request.files or 'job_category' not in request.form:
            return render_template(
                'candidate.html',
                error="Please select a role and upload your resume."
            )

        file = request.files['resume']
        job_category = request.form['job_category']

        if file.filename == '' or not allowed_file(file.filename):
            return render_template(
                'candidate.html',
                error="Invalid file type. Please upload PDF or DOCX."
            )

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process resume (no blind mode for candidates)
        raw_text = extract_text(filepath)
        processed_text = preprocess_text(raw_text)
        results = screener.score_resume(processed_text, job_category)

        # Candidate mode: mark it and do NOT generate PDF
        results['candidate_mode'] = True

        return render_template('candidate.html', results=results, job_category=job_category)

    except Exception as e:
        return render_template('candidate.html', error=str(e))

# ---------- MAIN APP ROUTES ----------

@app.route('/')
@login_required()  # any logged-in user
def index():
    """Main dashboard page (upload + charts)."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@login_required(role='hr')  # only HR can run screening
def analyze():
    """Handle single OR batch resume upload, run NLP scoring, generate PDFs."""
    try:
        job_category = request.form.get('job_category')
        if not job_category:
            return jsonify({'error': 'Job category required'}), 400

        batch_mode = 'batch_mode' in request.form
        blind_mode = 'blind_mode' in request.form

        if batch_mode:
            # **IDEA 4: BATCH PROCESSING**
            files = request.files.getlist('resumes')
            if not files or len(files) == 0:
                return jsonify({'error': 'No files selected'}), 400
            
            if len(files) > 10:
                return jsonify({'error': 'Max 10 files allowed'}), 400

            results = []
            pdf_files = []
            
            for i, file in enumerate(files):
                if file.filename == '' or not allowed_file(file.filename):
                    continue
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Process each resume
                raw_text = extract_text(filepath)
                
                # Apply blind screening if enabled
                raw_for_scoring = strip_identity_info(raw_text) if blind_mode else raw_text
                processed_text = preprocess_text(raw_for_scoring)
                result = screener.score_resume(processed_text, job_category)
                
                # Add filename to result for frontend display
                result['filename'] = filename
                result['blind_mode'] = blind_mode
                
                # Generate PDF for each (HR needs full reports)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_name = f"batch_{timestamp}_{i+1}_{filename.rsplit('.', 1)[0]}_{job_category}"
                generate_pdf_report(pdf_name, result, raw_text, job_category)
                pdf_files.append(pdf_name + '.pdf')
                
                results.append(result)

            # Sort by score for leaderboard
            results.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
            
            return jsonify({
                'success': True,
                'batch_results': results,
                'pdf_files': pdf_files,
                'total_processed': len([r for r in results if r.get('filename')])
            })

        else:
            # SINGLE resume processing
            if 'resume' not in request.files:
                return jsonify({'error': 'No resume file'}), 400

            file = request.files['resume']
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Use PDF or DOCX'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process resume
            raw_text = extract_text(filepath)
            
            # Apply blind screening if enabled
            raw_for_scoring = strip_identity_info(raw_text) if blind_mode else raw_text
            processed_text = preprocess_text(raw_for_scoring)
            results = screener.score_resume(processed_text, job_category)
            results['blind_mode'] = blind_mode

            # Generate PDF report
            pdf_name = f"report_{filename.rsplit('.', 1)[0]}_{job_category}"
            generate_pdf_report(pdf_name, results, raw_text, job_category)

            return jsonify({
                'success': True,
                'results': results,
                'pdf_url': f"/download/{pdf_name}.pdf"
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
@login_required()
def download(filename):
    """Download single generated PDF report."""
    filepath = os.path.join('outputs', filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/download-batch/<batch_id>')
@login_required()
def download_batch(batch_id):
    """Download all PDFs from a batch as ZIP."""
    try:
        # Find all PDF files for this batch
        batch_files = [f for f in os.listdir('outputs') 
                      if f.startswith(f'batch_{batch_id}') and f.endswith('.pdf')]
        
        if not batch_files:
            return jsonify({'error': 'No batch files found'}), 404
        
        zip_path = os.path.join('outputs', f'{batch_id}.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for pdf_file in batch_files:
                pdf_path = os.path.join('outputs', pdf_file)
                zipf.write(pdf_path, pdf_file)
        
        return send_file(zip_path, as_attachment=True, download_name=f'{batch_id}.zip')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------- ENTRY POINT ----------

if __name__ == '__main__':
    app.run(debug=True, port=5000)
