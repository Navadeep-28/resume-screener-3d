import os
import json
import hashlib
from functools import wraps

from flask import (
    Flask, render_template, request,
    jsonify, send_file, redirect, url_for, session
)
from werkzeug.utils import secure_filename

from config import Config
from src.preprocess import extract_text, preprocess_text
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


# ---------- MAIN APP ROUTES ----------

@app.route('/')
@login_required()  # any logged-in user
def index():
    """Main dashboard page (upload + charts)."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
@login_required(role='hr')  # only HR can run screening
def analyze():
    """Handle resume upload, run NLP scoring, generate PDF."""
    try:
        if 'resume' not in request.files or 'job_category' not in request.form:
            return jsonify({'error': 'Missing files or job category'}), 400

        file = request.files['resume']
        job_category = request.form['job_category']

        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PDF or DOCX'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process resume text
        raw_text = extract_text(filepath)
        processed_text = preprocess_text(raw_text)
        results = screener.score_resume(processed_text, job_category)

        # Generate PDF report
        pdf_name = f"report_{filename.rsplit('.', 1)[0]}_{job_category}"
        generate_pdf_report(pdf_name, results, raw_text, job_category)

        return jsonify({
            'success': True,
            'results': results,
            'pdf_url': f"/download/{pdf_name}.pdf"
        })

    except Exception as e:
        # In debug mode, full traceback will appear in terminal
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
@login_required()   # only logged-in users can download
def download(filename):
    """Download generated PDF report."""
    filepath = os.path.join('outputs', filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


# ---------- ENTRY POINT ----------

if __name__ == '__main__':
    app.run(debug=True, port=5000)
