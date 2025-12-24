import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Secret key for Flask sessions
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') or 'dev-key-3d-resume-screener-2025'

    # File upload configuration
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    # Job categories supported by the screener
    CATEGORIES = ['data_science', 'software_engineer', 'product_manager']

    # Path to JSON file where HR/Admin accounts are stored
    USERS_FILE = 'data/users.json'
