"""
config.py - Centralized Database and Application Configuration (SQLite Edition)
"""

import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ============================================================
# DATABASE CONFIGURATION
# ============================================================

# Use environment variable for DB path, or default to local file
DB_DIR = os.environ.get('DB_DIR', os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(DB_DIR, 'sales_analyzer.db')

# Ensure the database directory exists
Path(DB_DIR).mkdir(parents=True, exist_ok=True)

def get_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable row name accessing
    return conn

# ============================================================
# APPLICATION SETTINGS
# ============================================================

SECRET_KEY = os.environ.get('SECRET_KEY', 'sales_analyzer_secret_key_2024')
UPLOAD_FOLDER = os.path.join(DB_DIR, 'uploads')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# GetOTP Verification API Configuration
GETOTP_API_KEY = os.environ.get('GETOTP_API_KEY', '691658327da93697590df2af20bce042')
GETOTP_TEMPLATE_ID = os.environ.get('GETOTP_TEMPLATE_ID', 'ef7c4589-1f21-4fa1-8d35-ccaf81f88d0e')
GETOTP_SENDER = os.environ.get('GETOTP_SENDER', 'Pranjal Badola')
