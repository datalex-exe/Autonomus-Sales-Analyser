"""
app.py - Production Flask Web Application
=========================================
Features:
- Home/Landing page (static HTML)
- Login/Register pages for organizations
- Upload page for user CSV/Excel files (with multi-tenant isolation)
- Dashboard with working insights display (multi-tenant)
- API endpoints for all data
- Production-ready with CORS, env vars, and error handling
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, send_from_directory, send_file, session
from flask_cors import CORS
import pandas as pd
import os
import math
import io
import matplotlib
from datetime import datetime
from functools import wraps
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Import centralized configuration settings
from config import (
    SECRET_KEY, UPLOAD_FOLDER, MAX_CONTENT_LENGTH, ALLOWED_EXTENSIONS, DB_DIR,
    GETOTP_API_KEY, GETOTP_TEMPLATE_ID, GETOTP_SENDER
)

# Import our agents and database operations
from database import (
    init_database, get_table_as_df,
    create_organization, get_organization_by_join_code, get_organization_by_id,
    create_user, get_user_by_email, get_user_by_phone
)
from data_engineer_agent import run_data_engineer
from analyst_agent import run_analyst
from insight_agent import run_insight_agent

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = SECRET_KEY

# Enable CORS for all domains (configure for production)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Upload config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_database()


# ============================================================
# AUTHENTICATION DECORATOR & HELPER
# ============================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # For API endpoints, return JSON
            if request.path.startswith('/api/'):
                return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
            # For pages, redirect to login
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_json(data):
    if isinstance(data, pd.DataFrame):
        return safe_json(data.astype(object).where(pd.notna(data), None).to_dict(orient='records'))
    if isinstance(data, pd.Series):
        return safe_json(data.astype(object).where(pd.notna(data), None).to_dict())
    if isinstance(data, dict):
        return {k: safe_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [safe_json(v) for v in data]
    if isinstance(data, tuple):
        return [safe_json(v) for v in data]
    if isinstance(data, (pd.Timestamp, pd.Timedelta)):
        return str(data)
    if pd.isna(data):
        return None
    if isinstance(data, (float, int)) and (math.isnan(float(data)) or math.isinf(float(data))):
        return None
    return data

def get_kpi_data(org_id):
    df = get_table_as_df('sales_clean', org_id)
    if len(df) == 0:
        return {'total_revenue': 0, 'total_orders': 0, 'total_units': 0,
                'active_customers': 0, 'avg_order_value': 0}
    return {
        'total_revenue': float(df['revenue'].sum()),
        'total_orders': len(df),
        'total_units': int(df['units_sold'].sum()),
        'active_customers': int(df['customer_id'].nunique()),
        'avg_order_value': float(df['revenue'].mean())
    }

def _empty_chart_figure(message):
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    ax.axis('off')
    ax.text(0.5, 0.54, message, ha='center', va='center', fontsize=14, color='#94a3b8', fontweight='semibold')
    ax.text(0.5, 0.42, 'Run analysis to populate this visualization.', ha='center', va='center', fontsize=10, color='#64748b')
    fig.tight_layout()
    return fig

def _chart_response(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=160, bbox_inches='tight', facecolor='#1e293b')
    plt.close(fig)
    buffer.seek(0)
    response = send_file(buffer, mimetype='image/png', max_age=0)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

def _currency_formatter(value, _):
    if abs(value) >= 1000:
        return f'${value / 1000:.0f}k'
    return f'${value:,.0f}'

def _format_currency_text(value):
    if abs(value) >= 1000:
        return f'${value / 1000:,.1f}k'
    return f'${value:,.0f}'

def _apply_chart_style(ax, title, subtitle=None):
    ax.set_facecolor('#1e293b')
    ax.set_title(title, fontsize=16, fontweight='bold', color='#f8fafc', loc='left', pad=18)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color='#94a3b8')
    ax.grid(axis='y', alpha=0.15, linestyle='--', linewidth=0.8, color='#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94a3b8')

def _load_sales_frame(org_id):
    df = get_table_as_df('sales_clean', org_id)
    if len(df) == 0:
        return df
    if 'order_date' in df.columns:
        df = df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
    return df

def build_revenue_trend_chart(org_id):
    df = _load_sales_frame(org_id)
    if len(df) == 0:
        return _empty_chart_figure('No revenue data available yet.')

    trend = df.copy()
    trend['quarter'] = trend['order_date'].dt.to_period('Q').astype(str)
    grouped = trend.groupby('quarter')['revenue'].sum().reset_index()
    x = list(range(len(grouped)))
    y = grouped['revenue'].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    _apply_chart_style(ax, 'Revenue Trend by Quarter', 'Quarterly revenue momentum with latest period highlighted')
    ax.plot(x, y, color='#667eea', linewidth=3, marker='o', markersize=7, markerfacecolor='#1e293b', markeredgewidth=2)
    ax.fill_between(x, y, color='#667eea', alpha=0.12)
    ax.scatter([x[-1]], [y[-1]], s=160, color='#00d9ff', edgecolor='#1e293b', linewidth=2, zorder=4)
    ax.annotate(_format_currency_text(y[-1]), xy=(x[-1], y[-1]), xytext=(10, 10), textcoords='offset points', fontsize=10, color='#f8fafc', fontweight='bold')
    ax.set_ylabel('Revenue', color='#94a3b8')
    ax.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['quarter'], rotation=35, ha='right')
    ax.margins(x=0.04)
    fig.tight_layout()
    return fig

def build_revenue_by_region_chart(org_id):
    df = _load_sales_frame(org_id)
    if len(df) == 0 or 'region' not in df.columns:
        return _empty_chart_figure('No regional revenue data available yet.')

    grouped = df.groupby('region')['revenue'].sum().sort_values(ascending=False).reset_index()
    colors = ['#667eea', '#00d9ff', '#764ba2', '#4ecdc4', '#f093fb', '#ff6b6b']

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    _apply_chart_style(ax, 'Revenue by Region', 'Ranked by total revenue contribution')
    bars = ax.barh(grouped['region'], grouped['revenue'], color=colors[:len(grouped)], edgecolor='none', height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel('Revenue', color='#94a3b8')
    ax.xaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    for bar, value in zip(bars, grouped['revenue'].tolist()):
        ax.text(bar.get_width() + max(grouped['revenue']) * 0.015, bar.get_y() + bar.get_height() / 2, _format_currency_text(value), va='center', ha='left', fontsize=9, color='#e2e8f0', fontweight='bold')
    ax.margins(x=0.02)
    fig.tight_layout()
    return fig

def build_segment_distribution_chart(org_id):
    df = get_table_as_df('customer_segments', org_id)
    if len(df) == 0 or 'segment' not in df.columns:
        return _empty_chart_figure('No customer segment data available yet.')

    grouped = df['segment'].value_counts()
    colors = ['#00d9ff', '#667eea', '#f093fb', '#ff6b6b', '#4ecdc4']
    total = grouped.sum() or 1

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    wedges, texts, autotexts = ax.pie(
        grouped.values,
        labels=grouped.index,
        autopct=lambda pct: f'{pct:.0f}%' if pct >= 4 else '',
        startangle=90,
        colors=colors[:len(grouped)],
        wedgeprops={'width': 0.38, 'edgecolor': '#1e293b', 'linewidth': 2},
        pctdistance=0.8,
        textprops={'color': '#f8fafc', 'fontsize': 10, 'fontweight': 'bold'}
    )
    for text in texts:
        text.set_color('#94a3b8')
        text.set_fontsize(10)
    ax.text(0, 0.08, f'{int(total)}', ha='center', va='center', fontsize=24, fontweight='bold', color='#f8fafc')
    ax.text(0, -0.12, 'Customers', ha='center', va='center', fontsize=10, color='#94a3b8')
    ax.legend(wedges, grouped.index, title='Segments', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    
    legend = ax.get_legend()
    if legend:
        legend.get_title().set_color('#f8fafc')
        for text in legend.get_texts():
            text.set_color('#94a3b8')

    ax.set_title('Customer Segments', fontsize=16, fontweight='bold', color='#f8fafc', loc='left', pad=18)
    fig.tight_layout()
    return fig

def build_forecast_chart(org_id):
    df = get_table_as_df('forecasts', org_id)
    if len(df) == 0 or 'region' not in df.columns:
        return _empty_chart_figure('No forecast data available yet.')

    grouped = df[['region', 'predicted_revenue', 'lower_bound', 'upper_bound']].copy().fillna(0)
    x = list(range(len(grouped)))
    predicted = grouped['predicted_revenue'].tolist()
    lower = grouped['lower_bound'].tolist()
    upper = grouped['upper_bound'].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    _apply_chart_style(ax, 'Revenue Forecasts', 'Predicted revenue with confidence range by region')
    ax.vlines(x, lower, upper, color='#9db4ff', linewidth=8, alpha=0.35, zorder=1)
    ax.scatter(x, predicted, s=120, color='#00d9ff', edgecolor='#1e293b', linewidth=2, zorder=3, label='Predicted')
    ax.plot(x, predicted, color='#667eea', linewidth=2, zorder=2)
    for index, value in enumerate(predicted):
        ax.text(index, value + max(upper) * 0.02, _format_currency_text(value), ha='center', va='bottom', fontsize=9, color='#f8fafc', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['region'], rotation=25, ha='right')
    ax.set_ylabel('Revenue', color='#94a3b8')
    ax.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    ax.legend(frameon=False, loc='upper left')
    
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color('#94a3b8')

    ax.margins(x=0.06)
    fig.tight_layout()
    return fig

@app.route('/api/charts/<chart_name>.png')
@login_required
def api_chart_image(chart_name):
    org_id = session['org_id']
    chart_builders = {
        'revenue-trend': lambda: build_revenue_trend_chart(org_id),
        'revenue-by-region': lambda: build_revenue_by_region_chart(org_id),
        'segment-distribution': lambda: build_segment_distribution_chart(org_id),
        'forecasts': lambda: build_forecast_chart(org_id),
    }

    builder = chart_builders.get(chart_name)
    if not builder:
        return jsonify({'status': 'error', 'message': 'Unknown chart'}), 404

    try:
        return _chart_response(builder())
    except Exception as e:
        print(f'Chart generation failed for {chart_name}: {e}')
        return _chart_response(_empty_chart_figure('Chart unavailable. Please run analysis again.'))

# ============================================================
# PAGES (HTML serving)
# ============================================================

@app.route("/")
def home():
    """Landing page - serves index.html."""
    return send_from_directory('static', 'index.html')

@app.route("/login")
def login_page():
    """Serving organization login page."""
    if 'user_id' in session:
        return redirect(url_for('home'))
    return send_from_directory('static', 'login.html')

@app.route("/register")
def register_page():
    """Serving organization registration page."""
    if 'user_id' in session:
        return redirect(url_for('home'))
    return send_from_directory('static', 'register.html')

@app.route("/upload")
@login_required
def upload_page():
    """Upload page - requires login."""
    return send_from_directory('static', 'upload.html')

@app.route("/dashboard")
@login_required
def dashboard():
    """Dashboard page - requires login."""
    return send_from_directory('static', 'dashboard.html')

# ============================================================
# AUTHENTICATION API ENDPOINTS
# ============================================================

@app.route("/api/auth/send-otp", methods=["POST"])
def api_send_otp():
    import urllib.request
    import urllib.parse
    import json

    data = request.get_json() or request.form
    action = data.get('action')  # 'create' or 'join'
    email = data.get('email')
    phone = data.get('phone')
    name = data.get('name')

    if not email or not phone or not name or not action:
        return jsonify({'status': 'error', 'message': 'Please fill all required fields.'}), 400

    # Clean up phone and email
    email_clean = email.strip().lower()
    phone_clean = ''.join(c for c in phone.strip() if c.isdigit())
    if len(phone_clean) == 10:
        phone_clean = '91' + phone_clean

    # Uniqueness checks
    existing_user = get_user_by_email(email_clean)
    if existing_user:
        return jsonify({'status': 'error', 'message': 'An account with this email already exists.'}), 400

    existing_phone = get_user_by_phone(phone_clean)
    if existing_phone:
        return jsonify({'status': 'error', 'message': 'A user with this phone number already exists.'}), 400

    if action == 'create':
        org_name = data.get('org_name')
        if not org_name:
            return jsonify({'status': 'error', 'message': 'Organization name is required.'}), 400
    elif action == 'join':
        join_code = data.get('join_code')
        if not join_code:
            return jsonify({'status': 'error', 'message': 'Organization Join Code is required.'}), 400
        org_details = get_organization_by_join_code(join_code)
        if not org_details:
            return jsonify({'status': 'error', 'message': 'Invalid Organization Join Code. Please check and try again.'}), 400

    import urllib.error

    # Call GetOTP API to send code via SMS
    url = "https://api.otp.dev/v1/verifications"
    headers = {
        "X-OTP-Key": GETOTP_API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {
        "data": {
            "channel": "sms",
            "sender": GETOTP_SENDER,
            "phone": phone_clean,
            "template": GETOTP_TEMPLATE_ID,
            "code_length": 6
        }
    }

    print(f"\n[GetOTP Request] Sender: '{GETOTP_SENDER}' | Phone: '{phone_clean}' | Template: '{GETOTP_TEMPLATE_ID}'")

    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            res_data = json.loads(res_body)
            # Response check
            if response.status in (200, 201):
                return jsonify({'status': 'success', 'message': 'OTP sent successfully!'})
            else:
                raise Exception(f"Non-200 status: {response.status} - {res_data}")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode('utf-8')
            err_data = json.loads(err_body)
            err_msg = err_data.get('message', err_body)
        except Exception:
            err_msg = str(e)
            
        print("\n" + "*" * 60)
        print(f"GetOTP API HTTP Error: {err_msg}")
        print("*" * 60 + "\n")
        
        return jsonify({
            'status': 'error',
            'message': f"Failed to send OTP via SMS: {err_msg}"
        }), 400
    except Exception as e:
        print("\n" + "*" * 60)
        print(f"GetOTP Connection Failure: {str(e)}")
        print("*" * 60 + "\n")
        
        return jsonify({
            'status': 'error',
            'message': f"Could not connect to OTP service: {str(e)}"
        }), 500

@app.route("/api/auth/register", methods=["POST"])
def api_register():
    import urllib.request
    import urllib.parse
    import json

    data = request.get_json() or request.form
    action = data.get('action')  # 'create' or 'join'
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    phone = data.get('phone')
    otp = data.get('otp')
    
    if not email or not password or not name or not action or not phone or not otp:
        return jsonify({'status': 'error', 'message': 'Please fill all required fields, including phone number and OTP.'}), 400
        
    phone_clean = ''.join(c for c in phone.strip() if c.isdigit())
    if len(phone_clean) == 10:
        phone_clean = '91' + phone_clean

    email_clean = email.strip().lower()
    existing_user = get_user_by_email(email_clean)
    if existing_user:
        return jsonify({'status': 'error', 'message': 'An account with this email already exists.'}), 400
        
    existing_phone = get_user_by_phone(phone_clean)
    if existing_phone:
        return jsonify({'status': 'error', 'message': 'A user with this phone number already exists.'}), 400

    # Verify OTP code via GetOTP API
    try:
        encoded_phone = urllib.parse.quote(phone_clean)
        verify_url = f"https://api.otp.dev/v1/verifications?code={otp.strip()}&phone={encoded_phone}"
        req = urllib.request.Request(verify_url, headers={
            "X-OTP-Key": GETOTP_API_KEY,
            "User-Agent": "Mozilla/5.0"
        })
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            res_data = json.loads(res_body)
            if not res_data.get('data'):
                return jsonify({'status': 'error', 'message': 'Invalid or expired OTP. Please verify and try again.'}), 400
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode('utf-8')
            err_data = json.loads(err_body)
            err_msg = err_data.get('message', err_body)
        except Exception:
            err_msg = str(e)
        return jsonify({'status': 'error', 'message': f'OTP verification failed: {err_msg}'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'OTP verification service error: {str(e)}'}), 500
        
    try:
        if action == 'create':
            org_name = data.get('org_name')
            if not org_name:
                return jsonify({'status': 'error', 'message': 'Organization name is required.'}), 400
            
            # Create organization and user
            org_id, join_code = create_organization(org_name)
            user_id = create_user(org_id, email_clean, password, name, phone_clean, role='admin')
            org_details = get_organization_by_id(org_id)
            
            # Log in the user
            session['user_id'] = user_id
            session['org_id'] = org_id
            session['user_name'] = name
            session['org_name'] = org_details['name']
            
            return jsonify({
                'status': 'success',
                'message': 'Organization and Administrator accounts created successfully!',
                'join_code': join_code
            })
            
        elif action == 'join':
            join_code = data.get('join_code')
            if not join_code:
                return jsonify({'status': 'error', 'message': 'Organization Join Code is required.'}), 400
                
            org_details = get_organization_by_join_code(join_code)
            if not org_details:
                return jsonify({'status': 'error', 'message': 'Invalid Organization Join Code. Please check and try again.'}), 400
                
            # Create user linked to organization
            user_id = create_user(org_details['id'], email_clean, password, name, phone_clean, role='user')
            
            # Log in the user
            session['user_id'] = user_id
            session['org_id'] = org_details['id']
            session['user_name'] = name
            session['org_name'] = org_details['name']
            
            return jsonify({
                'status': 'success',
                'message': f'Successfully joined organization {org_details["name"]}!'
            })
            
        else:
            return jsonify({'status': 'error', 'message': 'Invalid registration type.'}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Registration failed: {str(e)}'}), 500

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json() or request.form
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'status': 'error', 'message': 'Please enter email and password.'}), 400
        
    user = get_user_by_email(email)
    from werkzeug.security import check_password_hash
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'status': 'error', 'message': 'Invalid email or password.'}), 400
        
    org = get_organization_by_id(user['org_id'])
    
    # Establish session
    session['user_id'] = user['id']
    session['org_id'] = user['org_id']
    session['user_name'] = user['name']
    session['org_name'] = org['name'] if org else 'Isolated Workspace'
    
    # Check if a custom uploaded dataset exists for this session
    # (they will be automatically loaded if uploaded previously)
    
    return jsonify({
        'status': 'success',
        'message': 'Logged in successfully!',
        'user_name': user['name'],
        'org_name': session['org_name']
    })

@app.route("/api/auth/logout", methods=["GET", "POST"])
def api_logout():
    session.clear()
    return jsonify({'status': 'success', 'message': 'Logged out successfully.'})

@app.route("/api/auth/status", methods=["GET"])
def api_auth_status():
    if 'user_id' in session:
        # Check org join code
        org_details = get_organization_by_id(session['org_id'])
        join_code = org_details['join_code'] if org_details else 'UNKNOWN'
        return jsonify({
            'logged_in': True,
            'user_name': session.get('user_name'),
            'org_name': session.get('org_name'),
            'org_id': session.get('org_id'),
            'join_code': join_code
        })
    return jsonify({'logged_in': False})

# ============================================================
# MULTI-TENANT FILE UPLOADS
# ============================================================

@app.route("/api/upload", methods=["POST"])
@login_required
def api_upload():
    """API endpoint for file uploads from static HTML pages."""
    org_id = session['org_id']

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file selected!'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected!'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Create organization isolated uploads folder
        org_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], org_id)
        os.makedirs(org_upload_dir, exist_ok=True)
        filepath = os.path.join(org_upload_dir, filename)
        file.save(filepath)

        # Store in session so it remains isolated per organization
        session['dataset_path'] = filepath
        session['dataset_name'] = filename

        return jsonify({
            'status': 'success', 
            'message': f'Successfully uploaded {filename}!',
            'filename': filename
        })
    else:
        return jsonify({
            'status': 'error', 
            'message': 'Invalid file type. Please upload .csv, .xlsx, or .xls'
        }), 400

# ============================================================
# MULTI-TENANT RUN ANALYSIS
# ============================================================

@app.route("/api/run-analysis", methods=["POST"])
@login_required
def run_analysis():
    """Trigger the full agent pipeline via API."""
    org_id = session['org_id']
    try:
        # Use uploaded file if available, else sample
        dataset_path = session.get('dataset_path')
        dataset_name = session.get('dataset_name')
        
        if dataset_path and os.path.exists(dataset_path):
            data_path = dataset_path
            print(f"\nUsing uploaded dataset: {dataset_name} for Org {org_id}")
        else:
            # Create sample data if none exists
            sample_path = os.path.join(DB_DIR, 'sample_sales.csv')
            if not os.path.exists(sample_path):
                create_sample_data(sample_path)
            data_path = sample_path
            print(f"\nUsing sample dataset for Org {org_id}")

        print(f"\nStarting full analysis pipeline for Org {org_id}...")
        cleaned_df = run_data_engineer(data_path, org_id)
        analysis_results = run_analyst(org_id)
        insights = run_insight_agent(analysis_results, org_id)

        return jsonify({
            'status': 'success',
            'message': 'Analysis complete!',
            'dataset': dataset_name if (dataset_path and os.path.exists(dataset_path)) else 'sample_sales.csv',
            'records_processed': len(cleaned_df),
            'insights_generated': len(insights),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

def create_sample_data(filepath):
    """Create sample sales data for demo purposes."""
    import numpy as np
    np.random.seed(42)
    
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    
    data = []
    for i in range(500):
        date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        region = np.random.choice(regions)
        product = np.random.choice(products)
        revenue = np.random.uniform(1000, 50000)
        units = np.random.randint(1, 500)
        customer_id = f'CUST{np.random.randint(1000, 9999):04d}'
        
        data.append({
            'order_date': date.strftime('%Y-%m-%d'),
            'region': region,
            'product': product,
            'revenue': round(revenue, 2),
            'units_sold': units,
            'customer_id': customer_id
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Created sample data at {filepath}")

# ============================================================
# CONSOLIDATED MULTI-TENANT API DATA ENDPOINTS (DRY ROUTING)
# ============================================================

@app.route("/api/<data_type>")
@login_required
def api_get_tenant_data(data_type):
    """
    Consolidated dynamic API endpoint to retrieve organization-isolated data.
    Using dynamic routing (<data_type>) removes code duplication and makes it beginner-friendly.
    """
    table_mapping = {
        'insights': 'insights',
        'sales': 'sales_clean',
        'forecasts': 'forecasts',
        'anomalies': 'anomalies',
        'segments': 'customer_segments'
    }
    
    table_name = table_mapping.get(data_type.lower())
    if not table_name:
        return jsonify({'status': 'error', 'message': f"Endpoint '/api/{data_type}' not found"}), 404
        
    org_id = session['org_id']
    df = get_table_as_df(table_name, org_id)
    return jsonify({'status': 'success', 'count': len(df), 'data': safe_json(df)})

@app.route("/api/kpis")
@login_required
def api_kpis():
    org_id = session['org_id']
    return jsonify({'status': 'success', 'data': get_kpi_data(org_id)})

@app.route("/api/revenue-by-region")
@login_required
def api_revenue_by_region():
    org_id = session['org_id']
    df = get_table_as_df('sales_clean', org_id)
    if len(df) == 0: return jsonify({'status': 'success', 'data': []})
    grouped = df.groupby('region')['revenue'].sum().reset_index()
    return jsonify({'status': 'success', 'data': safe_json(grouped)})

@app.route("/api/revenue-trend")
@login_required
def api_revenue_trend():
    org_id = session['org_id']
    df = get_table_as_df('sales_clean', org_id)
    if len(df) == 0: return jsonify({'status': 'success', 'data': []})
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['quarter'] = df['order_date'].dt.to_period('Q').astype(str)
    grouped = df.groupby('quarter')['revenue'].sum().reset_index()
    return jsonify({'status': 'success', 'data': safe_json(grouped)})

@app.route("/api/segment-distribution")
@login_required
def api_segment_distribution():
    org_id = session['org_id']
    df = get_table_as_df('customer_segments', org_id)
    if len(df) == 0: return jsonify({'status': 'success', 'data': []})
    grouped = df.groupby('segment').size().reset_index(name='count')
    return jsonify({'status': 'success', 'data': safe_json(grouped)})

@app.route("/api/dataset-info")
@login_required
def api_dataset_info():
    dataset_name = session.get('dataset_name', 'None')
    return jsonify({'status': 'success', 'dataset': dataset_name})

@app.route("/api/health")
def health_check():
    """Health check endpoint for deployment platforms."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\nStarting Flask server on port {port}...")
    print(f"Home:       http://localhost:{port}/")
    print(f"Login:      http://localhost:{port}/login")
    print(f"Register:   http://localhost:{port}/register")
    print(f"Upload:     http://localhost:{port}/upload")
    print(f"Dashboard:  http://localhost:{port}/dashboard")
    print(f"API Health: http://localhost:{port}/api/health")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port)