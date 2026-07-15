"""
run.py - Quick Start Script
============================
Just type: python run.py
"""

import sys
import subprocess
import webbrowser
import threading
import time

def check_and_install_packages():
    required = ['flask', 'pandas', 'numpy', 'scikit-learn', 'openpyxl', 'flask_cors', 'python_dotenv', 'requests']
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("All packages installed!")
    else:
        print("All packages ready!")

def open_browser():
    """Wait briefly for the server to start, then open the browser."""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000/')

def main():
    print("="*60)
    print("  AUTONOMOUS SALES PERFORMANCE ANALYZER ")
    print("="*60)
    print()

    print("Step 1: Checking dependencies...")
    check_and_install_packages()

    print("\nStep 2: Setting up database...")
    from database import init_database
    init_database()

    print("\nStep 3: Starting web server...")
    print()
    print("   Home:       http://localhost:5000/")
    print("   Upload:     http://localhost:5000/upload")
    print("   Dashboard:  http://localhost:5000/dashboard")
    print()
    print("   Opening your browser automatically...")
    print("   Press Ctrl+C to stop")
    print()

    import os
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        # Launch browser in a background thread so it doesn't block the server
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()

    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()