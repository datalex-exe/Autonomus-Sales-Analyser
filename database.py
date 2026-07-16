"""
database.py - Production Database Setup
Maintains strict table structures and provides multi-tenant isolation by organization.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# Import database connection and configurations
from config import get_connection, DB_DIR


def init_database():
    conn = get_connection()
    cursor = conn.cursor()

    # Dynamic schema upgrade logic:
    # If sales_clean exists but does not have `org_id` column, recreate all tables
    try:
        cursor.execute("PRAGMA table_info(sales_clean)")
        columns = [row['name'] for row in cursor.fetchall()]
        if columns and 'org_id' not in columns:
            print("Upgrading database schema for organizational multi-tenancy...")
            cursor.execute("DROP TABLE IF EXISTS sales_clean")
            cursor.execute("DROP TABLE IF EXISTS anomalies")
            cursor.execute("DROP TABLE IF EXISTS forecasts")
            cursor.execute("DROP TABLE IF EXISTS customer_segments")
            cursor.execute("DROP TABLE IF EXISTS insights")
            conn.commit()
    except Exception as e:
        print(f"Schema check error: {e}")

    # Upgrade check for users table (adding phone column support)
    try:
        cursor.execute("PRAGMA table_info(users)")
        columns = [row['name'] for row in cursor.fetchall()]
        if columns and 'phone' not in columns:
            print("Upgrading database schema for users table to add phone column...")
            cursor.execute("DROP TABLE IF EXISTS users")
            conn.commit()
    except Exception as e:
        print(f"Users table upgrade check error: {e}")

    # Organizations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS organizations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            join_code TEXT UNIQUE NOT NULL,
            created_at TEXT
        )
    """)

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT,
            role TEXT,
            phone TEXT UNIQUE,
            created_at TEXT,
            FOREIGN KEY (org_id) REFERENCES organizations(id)
        )
    """)

    # Core sales data table (Multi-tenant)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales_clean (
            org_id TEXT, order_date TEXT, region TEXT, product TEXT,
            revenue REAL, units_sold INTEGER, customer_id TEXT
        )
    """)

    # Anomalies table (Multi-tenant)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            org_id TEXT, order_date TEXT, region TEXT, product TEXT,
            revenue REAL, anomaly_score REAL, severity TEXT
        )
    """)

    # Forecasts table (Multi-tenant)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            org_id TEXT, region TEXT, product TEXT, forecast_date TEXT,
            predicted_revenue REAL, lower_bound REAL, upper_bound REAL
        )
    """)

    # Customer segments table (Multi-tenant)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_segments (
            org_id TEXT, customer_id TEXT, total_spend REAL, segment TEXT
        )
    """)

    # TEXT INSIGHTS TABLE (Multi-tenant)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS insights (
            org_id TEXT, category TEXT, insight_text TEXT, metric_value TEXT,
            region TEXT, product TEXT
        )
    """)

    # Column mappings table (Multi-tenant)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS column_mappings (
            org_id TEXT, standard_col TEXT, original_col TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized with organizational structures!")

# ============================================================
# MULTI-TENANT DATABASE HELPER (DRY PRINCIPLE)
# ============================================================

def _save_tenant_data(df, table_name, org_id, required_columns):
    """
    A helper function to clear old records for an organization (org_id)
    and save new Pandas DataFrame records to the SQLite database.
    
    This function keeps our code short and beginner-friendly by grouping 
    repeating SQL commands (DELETE, INSERT/to_sql) into a single reusable helper.
    """
    if org_id is None:
        raise ValueError("Cannot save data without a valid org_id context.")
        
    conn = get_connection()
    
    # If the DataFrame is empty, initialize it with the required column structure
    if df.empty:
        df = pd.DataFrame(columns=required_columns)
    else:
        df = df.copy()
        
    # Inject the organization ID for multi-tenant isolation
    df['org_id'] = org_id
    
    # Ensure all required columns exist (filling missing ones with None/NaN)
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
            
    # Keep only the columns matching the database schema
    final_df = df[['org_id'] + required_columns]
    
    # Delete previous database entries for this organization in this table
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table_name} WHERE org_id = ?", (org_id,))
    conn.commit()
    
    # Save the new DataFrame records into SQLite
    final_df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()
    return len(final_df)


# ============================================================
# MULTI-TENANT SALES DATA WRAPPERS
# ============================================================

def save_cleaned_data(df, org_id):
    """Saves standardized sales records after Data Engineering cleaning."""
    cols = ['order_date', 'region', 'product', 'revenue', 'units_sold', 'customer_id']
    count = _save_tenant_data(df, 'sales_clean', org_id, cols)
    print(f"Saved {count} cleaned records for Org: {org_id}")

def save_anomalies(df, org_id):
    """Saves transaction anomalies detected during statistical runs."""
    cols = ['order_date', 'region', 'product', 'revenue', 'anomaly_score', 'severity']
    count = _save_tenant_data(df, 'anomalies', org_id, cols)
    print(f"Saved {count} anomalies for Org: {org_id}")

def save_forecasts(df, org_id):
    """Saves dynamic linear predictions calculated by regional groups."""
    cols = ['region', 'product', 'forecast_date', 'predicted_revenue', 'lower_bound', 'upper_bound']
    count = _save_tenant_data(df, 'forecasts', org_id, cols)
    print(f"Saved {count} forecasts for Org: {org_id}")

def save_segments(df, org_id):
    """Saves customer spend classifications (High/Mid/Low value)."""
    cols = ['customer_id', 'total_spend', 'segment']
    count = _save_tenant_data(df, 'customer_segments', org_id, cols)
    print(f"Saved {count} segments for Org: {org_id}")

def save_insights(insights_list, org_id):
    """Saves human-readable report summaries compiled by the Insight Agent."""
    df = pd.DataFrame(insights_list)
    cols = ['category', 'insight_text', 'metric_value', 'region', 'product']
    count = _save_tenant_data(df, 'insights', org_id, cols)
    print(f"Saved {count} business insights for Org: {org_id}")

def save_column_mappings(mapping, org_id):
    """Saves the dynamic mapping from standard internal names to original headers."""
    if org_id is None:
        return
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM column_mappings WHERE org_id = ?", (org_id,))
    for orig, std in mapping.items():
        cursor.execute("INSERT INTO column_mappings (org_id, standard_col, original_col) VALUES (?, ?, ?)", (org_id, std, orig))
    conn.commit()
    conn.close()
    print(f"Saved {len(mapping)} column mappings for Org: {org_id}")

def get_column_mappings(org_id):
    """Retrieves the mapping dictionary {standard_col: original_col} for an organization."""
    if org_id is None:
        return {}
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT standard_col, original_col FROM column_mappings WHERE org_id = ?", (org_id,))
    rows = cursor.fetchall()
    conn.close()
    return {row['standard_col']: row['original_col'] for row in rows}

def get_table_as_df(table_name, org_id=None):
    conn = get_connection()
    df = None
    try:
        if org_id is not None:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE org_id = ?", conn, params=(org_id,))
        else:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception as e:
        print(f"Error fetching {table_name}: {e}")
        df = pd.DataFrame()
    conn.close()
    
    if df is None:
        df = pd.DataFrame()
    return df

# ============================================================
# USER & ORGANIZATION MANAGEMENT OPERATIONS
# ============================================================

def create_organization(name):
    conn = get_connection()
    cursor = conn.cursor()
    
    org_id = str(uuid.uuid4())
    # Generate unique 6 character join code
    import random
    import string
    
    while True:
        join_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        # Ensure unique join_code
        cursor.execute("SELECT id FROM organizations WHERE join_code = ?", (join_code,))
        if not cursor.fetchone():
            break
            
    created_at = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO organizations (id, name, join_code, created_at) VALUES (?, ?, ?, ?)",
        (org_id, name, join_code, created_at)
    )
    conn.commit()
    conn.close()
    return org_id, join_code

def get_organization_by_join_code(join_code):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM organizations WHERE join_code = ?", (join_code.strip().upper(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None
 
def get_organization_by_id(org_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM organizations WHERE id = ?", (org_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_all_organizations():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM organizations ORDER BY name ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def create_user(org_id, email, password, name, phone, role='user'):
    conn = get_connection()
    cursor = conn.cursor()
    
    user_id = str(uuid.uuid4())
    password_hash = generate_password_hash(password)
    created_at = datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO users (id, org_id, email, password_hash, name, role, phone, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, org_id, email.strip().lower(), password_hash, name, role, phone.strip(), created_at)
    )
    conn.commit()
    conn.close()
    return user_id

def get_user_by_email(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_user_by_phone(phone):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE phone = ?", (phone.strip(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

if __name__ == "__main__":
    init_database()
