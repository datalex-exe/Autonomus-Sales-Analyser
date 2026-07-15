"""
data_engineer_agent.py - Dynamic Data Cleaning Agent
Automatically detects column intent from headings, cleans, and standardizes it.
"""

import pandas as pd
from database import save_cleaned_data

def discover_and_map_headers(df):
    """
    Scans the raw DataFrame columns and maps them to standard internal 
    names based on keywords and data type properties.
    """
    print("\nDiscovering file structure and headers...")
    
    # Lowercase and strip whitespace to easily match keywords
    raw_cols = {col: col.lower().strip().replace(' ', '_') for col in df.columns}
    
    # Internal targets we want to map to
    mapping = {}
    
    # 1. Define matching criteria (target column name, mapping keywords)
    search_criteria = [
        ('order_date', ['date', 'time', 'timestamp']),
        ('revenue', ['revenue', 'sales', 'amount', 'spend', 'total', 'price']),
        ('units_sold', ['unit', 'sold', 'qty', 'quantity', 'count']),
        ('customer_id', ['id', 'cust', 'customer', 'client', 'key'])
    ]
    
    # Loop over criteria to map matching column names
    for target, keywords in search_criteria:
        for original, clean in raw_cols.items():
            if original not in mapping and any(kw in clean for kw in keywords):
                mapping[original] = target
                break

    # 5. Map the remaining object/text columns to categorical slots (Region / Product)
    remaining_text_cols = [orig for orig, clean in raw_cols.items() if orig not in mapping]
    categorical_slots = ['region', 'product']
    
    for i, orig in enumerate(remaining_text_cols):
        if i < len(categorical_slots):
            mapping[orig] = categorical_slots[i]
        else:
            # Fallback for extra text columns
            mapping[orig] = f"category_{i-1}"

    print(f"  Mapped incoming headings: {mapping}")
    return mapping

def clean_data(df):
    print("\nStarting dynamic data cleaning...")
    df = df.copy()

    # Dynamic Field Mapping Strategy
    header_mapping = discover_and_map_headers(df)
    df = df.rename(columns=header_mapping)
    print("  Dynamically standardized column structure.")

    # Core required fields fallback: If we couldn't match basic requirements, inject empty/dummy arrays
    if 'order_date' not in df.columns:
        df['order_date'] = pd.Timestamp.now().normalize()
    if 'revenue' not in df.columns:
        # Try to find any numerical column to act as revenue
        num_cols = df.select_dtypes(include=['number']).columns
        if not num_cols.empty:
            df = df.rename(columns={num_cols[0]: 'revenue'})
        else:
            df['revenue'] = 0.0

    # Ensure missing database core framework columns exist
    for fallback_col, default_val in [('units_sold', 1), ('region', 'Global'), ('product', 'General'), ('customer_id', 'UNKNOWN')]:
        if fallback_col not in df.columns:
            df[fallback_col] = default_val

    # Convert Types and validate safely
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce').astype('Int64')

    # Drop missing critical targets
    before = len(df)
    df = df.dropna(subset=['order_date', 'revenue'])
    print(f"  Removed {before - len(df)} rows missing Date or Revenue indicators.")

    # Remove duplicates safely
    before = len(df)
    unique_check_cols = [c for c in ['order_date', 'region', 'product', 'customer_id', 'revenue'] if c in df.columns]
    df = df.drop_duplicates(subset=unique_check_cols)
    print(f"  Removed {before - len(df)} duplicate row entries.")

    # Validate logical metric bound parameters
    df = df[df['revenue'] > 0]
    df = df[df['order_date'] >= '2000-01-01']

    # Final Text Standardization
    df['region'] = df['region'].astype(str).str.strip().str.title()
    df['product'] = df['product'].astype(str).str.strip().str.title()
    df['customer_id'] = df['customer_id'].astype(str).str.strip().str.upper()

    df = df.reset_index(drop=True)
    print(f"\nCleaning complete! {len(df)} records matched seamlessly to pipeline framework.")
    return df

def run_data_engineer(filepath, org_id=None):
    # Determine loader type dynamically
    if filepath.endswith('.csv'):
        raw_df = pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        raw_df = pd.read_excel(filepath)
    else:
        raise ValueError("File format not supported. Must be .csv or Excel format.")
        
    print(f"Loaded {len(raw_df)} rows from {filepath}")
    cleaned_df = clean_data(raw_df)
    
    # Persist the cleaned structure to sqlite standard framework tables
    if org_id is not None:
        save_cleaned_data(cleaned_df, org_id)
    else:
        save_cleaned_data(cleaned_df, 'test_org_id')
    return cleaned_df

if __name__ == "__main__":
    df = run_data_engineer('sample_sales.csv')