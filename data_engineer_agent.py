"""
data_engineer_agent.py - Dynamic Data Cleaning Agent
Automatically detects column intent from headings, cleans, and standardizes it.
"""

import pandas as pd
import numpy as np
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
        ('order_date', ['date', 'time', 'timestamp', 'dt']),
        ('revenue', ['revenue', 'sales', 'amount', 'spend', 'total', 'price', 'gross', 'amt', 'rate']),
        ('units_sold', ['unit', 'sold', 'qty', 'quantity', 'count', 'pcs', 'pieces', 'qty_sold']),
        ('customer_id', ['id', 'cust', 'customer', 'client', 'key', 'buyer'])
    ]
    
    # Loop over keywords first to match high priority terms first
    for target, keywords in search_criteria:
        found = False
        for kw in keywords:
            for original, clean in raw_cols.items():
                if original not in mapping and kw in clean:
                    mapping[original] = target
                    found = True
                    break
            if found:
                break
 
    # 2. Map remaining columns.
    # Text/categorical columns go to 'region' and 'product' slots.
    # Numeric or other columns go to category_X slots.
    remaining_cols = [orig for orig in df.columns if orig not in mapping]
    
    remaining_text_cols = []
    remaining_other_cols = []
    
    time_keywords = ['date', 'time', 'timestamp', 'month', 'year', 'day', 'quarter']
    
    for orig in remaining_cols:
        clean_orig = orig.lower().strip().replace(' ', '_')
        if (pd.api.types.is_numeric_dtype(df[orig]) or 
            any(tk in clean_orig for tk in time_keywords) or 
            df[orig].nunique() <= 1):
            remaining_other_cols.append(orig)
        else:
            remaining_text_cols.append(orig)
            
    categorical_slots = ['region', 'product']
    cat_mapped_count = 0
    
    # Map remaining text columns to categorical slots
    for orig in remaining_text_cols:
        if cat_mapped_count < len(categorical_slots):
            mapping[orig] = categorical_slots[cat_mapped_count]
            cat_mapped_count += 1
        else:
            mapping[orig] = f"category_{len(mapping)}"
            
    # Map remaining other (numeric/etc) columns to category_X slots
    for orig in remaining_other_cols:
        mapping[orig] = f"category_{len(mapping)}"

    print(f"  Mapped incoming headings: {mapping}")
    return mapping

def clean_data(df):
    print("\nStarting dynamic data cleaning...")
    df = df.copy()

    # Dynamic Field Mapping Strategy
    header_mapping = discover_and_map_headers(df)
    df = df.rename(columns=header_mapping)
    print("  Dynamically standardized column structure.")

    # 1. IMPUTE ORDER_DATE
    if 'order_date' not in df.columns:
        # Generate N dates spread over the last 365 days
        print("  Missing 'order_date'. Imputing distributed dates over past year...")
        df['order_date'] = pd.date_range(end=pd.Timestamp.now().normalize(), periods=len(df), freq='h' if len(df) > 365 else 'D')[:len(df)]
    else:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        
        # Coerce outlier dates (e.g., style codes JAN8641 parsed as year 8641) to NaT
        valid_year = df['order_date'].dt.year.between(2000, 2100)
        df.loc[~valid_year, 'order_date'] = pd.NaT
        
        # If all values are NaT/NaN:
        if df['order_date'].isna().all():
            print("  All values in 'order_date' are NaT. Imputing distributed dates over past year...")
            df['order_date'] = pd.date_range(end=pd.Timestamp.now().normalize(), periods=len(df), freq='h' if len(df) > 365 else 'D')[:len(df)]
        else:
            # Impute individual missing dates via forward/backward fill
            df['order_date'] = df['order_date'].ffill().bfill()
            # If any NaT remains, fill with today
            df['order_date'] = df['order_date'].fillna(pd.Timestamp.now().normalize())

    # 2. IMPUTE REVENUE
    if 'revenue' not in df.columns:
        # Try to find any numerical column to act as revenue (excluding units_sold)
        num_cols = df.select_dtypes(include=['number']).columns
        num_cols = [c for c in num_cols if c != 'units_sold']
        if num_cols:
            df = df.rename(columns={num_cols[0]: 'revenue'})
            print(f"  Mapped numeric column '{num_cols[0]}' to 'revenue'.")
        elif 'units_sold' in df.columns:
            df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
            df['revenue'] = df['units_sold'].fillna(1).astype(float) * 50.0 + np.random.uniform(5.0, 20.0, size=len(df))
            print("  Imputed 'revenue' from 'units_sold' (assuming average item price is $50).")
        else:
            np.random.seed(42)
            df['revenue'] = np.random.uniform(100.0, 5000.0, size=len(df))
            print("  No numerical columns found. Imputed random 'revenue' values between $100 and $5000.")
    else:
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        # Fill NaNs with column mean, fallback to 100.0
        mean_rev = df['revenue'].mean()
        if pd.isna(mean_rev) or mean_rev <= 0:
            mean_rev = 100.0
        df['revenue'] = df['revenue'].fillna(mean_rev)
        # Ensure positive values (replace negative/zero with absolute value or mean_rev)
        df['revenue'] = df['revenue'].apply(lambda val: abs(val) if val != 0 else mean_rev)

    # 3. IMPUTE UNITS_SOLD
    if 'units_sold' not in df.columns:
        # Impute based on revenue
        df['units_sold'] = (df['revenue'] / 50.0).fillna(1).astype(int).clip(1)
        print("  Missing 'units_sold'. Imputed based on 'revenue'.")
    else:
        df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
        # Fill NaNs
        df['units_sold'] = df['units_sold'].fillna((df['revenue'] / 50.0).fillna(1).astype(int).clip(1))
        # Ensure positive non-zero units
        df['units_sold'] = df['units_sold'].apply(lambda val: max(1, int(abs(val))) if not pd.isna(val) else 1).astype('Int64')

    # 4. IMPUTE REGION
    regions_list = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    if 'region' not in df.columns:
        df['region'] = [regions_list[i % len(regions_list)] for i in range(len(df))]
        print("  Missing 'region'. Imputed distributed regional groups sequentially.")
    else:
        df['region'] = df['region'].astype(str).str.strip()
        # If all values are NaN or 'nan' or empty:
        if df['region'].isin(['nan', '', 'None', 'NoneType']).all():
            df['region'] = [regions_list[i % len(regions_list)] for i in range(len(df))]
        else:
            df['region'] = df['region'].replace(['nan', '', 'None', 'NoneType'], np.nan).ffill().bfill()
            df['region'] = df['region'].fillna('Global')

    # 5. IMPUTE PRODUCT
    products_list = ['Product A', 'Product B', 'Product C', 'Product D']
    if 'product' not in df.columns:
        df['product'] = [products_list[i % len(products_list)] for i in range(len(df))]
        print("  Missing 'product'. Imputed distributed product groups sequentially.")
    else:
        df['product'] = df['product'].astype(str).str.strip()
        if df['product'].isin(['nan', '', 'None', 'NoneType']).all():
            df['product'] = [products_list[i % len(products_list)] for i in range(len(df))]
        else:
            df['product'] = df['product'].replace(['nan', '', 'None', 'NoneType'], np.nan).ffill().bfill()
            df['product'] = df['product'].fillna('General')

    # 6. IMPUTE CUSTOMER_ID
    if 'customer_id' not in df.columns:
        df['customer_id'] = [f'CUST-{1000 + i}' for i in range(len(df))]
        print("  Missing 'customer_id'. Imputed sequential customer keys.")
    else:
        df['customer_id'] = df['customer_id'].astype(str).str.strip()
        if df['customer_id'].isin(['nan', '', 'None', 'NoneType']).all():
            df['customer_id'] = [f'CUST-{1000 + i}' for i in range(len(df))]
        else:
            df['customer_id'] = df['customer_id'].replace(['nan', '', 'None', 'NoneType'], np.nan)
            # Create a fallback sequence generator for null values
            fallback_custs = [f'CUST-{1000 + i}' for i in range(len(df))]
            df['customer_id'] = df['customer_id'].fillna(pd.Series(fallback_custs))

    # Convert Types and validate safely
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce').astype('Int64')

    # Ensure no rows have null order_date or revenue
    df['order_date'] = df['order_date'].fillna(pd.Timestamp.now().normalize())
    df['revenue'] = df['revenue'].fillna(100.0)

    # Double check logical bounds
    df.loc[df['revenue'] <= 0, 'revenue'] = 100.0
    # Adjust old order dates if any to be in a valid range starting at '2000-01-01'
    cutoff = pd.to_datetime('2000-01-01')
    df.loc[df['order_date'] < cutoff, 'order_date'] = pd.Timestamp.now().normalize()

    # Drop missing critical targets
    before = len(df)
    df = df.dropna(subset=['order_date', 'revenue'])
    print(f"  Removed {before - len(df)} rows missing Date or Revenue indicators.")

    # Remove duplicates safely
    before = len(df)
    unique_check_cols = [c for c in ['order_date', 'region', 'product', 'customer_id', 'revenue'] if c in df.columns]
    df = df.drop_duplicates(subset=unique_check_cols)
    print(f"  Removed {before - len(df)} duplicate row entries.")

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
    
    active_org_id = org_id if org_id is not None else 'test_org_id'
    
    # Discover and save original column mappings to the database
    header_mapping = discover_and_map_headers(raw_df)
    from database import save_column_mappings
    save_column_mappings(header_mapping, active_org_id)
    
    cleaned_df = clean_data(raw_df)
    
    # Persist the cleaned structure to sqlite standard framework tables
    save_cleaned_data(cleaned_df, active_org_id)
    return cleaned_df

if __name__ == "__main__":
    df = run_data_engineer('sample_sales.csv')
