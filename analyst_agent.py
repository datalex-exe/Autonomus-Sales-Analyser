"""
analyst_agent.py - Dynamic Analysis Agent
Automatically discovers columns based on headings & types to run adaptive insights.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from database import get_table_as_df, save_anomalies, save_forecasts, save_segments

def schema_discovery(df):
    """Dynamically finds the best columns for analysis based on types and headings."""
    print("\nDiscovering schema dynamically...")
    columns = {
        'date': None,
        'numeric_target': None,  # Main target like revenue/sales
        'numeric_secondary': [], # Extra numeric targets like units_sold
        'categories': [],        # Grouping categories like region/product
        'id': None               # Unique identifier like customer_id
    }
    
    # 1. Detect Date Column
    date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'time' in col.lower():
            columns['date'] = col
            break
            
    # 2. Detect Numeric Targets (Revenue, Sales, Quantities)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Prioritize revenue/sales lookalikes as primary target
    target_keywords = ['revenue', 'sales', 'amount', 'spend', 'total']
    for col in num_cols:
        if any(kw in col.lower() for kw in target_keywords):
            columns['numeric_target'] = col
            break
    if not columns['numeric_target'] and num_cols:
        columns['numeric_target'] = num_cols[0] # Fallback to first numeric
        
    for col in num_cols:
        if col != columns['numeric_target']:
            columns['numeric_secondary'].append(col)

    # 3. Detect Categories and IDs (ignoring the structural org_id column)
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c != 'org_id']
    for col in cat_cols:
        if 'id' in col.lower() or 'key' in col.lower() or df[col].nunique() > len(df) * 0.6:
            if not columns['id']: 
                columns['id'] = col
        else:
            columns['categories'].append(col)
            
    # Fallback rules if categories or IDs weren't matched explicitly
    if not columns['id'] and len(cat_cols) > 0:
        columns['id'] = cat_cols[0]
        
    print(f"  Detected Schema Rules: {columns}")
    return columns

def compute_summary_stats(df, schema):
    print("\nComputing dynamic summary statistics...")
    stats = {}
    target = schema['numeric_target']
    
    if not target:
        print("  Error: No numeric target found for statistics calculation.")
        return stats

    # Overall Summary
    stats['overall'] = {
        f'total_{target}': df[target].sum(),
        f'avg_{target}': df[target].mean(),
        'total_records': len(df)
    }
    if schema['id']:
        stats['overall'][f'unique_{schema["id"]}'] = df[schema['id']].nunique()
    for sec in schema['numeric_secondary']:
        stats['overall'][f'total_{sec}'] = df[sec].sum()

    # Dynamic Aggregation by available categorical dimensions
    for cat in schema['categories'][:2]: # Limit to top 2 categories to prevent bloat
        agg_dict = {target: ['sum', 'mean', 'count']}
        for sec in schema['numeric_secondary']:
            agg_dict[sec] = 'sum'
        stats[f'by_{cat}'] = df.groupby(cat).agg(agg_dict).round(2)
        
    # Periodic Trend if date exists
    if schema['date']:
        df = df.copy()
        df['period'] = df[schema['date']].dt.to_period('Q') if hasattr(df[schema['date']], 'dt') else df[schema['date']]
        stats['by_period'] = df.groupby('period')[target].agg(['sum', 'mean', 'count']).round(2)
        
    print("  Summary stats computed successfully.")
    return stats

def calculate_growth_rates(df, schema):
    if not schema['date'] or not schema['numeric_target']:
        return pd.DataFrame()
    print("\nCalculating trend growth rates...")
    df = df.copy()
    target = schema['numeric_target']
    
    # Dynamic period selection based on dates span
    date_min = df[schema['date']].min()
    date_max = df[schema['date']].max()
    days_span = (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else 0
    if days_span <= 30:
        period_freq = 'D'
    elif days_span <= 180:
        period_freq = 'W'
    elif days_span <= 365:
        period_freq = 'M'
    else:
        period_freq = 'Q'
        
    df['period'] = df[schema['date']].dt.to_period(period_freq)
    
    periodic = df.groupby('period')[target].sum().reset_index()
    periodic['period_str'] = periodic['period'].astype(str)
    periodic['pop_growth'] = periodic[target].pct_change() * 100
    periodic['pop_growth'] = periodic['pop_growth'].round(2)
    print("  Growth rates calculated.")
    return periodic

def detect_anomalies(df, schema):
    target = schema['numeric_target']
    if not target:
        return pd.DataFrame()
        
    print("\nDetecting multi-variate anomalies...")
    df = df.copy()
    feature_cols = [target] + schema['numeric_secondary']
    features = df[feature_cols].fillna(0).values
    
    # Run Isolation Forest anomaly detection
    # Concept: Isolation Forest isolates anomalies by randomly selecting a feature and splitting it. 
    # Outliers (anomalies) require fewer splits to isolate than normal data points.
    # Optimization: Set n_estimators=30, max_samples=min(1000, len(features)), n_jobs=1 to reduce RAM/CPU usage on Render.
    iso_forest = IsolationForest(
        contamination=0.1, 
        random_state=42, 
        n_estimators=30, 
        max_samples=min(1000, len(features)), 
        n_jobs=1
    )
    predictions = iso_forest.fit_predict(features)  # Returns -1 for anomalies, 1 for normal data
    scores = iso_forest.decision_function(features)  # Lower score means highly anomalous
    df['anomaly_score'] = scores
    df['is_anomaly'] = predictions == -1

    df['severity'] = df['anomaly_score'].apply(lambda s: 'High' if s < -0.3 else ('Medium' if s < -0.15 else 'Low'))
    anomalies = df[df['is_anomaly']].copy()
    
    keep_cols = ([schema['date']] if schema['date'] else []) + schema['categories'] + [target, 'anomaly_score', 'severity']
    print(f"  Found {len(anomalies)} anomalies based on variables {feature_cols}")
    return anomalies[keep_cols]

def forecast_trends(df, schema):
    target = schema['numeric_target']
    if not schema['date'] or not target or not schema['categories']:
        return pd.DataFrame()
        
    print("\nForecasting targeted vectors...")
    forecasts = []
    primary_cat = schema['categories'][0] # Split forecast by primary categorical axis
    
    df = df.copy()
    
    # Dynamic period selection based on dates span
    date_min = df[schema['date']].min()
    date_max = df[schema['date']].max()
    days_span = (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else 0
    if days_span <= 30:
        period_freq = 'D'
    elif days_span <= 180:
        period_freq = 'W'
    elif days_span <= 365:
        period_freq = 'M'
    else:
        period_freq = 'Q'
        
    df['period'] = df[schema['date']].dt.to_period(period_freq)
    
    # Optimization: If primary_cat has high cardinality, looping over all groups can take a long time
    # and cause Render to timeout. We restrict forecasting to the top 10 groups by total target values.
    top_groups = df.groupby(primary_cat)[target].sum().sort_values(ascending=False).head(10).index.tolist()
    
    for group in top_groups:
        group_df = df[df[primary_cat] == group]
        periodic = group_df.groupby('period')[target].sum().reset_index()
        periodic['period_num'] = range(len(periodic))
        
        if len(periodic) >= 2:
            x = periodic['period_num'].values
            y = periodic[target].values
            n = len(x)
            
            # Linear Regression calculation using the Least Squares Method:
            # We want to find the line equation y = mx + b.
            # m (slope) tells us the average growth trend per period.
            # b (y-intercept) tells us where the trend started.
            m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            b = (np.sum(y) - m * np.sum(x)) / n
            
            # Predict the next period value
            next_x = len(x)
            predicted = max(0, m * next_x + b) # Floor values to zero so revenue is not negative
            
            # Calculate the next period's date range
            last_period = periodic['period'].iloc[-1]
            next_period = last_period + 1
            
            forecasts.append({
                'region': str(group), # Mapping primary category to default database column name
                'product': None,
                'forecast_date': str(next_period.end_time.date()),
                'predicted_revenue': round(predicted, 2),
                'lower_bound': round(predicted * 0.85, 2), # 15% lower bound estimate
                'upper_bound': round(predicted * 1.15, 2)  # 15% upper bound estimate
            })
            
    print(f"  Generated {len(forecasts)} forecasts matching dimension: {primary_cat}")
    return pd.DataFrame(forecasts)

def segment_entities(df, schema):
    target = schema['numeric_target']
    entity_id = schema['id']
    if not target or not entity_id:
        return pd.DataFrame()
        
    print(f"\nSegmenting core entities based on {entity_id}...")
    entity_spend = df.groupby(entity_id)[target].sum().reset_index()
    entity_spend.columns = ['customer_id', 'total_spend'] # Map to match standard DB output schema
    
    q25 = entity_spend['total_spend'].quantile(0.25)
    q75 = entity_spend['total_spend'].quantile(0.75)

    def assign_segment(spend):
        if spend >= q75: return 'High Value'
        elif spend >= q25: return 'Mid Value'
        else: return 'Low Value'

    entity_spend['segment'] = entity_spend['total_spend'].apply(assign_segment)
    print(f"  Segmented {len(entity_spend)} entities via distribution analysis.")
    return entity_spend

def run_analyst(org_id=None):
    print("\n" + "="*50)
    print("DYNAMIC ANALYST AGENT STARTING")
    print("="*50)
    
    active_org_id = org_id if org_id is not None else 'test_org_id'
    
    df = get_table_as_df('sales_clean', active_org_id)
    if len(df) == 0:
        print(f"No data found for Org {active_org_id}! Run Data Engineer first.")
        return None
        
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
        
    # Discover headings and types dynamically!
    schema = schema_discovery(df)
    
    results = {
        'summary_stats': compute_summary_stats(df, schema),
        'growth_rates': calculate_growth_rates(df, schema),
        'anomalies': detect_anomalies(df, schema),
        'forecasts': forecast_trends(df, schema),
        'segments': segment_entities(df, schema)
    }
    
    # Save results to base schema layout for API extraction
    save_anomalies(results['anomalies'], active_org_id)
    save_forecasts(results['forecasts'], active_org_id)
    save_segments(results['segments'], active_org_id)
    
    print("\n" + "="*50)
    print("DYNAMIC ANALYST AGENT COMPLETE")
    print("="*50)
    return results

if __name__ == "__main__":
    results = run_analyst()
