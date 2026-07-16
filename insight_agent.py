"""
insight_agent.py - Dynamic Insight Generation Agent

This agent takes statistical outputs (totals, growth trends, anomalies, forecasts) 
and translates them into simple, human-readable business English sentences.
This teaches beginners how to map complex numerical metrics to natural language outputs.
"""

import pandas as pd
from database import get_table_as_df, save_insights

def generate_revenue_insights(summary_stats, growth_rates):
    insights = []
    if summary_stats is None:
        summary_stats = {}
    if growth_rates is None:
        growth_rates = pd.DataFrame()
    if not summary_stats:
        return insights
        
    # 1. Dynamic Overall Target Metrics
    overall = summary_stats.get('overall', {})
    if not overall:
        return insights
        
    target_key = next((k for k in overall.keys() if 'total_' in k), None)
    
    if target_key:
        clean_name = target_key.replace('total_', '').replace('_', ' ').title()
        total_val = overall.get(target_key, 0) or 0
        insights.append({
            'category': 'revenue',
            'insight_text': f"Total {clean_name} across all tracked segments reached ${total_val:,.2f}.",
            'metric_value': f"${total_val:,.2f}",
            'region': None, 'product': None
        })

    # 2. Dynamic Growth Rates
    if not growth_rates.empty and len(growth_rates) > 1:
        latest = growth_rates.iloc[-1]
        growth_col = next((c for c in growth_rates.columns if 'growth' in c), None)
        period_col = next((c for c in growth_rates.columns if 'period' in c), None)
        
        if growth_col and period_col and pd.notna(latest.get(growth_col)):
            direction = "grew" if latest[growth_col] > 0 else "declined"
            insights.append({
                'category': 'revenue',
                'insight_text': f"Performance {direction} by {abs(latest[growth_col]):.1f}% in period {latest[period_col]} compared to the previous timeframe.",
                'metric_value': f"{latest[growth_col]:+.1f}%",
                'region': None, 'product': None
            })

    # 3. Dynamic Categorical Top Performers
    for key, df_group in summary_stats.items():
        if df_group is not None and not isinstance(df_group, dict) and key.startswith('by_') and key != 'by_period' and not df_group.empty:
            cat_name = key.replace('by_', '').replace('_', ' ').title()
            
            # Extract first column level cleanly
            try:
                if isinstance(df_group.columns, pd.MultiIndex):
                    sum_col = [c for c in df_group.columns if 'sum' in c[1]][0]
                    top_performer = df_group[sum_col].idxmax()
                    top_value = df_group[sum_col].max()
                else:
                    sum_col = [c for c in df_group.columns if 'sum' in c or 'revenue' in c][0]
                    top_performer = df_group[sum_col].idxmax()
                    top_value = df_group[sum_col].max()

                insights.append({
                    'category': 'revenue',
                    'insight_text': f"The highest contributing feature for {cat_name} is '{top_performer}' yielding ${top_value:,.2f}.",
                    'metric_value': f"${top_value:,.2f}",
                    'region': str(top_performer), 'product': None
                })
            except Exception:
                continue # Gracefully skip if structure is too nested
                
    return insights

def generate_anomaly_insights(anomalies_df):
    insights = []
    if anomalies_df is None or len(anomalies_df) == 0:
        insights.append({
            'category': 'anomaly',
            'insight_text': "Data scanning complete. No major structural anomalies detected.",
            'metric_value': "0 anomalies",
            'region': None, 'product': None
        })
        return insights
        
    high_count = len(anomalies_df[anomalies_df['severity'] == 'High']) if 'severity' in anomalies_df.columns else 0
    if high_count > 0:
        insights.append({
            'category': 'anomaly',
            'insight_text': f"Alert: {high_count} high-severity anomaly patterns detected requiring inspection.",
            'metric_value': f"{high_count} high",
            'region': None, 'product': None
        })
        
    # Dynamically pick available text groupings for tracking anomalies
    text_cols = [c for c in anomalies_df.select_dtypes(include=['object']).columns if c != 'severity']
    if text_cols and len(anomalies_df) > 0:
        focus_col = text_cols[0]
        val_counts = anomalies_df[focus_col].value_counts()
        if not val_counts.empty:
            top_anomaly_group = val_counts.index[0]
            count = val_counts.iloc[0]
            insights.append({
                'category': 'anomaly',
                'insight_text': f"Category group '{top_anomaly_group}' demonstrates the highest density of unusual variance with {count} flagged rows.",
                'metric_value': f"{count} flags",
                'region': str(top_anomaly_group), 'product': None
            })
    return insights

def generate_forecast_insights(forecasts_df):
    insights = []
    if forecasts_df is None or len(forecasts_df) == 0:
        return insights
    for _, row in forecasts_df.iterrows():
        region = row.get('region', 'Target Vector')
        predicted = row.get('predicted_revenue', 0)
        lower = row.get('lower_bound', 0)
        upper = row.get('upper_bound', 0)
        insights.append({
            'category': 'forecast',
            'insight_text': f"Projected trajectory for '{region}' scales to ${predicted:,.2f} next phase (Confidence band: ${lower:,.2f} to ${upper:,.2f}).",
            'metric_value': f"${predicted:,.2f}",
            'region': str(region), 'product': None
        })
    return insights

def generate_segment_insights(segments_df):
    insights = []
    if segments_df is None or len(segments_df) == 0:
        return insights
        
    if 'segment' not in segments_df.columns:
        return insights
        
    segment_counts = segments_df['segment'].value_counts()
    total_customers = len(segments_df)
    
    for segment, count in segment_counts.items():
        pct = (count / total_customers) * 100 if total_customers > 0 else 0
        avg_spend = segments_df[segments_df['segment'] == segment]['total_spend'].mean()
        if pd.isna(avg_spend):
            avg_spend = 0.0
        insights.append({
            'category': 'segment',
            'insight_text': f"The '{segment}' classification represents {pct:.0f}% of your distribution base with a mean value weight of ${avg_spend:,.2f}.",
            'metric_value': f"{pct:.0f}%",
            'region': None, 'product': None
        })
    return insights

def run_insight_agent(analysis_results=None, org_id=None):
    print("\n" + "="*50)
    print("DYNAMIC INSIGHT AGENT STARTING")
    print("="*50)

    active_org_id = org_id if org_id is not None else 'test_org_id'
    all_insights = []

    if analysis_results:
        summary_stats = analysis_results.get('summary_stats', {}) or {}
        growth_rates = analysis_results.get('growth_rates', pd.DataFrame())
        if growth_rates is None: growth_rates = pd.DataFrame()
        anomalies = analysis_results.get('anomalies', pd.DataFrame())
        if anomalies is None: anomalies = pd.DataFrame()
        forecasts = analysis_results.get('forecasts', pd.DataFrame())
        if forecasts is None: forecasts = pd.DataFrame()
        segments = analysis_results.get('segments', pd.DataFrame())
        if segments is None: segments = pd.DataFrame()
    else:
        summary_stats = {}
        growth_rates = pd.DataFrame()
        anomalies = get_table_as_df('anomalies', active_org_id)
        forecasts = get_table_as_df('forecasts', active_org_id)
        segments = get_table_as_df('customer_segments', active_org_id)
        
        sales_df = get_table_as_df('sales_clean', active_org_id)
        if sales_df is not None and len(sales_df) > 0:
            summary_stats = {
                'overall': {f'total_revenue': sales_df['revenue'].sum()},
                'by_region': sales_df.groupby('region').size().to_frame()
            }

    all_insights.extend(generate_revenue_insights(summary_stats, growth_rates))
    all_insights.extend(generate_anomaly_insights(anomalies))
    all_insights.extend(generate_forecast_insights(forecasts))
    all_insights.extend(generate_segment_insights(segments))

    if all_insights:
        save_insights(all_insights, active_org_id)

    print(f"\nSuccessfully built {len(all_insights)} text business insights matrices for Org {active_org_id}.")
    print("="*50)
    
    return all_insights

if __name__ == "__main__":
    insights = run_insight_agent()
