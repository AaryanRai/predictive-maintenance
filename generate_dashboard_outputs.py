"""
Generate Dashboard-Ready Outputs
Delta Industries Ltd.

This script creates a clean CSV file with predictions and risk categories
suitable for integration with:
- Streamlit dashboards
- Power BI reports
- Tableau visualizations
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data(filename="outputs/delta_industries_machine_health.csv"):
    """Load the dataset."""
    df = pd.read_csv(filename)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    return df

def load_models():
    """Load trained models and scalers."""
    rul_model = joblib.load('outputs/rul_model.pkl')
    rul_scaler = joblib.load('outputs/rul_scaler.pkl')
    failure_model = joblib.load('outputs/failure_risk_model.pkl')
    failure_scaler = joblib.load('outputs/failure_risk_scaler.pkl')
    return rul_model, rul_scaler, failure_model, failure_scaler

def prepare_features(df):
    """Prepare feature matrix (same as in modeling.py)."""
    df_encoded = pd.get_dummies(df, columns=['machine_type'], prefix='type', drop_first=True)
    exclude_cols = ['machine_id', 'snapshot_date', 'RUL_days', 'fail_in_30d']
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    X = df_encoded[feature_cols]
    return X, feature_cols

def assign_risk_category(rul_days, failure_probability):
    """
    Assign risk category based on RUL and failure probability.
    
    Business Logic:
    - RED: Urgent action (high probability OR low RUL)
    - YELLOW: Schedule maintenance (moderate risk)
    - GREEN: Monitor only (low risk)
    """
    if failure_probability >= 0.7 or rul_days <= 30:
        return "RED"
    elif failure_probability >= 0.4 or rul_days <= 60:
        return "YELLOW"
    else:
        return "GREEN"

def generate_dashboard_outputs():
    """
    Generate dashboard-ready CSV with predictions and risk categories.
    """
    print("=" * 60)
    print("GENERATING DASHBOARD OUTPUTS")
    print("Delta Industries Ltd.")
    print("=" * 60)
    
    # Load data
    print("\nLoading dataset...")
    df = load_data()
    
    # Load models
    print("Loading trained models...")
    rul_model, rul_scaler, failure_model, failure_scaler = load_models()
    
    # Prepare features
    print("Preparing features...")
    X, feature_cols = prepare_features(df)
    
    # Generate predictions
    print("Generating predictions...")
    X_scaled_rul = rul_scaler.transform(X)
    X_scaled_failure = failure_scaler.transform(X)
    
    rul_predictions = rul_model.predict(X_scaled_rul)
    failure_probabilities = failure_model.predict_proba(X_scaled_failure)[:, 1]
    
    # Ensure RUL predictions are non-negative
    rul_predictions = np.maximum(rul_predictions, 0)
    
    # Assign risk categories
    print("Assigning risk categories...")
    risk_categories = [
        assign_risk_category(rul, prob) 
        for rul, prob in zip(rul_predictions, failure_probabilities)
    ]
    
    # Create dashboard output DataFrame
    dashboard_df = pd.DataFrame({
        'machine_id': df['machine_id'].values,
        'snapshot_date': df['snapshot_date'].values,
        'RUL_days_predicted': np.round(rul_predictions, 1),
        'failure_probability': np.round(failure_probabilities, 3),
        'risk_category': risk_categories
    })
    
    # Add some context columns for dashboard
    dashboard_df['machine_type'] = df['machine_type'].values
    dashboard_df['criticality_score'] = df['criticality_score'].values
    dashboard_df['actual_RUL_days'] = df['RUL_days'].values  # For comparison in dashboard
    
    # Add explicit failure_flag column (1 = RED category, 0 = otherwise)
    dashboard_df['failure_flag'] = (dashboard_df['risk_category'] == 'RED').astype(int)
    
    # Sort by risk (RED first, then YELLOW, then GREEN)
    risk_order = {'RED': 1, 'YELLOW': 2, 'GREEN': 3}
    dashboard_df['risk_order'] = dashboard_df['risk_category'].map(risk_order)
    dashboard_df = dashboard_df.sort_values(['risk_order', 'failure_probability'], ascending=[True, False])
    dashboard_df = dashboard_df.drop('risk_order', axis=1)
    
    # Save to CSV
    output_file = 'outputs/dashboard_outputs.csv'
    dashboard_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Dashboard outputs saved to: {output_file}")
    print(f"  Shape: {dashboard_df.shape}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("DASHBOARD OUTPUT SUMMARY")
    print("="*60)
    
    print(f"\nRisk Category Distribution:")
    risk_counts = dashboard_df['risk_category'].value_counts()
    for category, count in risk_counts.items():
        pct = 100 * count / len(dashboard_df)
        print(f"  {category:10s}: {count:4d} machines ({pct:5.1f}%)")
    
    print(f"\nPrediction Statistics:")
    print(f"  Average RUL: {dashboard_df['RUL_days_predicted'].mean():.1f} days")
    print(f"  Min RUL: {dashboard_df['RUL_days_predicted'].min():.1f} days")
    print(f"  Max RUL: {dashboard_df['RUL_days_predicted'].max():.1f} days")
    print(f"  Average Failure Probability: {dashboard_df['failure_probability'].mean():.3f}")
    
    print(f"\n" + "="*60)
    print("DASHBOARD INTEGRATION GUIDE")
    print("="*60)
    
    print("\n1. STREAMLIT:")
    print("   import pandas as pd")
    print("   df = pd.read_csv('outputs/dashboard_outputs.csv')")
    print("   # Filter by risk_category, create charts, etc.")
    
    print("\n2. POWER BI:")
    print("   - Import CSV as data source")
    print("   - Create visualizations:")
    print("     * Risk category pie chart")
    print("     * RUL distribution histogram")
    print("     * Machine list sorted by risk")
    print("     * Failure probability gauge charts")
    
    print("\n3. TABLEAU:")
    print("   - Connect to CSV file")
    print("   - Create dashboard with:")
    print("     * Risk category color coding")
    print("     * RUL timeline visualization")
    print("     * Machine health scorecard")
    print("     * Interactive filters by machine_type, criticality")
    
    print("\n4. KEY VISUALIZATIONS TO CREATE:")
    print("   - Risk category distribution (pie/bar chart)")
    print("   - Machines sorted by failure probability (table)")
    print("   - RUL predictions over time (line chart)")
    print("   - Alert list (RED category machines)")
    print("   - Maintenance schedule (YELLOW category machines)")
    
    # Show sample of high-risk machines
    print("\n" + "="*60)
    print("SAMPLE: HIGH-RISK MACHINES (RED Category)")
    print("="*60)
    red_machines = dashboard_df[dashboard_df['risk_category'] == 'RED'].head(10)
    print(red_machines[['machine_id', 'machine_type', 'RUL_days_predicted', 
                        'failure_probability', 'criticality_score']].to_string(index=False))
    
    print("\n" + "="*60)
    print("DASHBOARD OUTPUT GENERATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    generate_dashboard_outputs()

