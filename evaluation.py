"""
Model Evaluation and Business Interpretation
Delta Industries Ltd.

This script provides:
- Detailed model performance analysis
- Business interpretation of results
- Cost-benefit analysis
- Actionable insights for maintenance teams
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

def evaluate_rul_model(df, rul_model, rul_scaler, feature_cols):
    """
    Evaluate RUL model and provide business interpretation.
    """
    print("\n" + "="*60)
    print("RUL MODEL EVALUATION & BUSINESS INTERPRETATION")
    print("="*60)
    
    X, _ = prepare_features(df)
    X_scaled = rul_scaler.transform(X)
    
    # Predictions
    rul_predictions = rul_model.predict(X_scaled)
    actual_rul = df['RUL_days'].values
    
    # Overall metrics
    mae = mean_absolute_error(actual_rul, rul_predictions)
    rmse = np.sqrt(np.mean((actual_rul - rul_predictions) ** 2))
    
    print(f"\nOverall Performance:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} days")
    
    # Performance by RUL range
    print(f"\n" + "-"*60)
    print("Performance by RUL Range (Critical for Business)")
    print("-"*60)
    
    ranges = [
        (0, 30, "Critical (0-30 days)"),
        (30, 60, "High Priority (30-60 days)"),
        (60, 90, "Medium Priority (60-90 days)"),
        (90, float('inf'), "Low Priority (90+ days)")
    ]
    
    for min_rul, max_rul, label in ranges:
        mask = (actual_rul >= min_rul) & (actual_rul < max_rul)
        if mask.sum() > 0:
            range_mae = mean_absolute_error(actual_rul[mask], rul_predictions[mask])
            print(f"  {label:30s}: MAE = {range_mae:.2f} days (n={mask.sum()})")
    
    # Business interpretation
    print(f"\n" + "-"*60)
    print("BUSINESS INTERPRETATION")
    print("-"*60)
    print("1. RUL Prediction Accuracy:")
    print(f"   - Average prediction error: {mae:.1f} days")
    print("   - This means if a machine is predicted to fail in 45 days,")
    print(f"     the actual failure will likely occur between {45-mae:.0f} and {45+mae:.0f} days")
    print("\n2. Planning Horizon:")
    print("   - Machines with RUL > 60 days: Schedule maintenance in next month")
    print("   - Machines with RUL 30-60 days: Schedule maintenance within 2 weeks")
    print("   - Machines with RUL < 30 days: URGENT - Schedule maintenance immediately")
    print("\n3. Resource Allocation:")
    print("   - Use RUL predictions to batch maintenance activities")
    print("   - Optimize technician schedules based on predicted failure dates")
    print("   - Order spare parts in advance for machines with low RUL")
    
    return rul_predictions

def evaluate_failure_risk_model(df, failure_model, failure_scaler, feature_cols):
    """
    Evaluate failure risk model and provide business interpretation.
    """
    print("\n" + "="*60)
    print("FAILURE RISK MODEL EVALUATION & BUSINESS INTERPRETATION")
    print("="*60)
    
    X, _ = prepare_features(df)
    X_scaled = failure_scaler.transform(X)
    
    # Predictions
    failure_proba = failure_model.predict_proba(X_scaled)[:, 1]
    failure_pred = failure_model.predict(X_scaled)
    actual_failure = df['fail_in_30d'].values
    
    # Metrics
    auc = roc_auc_score(actual_failure, failure_proba)
    precision = precision_score(actual_failure, failure_pred)
    recall = recall_score(actual_failure, failure_pred)
    cm = confusion_matrix(actual_failure, failure_pred)
    
    print(f"\nOverall Performance:")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    # Confusion matrix interpretation
    print(f"\n" + "-"*60)
    print("CONFUSION MATRIX INTERPRETATION")
    print("-"*60)
    print(f"                Predicted")
    print(f"              Low Risk  High Risk")
    print(f"Actual Low Risk   {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"Actual High Risk  {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    print(f"\nBusiness Impact:")
    print(f"  True Positives (TP): {tp} - Correctly identified failures")
    print(f"    → Action: Schedule maintenance, prevent downtime")
    print(f"  False Positives (FP): {fp} - Unnecessary maintenance alerts")
    print(f"    → Cost: Maintenance cost without preventing failure")
    print(f"  False Negatives (FN): {fn} - Missed failures")
    print(f"    → Cost: Unplanned downtime, emergency repairs")
    print(f"  True Negatives (TN): {tn} - Correctly identified safe machines")
    print(f"    → Benefit: No unnecessary maintenance")
    
    # Cost-benefit analysis
    print(f"\n" + "-"*60)
    print("COST-BENEFIT ANALYSIS (Hypothetical)")
    print("-"*60)
    print("Assumptions:")
    print("  - Planned maintenance cost: $2,000 per machine")
    print("  - Unplanned downtime cost: $10,000 per machine")
    print("  - Emergency repair cost: $5,000 per machine")
    
    planned_maintenance_cost = (tp + fp) * 2000
    prevented_downtime_savings = tp * 10000
    missed_failures_cost = fn * (10000 + 5000)  # Downtime + emergency repair
    
    net_benefit = prevented_downtime_savings - planned_maintenance_cost - missed_failures_cost
    
    print(f"\nCosts:")
    print(f"  Planned maintenance (TP + FP): ${planned_maintenance_cost:,}")
    print(f"  Missed failures (FN): ${missed_failures_cost:,}")
    print(f"\nSavings:")
    print(f"  Prevented downtime (TP): ${prevented_downtime_savings:,}")
    print(f"\nNet Benefit: ${net_benefit:,}")
    
    if net_benefit > 0:
        print(f"\n✓ Model provides positive ROI")
    else:
        print(f"\n⚠ Model needs optimization (adjust threshold or improve accuracy)")
    
    # Threshold recommendation
    print(f"\n" + "-"*60)
    print("THRESHOLD RECOMMENDATION")
    print("-"*60)
    print("Current threshold: 0.5 (50% probability)")
    print("\nThreshold selection strategy:")
    print("  - For critical machines: Use lower threshold (0.3-0.4)")
    print("    → Prioritize recall, minimize missed failures")
    print("  - For non-critical machines: Use higher threshold (0.6-0.7)")
    print("    → Prioritize precision, reduce false alarms")
    print("  - Current balanced approach: 0.5")
    print("    → Good balance between catching failures and avoiding false alarms")
    
    return failure_proba, failure_pred

def create_action_recommendations(df, rul_predictions, failure_proba):
    """
    Create actionable recommendations based on model outputs.
    """
    print("\n" + "="*60)
    print("ACTION RECOMMENDATIONS FRAMEWORK")
    print("="*60)
    
    # Create risk categories
    risk_categories = []
    for i in range(len(df)):
        rul = rul_predictions[i]
        prob = failure_proba[i]
        
        if prob >= 0.7 or rul <= 30:
            category = "RED - Urgent Action Required"
        elif prob >= 0.4 or rul <= 60:
            category = "YELLOW - Schedule Maintenance"
        else:
            category = "GREEN - Monitor Only"
        
        risk_categories.append(category)
    
    # Count by category
    category_counts = pd.Series(risk_categories).value_counts()
    
    print("\nMachine Status Distribution:")
    for category, count in category_counts.items():
        pct = 100 * count / len(df)
        print(f"  {category:35s}: {count:3d} machines ({pct:5.1f}%)")
    
    print("\n" + "-"*60)
    print("RECOMMENDED ACTIONS BY CATEGORY")
    print("-"*60)
    print("\n🔴 RED - Urgent Action Required:")
    print("   - Schedule maintenance within 1 week")
    print("   - Prepare spare parts inventory")
    print("   - Assign priority technician")
    print("   - Monitor daily until maintenance complete")
    
    print("\n🟡 YELLOW - Schedule Maintenance:")
    print("   - Schedule maintenance within 2-4 weeks")
    print("   - Include in next maintenance cycle")
    print("   - Monitor weekly")
    
    print("\n🟢 GREEN - Monitor Only:")
    print("   - Continue routine monitoring")
    print("   - No immediate action required")
    print("   - Review in next monthly assessment")
    
    return risk_categories

def plot_model_performance(df, rul_predictions, failure_proba, actual_rul, actual_failure):
    """
    Create visualization of model performance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RUL: Actual vs Predicted
    axes[0, 0].scatter(actual_rul, rul_predictions, alpha=0.5, s=20)
    axes[0, 0].plot([0, max(actual_rul)], [0, max(actual_rul)], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual RUL (Days)', fontsize=11)
    axes[0, 0].set_ylabel('Predicted RUL (Days)', fontsize=11)
    axes[0, 0].set_title('RUL Model: Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RUL: Residuals
    residuals = actual_rul - rul_predictions
    axes[0, 1].scatter(rul_predictions, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted RUL (Days)', fontsize=11)
    axes[0, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    axes[0, 1].set_title('RUL Model: Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Failure Risk: Probability Distribution
    axes[1, 0].hist(failure_proba[actual_failure == 0], bins=20, alpha=0.6, 
                   label='Low Risk (Actual)', color='green', edgecolor='black')
    axes[1, 0].hist(failure_proba[actual_failure == 1], bins=20, alpha=0.6, 
                   label='High Risk (Actual)', color='red', edgecolor='black')
    axes[1, 0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    axes[1, 0].set_xlabel('Predicted Failure Probability', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Failure Risk Model: Probability Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Failure Risk: ROC Curve Data (simplified)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(actual_failure, failure_proba)
    axes[1, 1].plot(fpr, tpr, linewidth=2, label='ROC Curve')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    axes[1, 1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1, 1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1, 1].set_title('Failure Risk Model: ROC Curve', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/model_performance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: outputs/model_performance.png")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL EVALUATION & BUSINESS INTERPRETATION")
    print("Delta Industries Ltd.")
    print("=" * 60)
    
    # Create outputs directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Load data
    print("\nLoading dataset...")
    df = load_data()
    
    # Load models
    print("Loading trained models...")
    rul_model, rul_scaler, failure_model, failure_scaler = load_models()
    
    # Prepare features
    X, feature_cols = prepare_features(df)
    
    # Evaluate RUL model
    rul_predictions = evaluate_rul_model(df, rul_model, rul_scaler, feature_cols)
    
    # Evaluate failure risk model
    failure_proba, failure_pred = evaluate_failure_risk_model(
        df, failure_model, failure_scaler, feature_cols
    )
    
    # Create action recommendations
    risk_categories = create_action_recommendations(
        df, rul_predictions, failure_proba
    )
    
    # Visualizations
    print("\nGenerating performance visualizations...")
    plot_model_performance(
        df, rul_predictions, failure_proba, 
        df['RUL_days'].values, df['fail_in_30d'].values
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nNext step: Run generate_dashboard_outputs.py to create dashboard-ready CSV")

