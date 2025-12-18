"""
Predictive Maintenance Modeling
Delta Industries Ltd.

This script builds two models:
1. Linear Regression - Predicts RUL (Remaining Useful Life) in days
2. Logistic Regression - Predicts failure risk (fail_in_30d) probability

Both models use simple, explainable algorithms suitable for business stakeholders.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data(filename="outputs/delta_industries_machine_health.csv"):
    """Load and prepare the dataset."""
    df = pd.read_csv(filename)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    return df

def prepare_features(df):
    """
    Prepare feature matrix and target variables.
    Excludes machine_id, snapshot_date, and target variables from features.
    """
    # Categorical features to encode
    df_encoded = pd.get_dummies(df, columns=['machine_type'], prefix='type', drop_first=True)
    
    # Feature columns (exclude targets and identifiers)
    exclude_cols = ['machine_id', 'snapshot_date', 'RUL_days', 'fail_in_30d']
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X = df_encoded[feature_cols]
    y_rul = df['RUL_days']
    y_failure = df['fail_in_30d']
    
    return X, y_rul, y_failure, feature_cols

def train_rul_model(X_train, y_train, X_test, y_test, feature_names):
    """
    Train Linear Regression model to predict RUL.
    
    Linear Regression is appropriate because:
    - RUL is a continuous variable
    - We expect linear relationships between features and RUL
    - Model is highly interpretable (coefficients show feature impact)
    """
    print("\n" + "="*60)
    print("TRAINING LINEAR REGRESSION MODEL (RUL Prediction)")
    print("="*60)
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTraining Metrics:")
    print(f"  MAE:  {train_mae:.2f} days")
    print(f"  RMSE: {train_rmse:.2f} days")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  MAE:  {test_mae:.2f} days")
    print(f"  RMSE: {test_rmse:.2f} days")
    print(f"  R²:   {test_r2:.4f}")
    
    # Feature importance (coefficients)
    print(f"\n" + "-"*60)
    print("FEATURE IMPORTANCE (Coefficients)")
    print("-"*60)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # Interpretation
    print(f"\n" + "-"*60)
    print("MODEL INTERPRETATION")
    print("-"*60)
    print("Positive coefficients: Feature increase → Higher RUL (longer life)")
    print("Negative coefficients: Feature increase → Lower RUL (shorter life)")
    print("\nTop 5 features affecting RUL:")
    for i, row in feature_importance.head(5).iterrows():
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"  {row['Feature']}: {direction} RUL by {abs(row['Coefficient']):.2f} days per std dev")
    
    # Save model and scaler
    joblib.dump(model, 'outputs/rul_model.pkl')
    joblib.dump(scaler, 'outputs/rul_scaler.pkl')
    print(f"\n✓ Model saved: outputs/rul_model.pkl")
    print(f"✓ Scaler saved: outputs/rul_scaler.pkl")
    
    return model, scaler, y_test_pred

def train_failure_risk_model(X_train, y_train, X_test, y_test, feature_names):
    """
    Train Logistic Regression model to predict failure risk (fail_in_30d).
    
    Logistic Regression is appropriate because:
    - Target is binary (fail or not fail)
    - Provides probability scores (0-1) for risk assessment
    - Highly interpretable (odds ratios, coefficients)
    - Uses class_weight='balanced' to handle class imbalance
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION MODEL (Failure Risk Prediction)")
    print("="*60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with balanced class weights (handles class imbalance)
    # Use None for random_state to allow different model training each run
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=None)
    model.fit(X_train_scaled, y_train)
    
    # Predictions (probabilities and binary)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Use 0.5 threshold for binary predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    print(f"\nTraining Metrics:")
    print(f"  ROC-AUC:   {train_auc:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  ROC-AUC:   {test_auc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    
    # Confusion Matrix
    print(f"\n" + "-"*60)
    print("CONFUSION MATRIX (Test Set)")
    print("-"*60)
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                Predicted")
    print(f"              Low Risk  High Risk")
    print(f"Actual Low Risk   {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"Actual High Risk  {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # Classification Report
    print(f"\n" + "-"*60)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-"*60)
    print(classification_report(y_test, y_test_pred, target_names=['Low Risk', 'High Risk']))
    
    # Feature importance (coefficients)
    print(f"\n" + "-"*60)
    print("FEATURE IMPORTANCE (Coefficients)")
    print("-"*60)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    })
    feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # Interpretation
    print(f"\n" + "-"*60)
    print("MODEL INTERPRETATION")
    print("-"*60)
    print("Positive coefficients: Feature increase → Higher failure risk")
    print("Negative coefficients: Feature increase → Lower failure risk")
    print("\nTop 5 features affecting failure risk:")
    for i, row in feature_importance.head(5).iterrows():
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"  {row['Feature']}: {direction} failure risk (odds ratio: {np.exp(row['Coefficient']):.3f})")
    
    # Threshold explanation
    print(f"\n" + "-"*60)
    print("THRESHOLD SELECTION (Business Context)")
    print("-"*60)
    print("Default threshold: 0.5 (50% probability)")
    print("\nBusiness considerations:")
    print("  - Lower threshold (e.g., 0.3): More sensitive, catches more failures")
    print("    → Higher recall, but more false positives (unnecessary maintenance)")
    print("  - Higher threshold (e.g., 0.7): More conservative, fewer false alarms")
    print("    → Higher precision, but may miss some failures")
    print("\nFor critical machines, use lower threshold to prioritize safety.")
    print("For non-critical machines, use higher threshold to reduce costs.")
    
    # Save model and scaler
    joblib.dump(model, 'outputs/failure_risk_model.pkl')
    joblib.dump(scaler, 'outputs/failure_risk_scaler.pkl')
    print(f"\n✓ Model saved: outputs/failure_risk_model.pkl")
    print(f"✓ Scaler saved: outputs/failure_risk_scaler.pkl")
    
    return model, scaler, y_test_proba, y_test_pred

if __name__ == "__main__":
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE MODELING")
    print("Delta Industries Ltd.")
    print("=" * 60)
    
    # Create outputs directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Load data
    print("\nLoading dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features
    print("\nPreparing features...")
    X, y_rul, y_failure, feature_names = prepare_features(df)
    print(f"Features: {len(feature_names)}")
    print(f"Feature names: {', '.join(feature_names[:5])}...")
    
    # Train-test split (80-20)
    print("\nSplitting data (80% train, 20% test)...")
    # Use None for random_state to allow different splits each run
    X_train, X_test, y_rul_train, y_rul_test, y_failure_train, y_failure_test = train_test_split(
        X, y_rul, y_failure, test_size=0.2, random_state=None, stratify=y_failure
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train RUL model
    rul_model, rul_scaler, rul_predictions = train_rul_model(
        X_train, y_rul_train, X_test, y_rul_test, feature_names
    )
    
    # Train failure risk model
    failure_model, failure_scaler, failure_proba, failure_predictions = train_failure_risk_model(
        X_train, y_failure_train, X_test, y_failure_test, feature_names
    )
    
    print("\n" + "="*60)
    print("MODELING COMPLETE!")
    print("="*60)
    print("\nBoth models trained and saved successfully.")
    print("Next step: Run evaluation.py for detailed business interpretation.")

