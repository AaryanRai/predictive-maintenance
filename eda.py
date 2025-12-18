"""
Exploratory Data Analysis (EDA)
Delta Industries Predictive Maintenance Project

This script performs comprehensive EDA to understand:
- Distribution of target variables
- Feature correlations
- Degradation patterns
- Relationships between features and failure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(filename="outputs/delta_industries_machine_health.csv"):
    """Load the machine health dataset."""
    df = pd.read_csv(filename)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    return df

def plot_rul_distribution(df):
    """
    Analyze RUL distribution.
    Shows how remaining useful life is distributed across all snapshots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['RUL_days'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(df['RUL_days'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["RUL_days"].mean():.1f} days')
    axes[0].axvline(30, color='orange', linestyle='--', linewidth=2, label='30-day threshold')
    axes[0].set_xlabel('Remaining Useful Life (Days)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of RUL (Remaining Useful Life)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by failure risk
    df['Risk Category'] = df['fail_in_30d'].map({0: 'Low Risk', 1: 'High Risk'})
    sns.boxplot(data=df, x='Risk Category', y='RUL_days', ax=axes[1])
    axes[1].set_title('RUL Distribution by Failure Risk Category', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Remaining Useful Life (Days)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/rul_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/rul_distribution.png")
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("RUL DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Mean RUL: {df['RUL_days'].mean():.2f} days")
    print(f"Median RUL: {df['RUL_days'].median():.2f} days")
    print(f"Std Dev: {df['RUL_days'].std():.2f} days")
    print(f"Min RUL: {df['RUL_days'].min()} days")
    print(f"Max RUL: {df['RUL_days'].max()} days")
    print(f"Records with RUL ≤ 30 days: {len(df[df['RUL_days'] <= 30])} ({100*len(df[df['RUL_days'] <= 30])/len(df):.1f}%)")

def plot_failure_risk_distribution(df):
    """
    Analyze failure risk (binary target) distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    risk_counts = df['fail_in_30d'].value_counts()
    axes[0].bar(['Low Risk (0)', 'High Risk (1)'], risk_counts.values, 
                color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Failure Risk Distribution (fail_in_30d)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    total = len(df)
    for i, v in enumerate(risk_counts.values):
        axes[0].text(i, v + total*0.01, f'{v}\n({100*v/total:.1f}%)', 
                    ha='center', fontsize=11, fontweight='bold')
    
    # Pie chart
    axes[1].pie(risk_counts.values, labels=['Low Risk', 'High Risk'], 
               autopct='%1.1f%%', startangle=90, colors=['green', 'red'], 
               explode=(0.05, 0.05))
    axes[1].set_title('Failure Risk Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/failure_risk_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/failure_risk_distribution.png")
    plt.close()
    
    print("\n" + "="*60)
    print("FAILURE RISK DISTRIBUTION")
    print("="*60)
    print(f"Low Risk (fail_in_30d=0): {risk_counts[0]} ({100*risk_counts[0]/total:.1f}%)")
    print(f"High Risk (fail_in_30d=1): {risk_counts[1]} ({100*risk_counts[1]/total:.1f}%)")
    print(f"Class imbalance ratio: {risk_counts[0]/risk_counts[1]:.2f}:1")

def plot_correlation_heatmap(df):
    """
    Create correlation heatmap to identify feature relationships.
    """
    # Select numeric columns for correlation
    numeric_cols = ['age_days', 'hours_since_last_maintenance', 'num_breakdowns_last_6_months',
                   'avg_temp_7d', 'max_temp_7d', 'avg_vibration_7d', 'vibration_std_7d',
                   'avg_pressure_7d', 'pressure_drop_pct_7d', 'avg_load_factor_7d',
                   'ambient_temp_7d', 'criticality_score', 'RUL_days', 'fail_in_30d']
    
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/correlation_heatmap.png")
    plt.close()
    
    # Print key correlations with RUL
    print("\n" + "="*60)
    print("KEY CORRELATIONS WITH RUL_days")
    print("="*60)
    rul_corr = corr_matrix['RUL_days'].sort_values(ascending=False)
    for feature, corr in rul_corr.items():
        if feature != 'RUL_days':
            print(f"{feature:30s}: {corr:6.3f}")

def plot_degradation_patterns(df):
    """
    Visualize how sensor values degrade as machines approach failure.
    """
    # Select a few example machines
    sample_machines = df['machine_id'].unique()[:3]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for machine_id in sample_machines:
        machine_data = df[df['machine_id'] == machine_id].sort_values('RUL_days')
        
        # Temperature vs RUL
        axes[0, 0].plot(machine_data['RUL_days'], machine_data['avg_temp_7d'], 
                       marker='o', label=machine_id, alpha=0.7, markersize=4)
        
        # Vibration vs RUL
        axes[0, 1].plot(machine_data['RUL_days'], machine_data['avg_vibration_7d'], 
                       marker='s', label=machine_id, alpha=0.7, markersize=4)
        
        # Pressure vs RUL
        axes[1, 0].plot(machine_data['RUL_days'], machine_data['avg_pressure_7d'], 
                       marker='^', label=machine_id, alpha=0.7, markersize=4)
        
        # Load factor vs RUL
        axes[1, 1].plot(machine_data['RUL_days'], machine_data['avg_load_factor_7d'], 
                       marker='d', label=machine_id, alpha=0.7, markersize=4)
    
    axes[0, 0].set_xlabel('RUL (Days)', fontsize=11)
    axes[0, 0].set_ylabel('Average Temperature (°C)', fontsize=11)
    axes[0, 0].set_title('Temperature Degradation Pattern', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].invert_xaxis()  # RUL decreases as failure approaches
    
    axes[0, 1].set_xlabel('RUL (Days)', fontsize=11)
    axes[0, 1].set_ylabel('Average Vibration', fontsize=11)
    axes[0, 1].set_title('Vibration Degradation Pattern', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].invert_xaxis()
    
    axes[1, 0].set_xlabel('RUL (Days)', fontsize=11)
    axes[1, 0].set_ylabel('Average Pressure', fontsize=11)
    axes[1, 0].set_title('Pressure Degradation Pattern', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].invert_xaxis()
    
    axes[1, 1].set_xlabel('RUL (Days)', fontsize=11)
    axes[1, 1].set_ylabel('Average Load Factor (%)', fontsize=11)
    axes[1, 1].set_title('Load Factor Degradation Pattern', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('outputs/degradation_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/degradation_patterns.png")
    plt.close()

def feature_importance_analysis(df):
    """
    Analyze which features are most important for predicting failure.
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Compare feature means between high-risk and low-risk machines
    high_risk = df[df['fail_in_30d'] == 1]
    low_risk = df[df['fail_in_30d'] == 0]
    
    features_to_compare = ['avg_temp_7d', 'max_temp_7d', 'avg_vibration_7d', 
                          'vibration_std_7d', 'avg_pressure_7d', 'pressure_drop_pct_7d',
                          'avg_load_factor_7d', 'hours_since_last_maintenance',
                          'num_breakdowns_last_6_months', 'age_days']
    
    print("\nFeature Comparison: High Risk vs Low Risk Machines")
    print("-" * 60)
    print(f"{'Feature':<35s} {'Low Risk':>12s} {'High Risk':>12s} {'Difference':>12s}")
    print("-" * 60)
    
    for feature in features_to_compare:
        low_mean = low_risk[feature].mean()
        high_mean = high_risk[feature].mean()
        diff = high_mean - low_mean
        print(f"{feature:<35s} {low_mean:>12.2f} {high_mean:>12.2f} {diff:>12.2f}")
    
    print("\n" + "="*60)
    print("KEY OBSERVATIONS:")
    print("="*60)
    print("1. High-risk machines show:")
    print("   - Higher temperature (degrading components generate more heat)")
    print("   - Higher vibration (bearing wear, misalignment)")
    print("   - Lower pressure (seal degradation, leaks)")
    print("   - Lower load factor (reduced efficiency)")
    print("   - More breakdowns in recent history")
    print("   - Longer time since last maintenance")
    print("\n2. These patterns align with real-world machine degradation physics.")
    print("3. Features are correlated with failure, making them good predictors.")

if __name__ == "__main__":
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("Delta Industries Predictive Maintenance Project")
    print("=" * 60)
    
    # Create outputs directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Load data
    print("\nLoading dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    # Perform EDA
    print("\n" + "="*60)
    print("1. RUL Distribution Analysis")
    print("="*60)
    plot_rul_distribution(df)
    
    print("\n" + "="*60)
    print("2. Failure Risk Distribution Analysis")
    print("="*60)
    plot_failure_risk_distribution(df)
    
    print("\n" + "="*60)
    print("3. Correlation Analysis")
    print("="*60)
    plot_correlation_heatmap(df)
    
    print("\n" + "="*60)
    print("4. Degradation Pattern Visualization")
    print("="*60)
    plot_degradation_patterns(df)
    
    print("\n" + "="*60)
    print("5. Feature Importance Analysis")
    print("="*60)
    feature_importance_analysis(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print("\nAll visualizations saved to 'outputs/' directory")

