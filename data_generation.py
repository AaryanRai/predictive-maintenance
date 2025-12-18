"""
Synthetic Data Generation for Predictive Maintenance
Delta Industries Ltd.

This script generates realistic machine health data with:
- Weekly snapshots for each machine
- Degrading sensor values as failure approaches
- Realistic correlations between features and failure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_machine_failure_dates(n_machines=40):
    """
    Generate realistic failure dates for machines.
    Machines fail at different times based on their criticality and type.
    """
    machine_types = ['CNC', 'Lathe', 'Press', 'Milling', 'Grinding']
    
    machines = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n_machines):
        machine_id = f"M{str(i+1).zfill(3)}"
        machine_type = random.choice(machine_types)
        
        # Higher criticality machines fail sooner (more usage)
        criticality = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.25, 0.3, 0.2])[0]
        
        # Failure date: between 90 and 450 days from base date
        # Higher criticality = earlier failure
        days_to_failure = 450 - (criticality * 30) + np.random.normal(0, 20)
        days_to_failure = max(90, min(450, days_to_failure))
        failure_date = base_date + timedelta(days=int(days_to_failure))
        
        machines.append({
            'machine_id': machine_id,
            'machine_type': machine_type,
            'criticality_score': criticality,
            'failure_date': failure_date
        })
    
    return pd.DataFrame(machines)

def generate_weekly_snapshots(machine_info, weeks_before_failure=20):
    """
    Generate weekly health snapshots for each machine.
    Sensor values degrade as failure approaches.
    """
    snapshots = []
    base_date = datetime(2023, 1, 1)
    
    for _, machine in machine_info.iterrows():
        machine_id = machine['machine_id']
        machine_type = machine['machine_type']
        criticality = machine['criticality_score']
        failure_date = machine['failure_date']
        
        # Generate snapshots starting 20 weeks before failure
        snapshot_date = failure_date - timedelta(weeks=weeks_before_failure)
        
        # Machine age at first snapshot (random between 100-500 days)
        initial_age = random.randint(100, 500)
        
        week_num = 0
        while snapshot_date < failure_date:
            # Calculate days until failure (RUL)
            rul_days = (failure_date - snapshot_date).days
            
            # Normalized degradation factor (0 = healthy, 1 = failure)
            # More degradation as we approach failure
            degradation = 1 - (rul_days / (weeks_before_failure * 7))
            degradation = max(0, min(1, degradation))
            
            # Age increases with each snapshot
            age_days = initial_age + (week_num * 7)
            
            # Hours since last maintenance (increases, then resets after maintenance)
            # Maintenance happens roughly every 200-300 hours
            maintenance_interval = 200 + (criticality * 20)
            hours_since_maintenance = (week_num * 40) % maintenance_interval
            
            # Number of breakdowns in last 6 months (increases as machine degrades)
            num_breakdowns = max(0, int(degradation * 3 + np.random.normal(0, 0.3)))
            
            # Temperature increases as machine degrades
            base_temp = 45 + (criticality * 5)
            avg_temp_7d = base_temp + (degradation * 15) + np.random.normal(0, 2)
            max_temp_7d = avg_temp_7d + 5 + (degradation * 5) + np.random.normal(0, 1.5)
            
            # Vibration increases as bearings/components wear
            base_vibration = 2.5 + (criticality * 0.5)
            avg_vibration_7d = base_vibration + (degradation * 3) + np.random.normal(0, 0.3)
            vibration_std_7d = 0.3 + (degradation * 0.8) + abs(np.random.normal(0, 0.1))
            
            # Pressure drops as seals degrade
            base_pressure = 100 - (criticality * 5)
            avg_pressure_7d = base_pressure - (degradation * 20) + np.random.normal(0, 3)
            pressure_drop_pct_7d = degradation * 15 + abs(np.random.normal(0, 2))
            
            # Load factor decreases as efficiency degrades
            base_load = 85 - (criticality * 3)
            avg_load_factor_7d = base_load - (degradation * 10) + np.random.normal(0, 2)
            
            # Ambient temperature (seasonal variation)
            days_from_start = (snapshot_date - base_date).days
            ambient_temp_7d = 22 + 8 * np.sin(2 * np.pi * days_from_start / 365) + np.random.normal(0, 2)
            
            # Target variables
            fail_in_30d = 1 if rul_days <= 30 else 0
            
            snapshot = {
                'machine_id': machine_id,
                'machine_type': machine_type,
                'snapshot_date': snapshot_date.strftime('%Y-%m-%d'),
                'age_days': int(age_days),
                'hours_since_last_maintenance': int(hours_since_maintenance),
                'num_breakdowns_last_6_months': num_breakdowns,
                'avg_temp_7d': round(avg_temp_7d, 2),
                'max_temp_7d': round(max_temp_7d, 2),
                'avg_vibration_7d': round(avg_vibration_7d, 2),
                'vibration_std_7d': round(vibration_std_7d, 2),
                'avg_pressure_7d': round(avg_pressure_7d, 2),
                'pressure_drop_pct_7d': round(pressure_drop_pct_7d, 2),
                'avg_load_factor_7d': round(avg_load_factor_7d, 2),
                'ambient_temp_7d': round(ambient_temp_7d, 2),
                'criticality_score': criticality,
                'RUL_days': rul_days,
                'fail_in_30d': fail_in_30d
            }
            
            snapshots.append(snapshot)
            
            # Move to next week
            snapshot_date += timedelta(weeks=1)
            week_num += 1
    
    return pd.DataFrame(snapshots)

def validate_dataset(df):
    """
    Validate the generated dataset for logical consistency.
    """
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    
    # Check 1: RUL decreases as snapshots approach failure
    print("\n1. Checking RUL progression per machine...")
    for machine_id in df['machine_id'].unique()[:5]:
        machine_data = df[df['machine_id'] == machine_id].sort_values('snapshot_date')
        rul_values = machine_data['RUL_days'].values
        is_decreasing = all(rul_values[i] >= rul_values[i+1] for i in range(len(rul_values)-1))
        print(f"   {machine_id}: RUL decreases correctly = {is_decreasing}")
    
    # Check 2: fail_in_30d aligns with RUL_days
    print("\n2. Checking fail_in_30d alignment with RUL_days...")
    mismatches = df[(df['RUL_days'] <= 30) & (df['fail_in_30d'] == 0)] | \
                df[(df['RUL_days'] > 30) & (df['fail_in_30d'] == 1)]
    print(f"   Mismatches found: {len(mismatches)} (should be 0)")
    
    # Check 3: Data summary
    print("\n3. Dataset Summary:")
    print(f"   Total records: {len(df)}")
    print(f"   Unique machines: {df['machine_id'].nunique()}")
    print(f"   Date range: {df['snapshot_date'].min()} to {df['snapshot_date'].max()}")
    print(f"   RUL range: {df['RUL_days'].min()} to {df['RUL_days'].max()} days")
    print(f"   Failure risk (fail_in_30d=1): {df['fail_in_30d'].sum()} ({100*df['fail_in_30d'].mean():.1f}%)")
    
    # Check 4: Feature ranges
    print("\n4. Feature ranges (sanity check):")
    numeric_cols = ['age_days', 'avg_temp_7d', 'avg_vibration_7d', 'avg_pressure_7d']
    for col in numeric_cols:
        print(f"   {col}: {df[col].min():.2f} to {df[col].max():.2f}")
    
    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    print("Generating synthetic machine health data...")
    print("=" * 60)
    
    # Generate machine information
    print("\nStep 1: Generating machine failure dates...")
    machine_info = generate_machine_failure_dates(n_machines=40)
    print(f"Generated {len(machine_info)} machines")
    
    # Generate weekly snapshots
    print("\nStep 2: Generating weekly health snapshots...")
    df = generate_weekly_snapshots(machine_info, weeks_before_failure=20)
    print(f"Generated {len(df)} weekly snapshots")
    
    # Validate dataset
    validate_dataset(df)
    
    # Save to CSV
    import os
    os.makedirs('outputs', exist_ok=True)
    output_file = "outputs/delta_industries_machine_health.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    print(f"Shape: {df.shape}")

