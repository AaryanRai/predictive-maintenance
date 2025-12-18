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
import time

# Set random seed based on current time for unique data each run
# This ensures different data is generated each time the pipeline runs
current_time_seed = int(time.time() * 1000) % (2**31)  # Use milliseconds for uniqueness
np.random.seed(current_time_seed)
random.seed(current_time_seed)
print(f"Using random seed: {current_time_seed} (ensures unique data generation)")

def generate_machine_failure_dates(n_machines=40):
    """
    Generate realistic machine information with varied lifecycle stages.
    Not all machines are approaching failure - we need a realistic mix.
    """
    machine_types = ['CNC', 'Lathe', 'Press', 'Milling', 'Grinding']
    
    machines = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n_machines):
        machine_id = f"M{str(i+1).zfill(3)}"
        machine_type = random.choice(machine_types)
        
        # Higher criticality machines fail sooner (more usage)
        criticality = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.25, 0.3, 0.2])[0]
        
        # Realistic approach: Only some machines will fail during observation period
        # ~60% will be healthy (won't fail during observation), ~25% will be in warning, ~15% will be critical
        # Observation period is 52 weeks = 364 days
        lifecycle_stage = random.choices(
            ['healthy', 'warning', 'critical'],
            weights=[0.6, 0.25, 0.15]
        )[0]
        
        if lifecycle_stage == 'healthy':
            # Healthy machines: failure is far in the future, well beyond observation period (450-700 days)
            # This ensures they remain healthy throughout the 52-week (364-day) observation period
            days_to_failure = random.randint(450, 700)
        elif lifecycle_stage == 'warning':
            # Warning machines: failure is medium-term, may or may not fail during observation (150-400 days)
            # Some will fail, some won't
            days_to_failure = random.randint(150, 400)
        else:  # critical
            # Critical machines: failure is near-term, will likely fail during observation (10-150 days)
            days_to_failure = random.randint(10, 150)
        
        # Adjust based on criticality (higher criticality = earlier failure, but not too extreme)
        # For healthy machines, reduce the adjustment to keep them healthy
        if lifecycle_stage == 'healthy':
            days_to_failure = max(400, days_to_failure - (criticality * 5))
        else:
            days_to_failure = max(10, days_to_failure - (criticality * 8))
        
        failure_date = base_date + timedelta(days=int(days_to_failure))
        
        machines.append({
            'machine_id': machine_id,
            'machine_type': machine_type,
            'criticality_score': criticality,
            'failure_date': failure_date,
            'lifecycle_stage': lifecycle_stage
        })
    
    return pd.DataFrame(machines)

def generate_weekly_snapshots(machine_info, observation_weeks=52):
    """
    Generate weekly health snapshots for each machine over a fixed observation period.
    Creates realistic mix of healthy, warning, and critical machines.
    """
    snapshots = []
    base_date = datetime(2023, 1, 1)
    end_date = base_date + timedelta(weeks=observation_weeks)
    
    for _, machine in machine_info.iterrows():
        machine_id = machine['machine_id']
        machine_type = machine['machine_type']
        criticality = machine['criticality_score']
        failure_date = machine['failure_date']
        lifecycle_stage = machine['lifecycle_stage']
        
        # Start generating snapshots from base_date
        snapshot_date = base_date
        
        # Machine age at first snapshot (random between 100-500 days)
        initial_age = random.randint(100, 500)
        
        week_num = 0
        while snapshot_date < end_date:
            # Calculate days until failure (RUL) - can be negative if failure already passed
            rul_days = (failure_date - snapshot_date).days
            
            # Only generate snapshots if machine hasn't failed yet (RUL > 0)
            # Or if we want to include post-failure data (for now, skip post-failure)
            if rul_days <= 0:
                snapshot_date += timedelta(weeks=1)
                week_num += 1
                continue
            
            # Normalized degradation factor (0 = healthy, 1 = failure)
            # Use a realistic degradation curve based on RUL
            # Degradation increases as RUL decreases, but with realistic bounds
            
            # Base degradation: inverse relationship with RUL
            # For very high RUL (>300 days), degradation is minimal (0-0.2)
            # For medium RUL (100-300 days), degradation is moderate (0.2-0.6)
            # For low RUL (<100 days), degradation is high (0.6-1.0)
            
            if rul_days > 300:
                # Very healthy: minimal degradation (0-0.2)
                base_degradation = 0.0 + (300 / rul_days) * 0.2
                degradation = base_degradation + np.random.normal(0, 0.05)
                degradation = max(0.0, min(0.3, degradation))
            elif rul_days > 100:
                # Moderate health: medium degradation (0.2-0.6)
                # Linear interpolation between 0.2 (at 300 days) and 0.6 (at 100 days)
                base_degradation = 0.2 + (300 - rul_days) / 200 * 0.4
                degradation = base_degradation + np.random.normal(0, 0.1)
                degradation = max(0.2, min(0.7, degradation))
            else:
                # Critical: high degradation (0.6-1.0)
                # Linear interpolation between 0.6 (at 100 days) and 1.0 (at 0 days)
                base_degradation = 0.6 + (100 - rul_days) / 100 * 0.4
                degradation = base_degradation + np.random.normal(0, 0.1)
                degradation = max(0.6, min(1.0, degradation))
            
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
    
    # Generate weekly snapshots (52 weeks = 1 year of observations)
    print("\nStep 2: Generating weekly health snapshots...")
    df = generate_weekly_snapshots(machine_info, observation_weeks=52)
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

