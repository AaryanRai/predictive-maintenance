"""
Main Execution Script
Delta Industries Predictive Maintenance Project

This script runs all project steps in sequence:
1. Data generation
2. EDA
3. Modeling
4. Evaluation
5. Dashboard outputs

Run this script to execute the entire pipeline.
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    print(f"Running: {script_name}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {script_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        return False

def main():
    """Execute all project steps."""
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE PROJECT - FULL PIPELINE")
    print("Delta Industries Ltd.")
    print("=" * 60)
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Define execution steps
    steps = [
        ("data_generation.py", "Data Generation"),
        ("eda.py", "Exploratory Data Analysis"),
        ("modeling.py", "Model Training"),
        ("evaluation.py", "Model Evaluation"),
        ("generate_dashboard_outputs.py", "Dashboard Output Generation")
    ]
    
    # Execute each step
    for script, description in steps:
        success = run_script(script, description)
        if not success:
            print(f"\n✗ Pipeline stopped due to error in {script}")
            print("Please fix the error and rerun the pipeline.")
            sys.exit(1)
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*60)
    print("\nAll outputs saved to 'outputs/' directory:")
    print("  - delta_industries_machine_health.csv (dataset)")
    print("  - rul_model.pkl (RUL prediction model)")
    print("  - failure_risk_model.pkl (failure risk model)")
    print("  - dashboard_outputs.csv (dashboard-ready predictions)")
    print("  - *.png (visualization files)")
    print("\nProject is ready for viva/evaluation!")

if __name__ == "__main__":
    main()

