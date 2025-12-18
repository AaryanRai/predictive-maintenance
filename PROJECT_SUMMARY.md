# Project Summary - Predictive Maintenance for Delta Industries

## Quick Start

To run the entire project pipeline:
```bash
python3 run_all.py
```

Or run individual steps:
```bash
python3 data_generation.py    # Generate synthetic dataset
python3 eda.py                # Exploratory data analysis
python3 modeling.py           # Train models
python3 evaluation.py         # Evaluate and interpret
python3 generate_dashboard_outputs.py  # Create dashboard CSV
```

## Project Components

### 1. Problem Definition (`PROBLEM_DEFINITION.md`)
- Clear business context
- Two complementary objectives (RUL + Failure Risk)
- Explanation of why both metrics are needed

### 2. Data Generation (`data_generation.py`)
- Generates 40 machines with weekly snapshots
- Realistic degradation patterns
- All required features included
- Validation checks built-in
- Output: `outputs/delta_industries_machine_health.csv`

### 3. EDA (`eda.py`)
- RUL distribution analysis
- Failure risk distribution
- Correlation heatmap
- Degradation pattern visualizations
- Feature importance analysis
- Outputs: Multiple PNG files in `outputs/`

### 4. Modeling (`modeling.py`)
- **Linear Regression**: Predicts RUL (continuous)
- **Logistic Regression**: Predicts failure risk (binary)
- Proper train/test split
- Feature scaling
- Model interpretation
- Outputs: `outputs/rul_model.pkl`, `outputs/failure_risk_model.pkl`

### 5. Evaluation (`evaluation.py`)
- Performance metrics for both models
- Business interpretation
- Cost-benefit analysis
- Action recommendations (RED/YELLOW/GREEN)
- Performance visualizations
- Outputs: `outputs/model_performance.png`

### 6. Dashboard Outputs (`generate_dashboard_outputs.py`)
- Clean CSV with predictions
- Risk categories assigned
- Ready for Streamlit/Power BI/Tableau
- Output: `outputs/dashboard_outputs.csv`

## Key Features for Viva

### Why Two Models?
- **RUL Model**: Continuous planning metric (e.g., "45 days remaining")
- **Failure Risk Model**: Binary decision metric (e.g., "85% chance of failure")
- **Complementary**: RUL tells "when", Risk tells "if urgent"

### Model Choices
- **Linear Regression**: Simple, interpretable, coefficients show feature impact
- **Logistic Regression**: Provides probabilities, handles class imbalance
- **No Deep Learning**: Kept simple for explainability

### Data Quality
- Realistic degradation patterns
- Sensor values correlate with failure
- No data leakage
- Proper validation checks

### Business Value
- Reduces unplanned downtime
- Optimizes maintenance scheduling
- Enables proactive planning
- Clear risk categories for decision-making

## Expected Outputs

After running the full pipeline, you should have:

1. **Dataset**: `outputs/delta_industries_machine_health.csv` (800 rows, 17 columns)
2. **Models**: 
   - `outputs/rul_model.pkl`
   - `outputs/failure_risk_model.pkl`
3. **Visualizations**: 
   - `outputs/rul_distribution.png`
   - `outputs/failure_risk_distribution.png`
   - `outputs/correlation_heatmap.png`
   - `outputs/degradation_patterns.png`
   - `outputs/model_performance.png`
4. **Dashboard**: `outputs/dashboard_outputs.csv`

## Model Performance Expectations

- **RUL Model**: MAE ~8-12 days, R² > 0.7
- **Failure Risk Model**: ROC-AUC > 0.75, Balanced precision/recall

## Viva Preparation

### Questions You Should Be Ready For:

1. **Why not use time series models?**
   - Each row is a snapshot, not a time series
   - Linear/logistic regression are simpler and more interpretable
   - Sufficient for the problem at hand

2. **How do you know the data is realistic?**
   - Sensor values degrade as failure approaches
   - Correlations match real-world physics
   - Validation checks ensure logical consistency

3. **What if a machine has high RUL but high failure probability?**
   - This is unlikely but possible
   - Risk category logic handles this (RED if either condition met)
   - Business decision: prioritize safety (use failure probability)

4. **How would you improve the models?**
   - Feature engineering (interaction terms)
   - Ensemble methods
   - More data
   - Domain-specific features

5. **What are the limitations?**
   - Synthetic data (real data would be better)
   - Assumes linear relationships
   - Doesn't account for external factors (operator skill, etc.)

## Project Strengths

✅ **Complete Pipeline**: End-to-end from data to dashboard
✅ **Well Documented**: Clear comments and explanations
✅ **Business Focused**: Actionable insights, not just metrics
✅ **Explainable**: Simple models, interpretable results
✅ **Professional Structure**: Organized, modular code
✅ **Viva Ready**: All components explained and validated

