# Predictive Maintenance Model
## Delta Industries Ltd.

### Project Overview

This project implements an end-to-end predictive maintenance system for manufacturing machines. The system provides two complementary outputs:

1. **Remaining Useful Life (RUL)** - Continuous prediction of days until failure
2. **Failure Risk Classification** - Binary prediction of failure within 30 days

### Problem Statement

Delta Industries operates multiple critical manufacturing machines. Unplanned failures cause production downtime and financial losses. This system enables proactive maintenance scheduling and resource optimization.

### Project Structure

```
predictive_maintenance_model/
├── PROBLEM_DEFINITION.md          # Detailed problem statement
├── data_generation.py             # Synthetic dataset generation
├── eda.py                         # Exploratory data analysis
├── modeling.py                    # Model training (Linear & Logistic Regression)
├── evaluation.py                  # Model evaluation & business interpretation
├── generate_dashboard_outputs.py  # Dashboard-ready CSV generation
├── dashboard.py                   # Interactive Streamlit dashboard
├── DASHBOARD_GUIDE.md             # Dashboard usage guide
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── outputs/                       # Generated outputs
    ├── delta_industries_machine_health.csv  # Dataset
    ├── rul_model.pkl              # Trained RUL model
    ├── failure_risk_model.pkl     # Trained failure risk model
    ├── dashboard_outputs.csv      # Dashboard-ready predictions
    └── *.png                      # Visualization files
```

### Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Run All Steps Sequentially

```bash
# Step 1: Generate synthetic dataset
python data_generation.py

# Step 2: Perform exploratory data analysis
python eda.py

# Step 3: Train models
python modeling.py

# Step 4: Evaluate models and generate business insights
python evaluation.py

# Step 5: Generate dashboard outputs
python generate_dashboard_outputs.py
```

#### Option 2: Run Individual Scripts

Each script can be run independently if you have the required inputs from previous steps.

### Dataset Description

The synthetic dataset contains weekly machine health snapshots with the following features:

**Machine Information:**
- `machine_id`: Unique machine identifier
- `machine_type`: Type of machine (CNC, Lathe, Press, etc.)
- `snapshot_date`: Date of health snapshot
- `age_days`: Machine age in days
- `criticality_score`: Machine criticality (1-5)

**Operational Metrics:**
- `hours_since_last_maintenance`: Hours since last maintenance
- `num_breakdowns_last_6_months`: Historical breakdown count

**Sensor Data (7-day averages):**
- `avg_temp_7d`: Average temperature
- `max_temp_7d`: Maximum temperature
- `avg_vibration_7d`: Average vibration
- `vibration_std_7d`: Vibration standard deviation
- `avg_pressure_7d`: Average pressure
- `pressure_drop_pct_7d`: Pressure drop percentage
- `avg_load_factor_7d`: Average load factor
- `ambient_temp_7d`: Ambient temperature

**Target Variables:**
- `RUL_days`: Remaining Useful Life in days (continuous)
- `fail_in_30d`: Binary indicator (1 = failure within 30 days, 0 = no failure)

### Models

#### 1. Linear Regression (RUL Prediction)
- **Purpose**: Predict continuous RUL in days
- **Evaluation Metrics**: MAE, RMSE, R²
- **Interpretation**: Coefficients show feature impact on machine life

#### 2. Logistic Regression (Failure Risk)
- **Purpose**: Predict probability of failure within 30 days
- **Evaluation Metrics**: ROC-AUC, Precision, Recall
- **Interpretation**: Probability scores enable risk-based decision making

### Business Interpretation

**Risk Categories:**
- **RED**: Urgent action required (failure probability ≥ 70% OR RUL ≤ 30 days)
- **YELLOW**: Schedule maintenance (failure probability ≥ 40% OR RUL ≤ 60 days)
- **GREEN**: Monitor only (low risk)

**Decision Framework:**
- High failure probability → Urgent maintenance
- Moderate RUL → Planned maintenance
- High RUL + low risk → No action needed

### Running the Interactive Dashboard

The project includes a **Streamlit dashboard** for interactive visualization and exploration of model outputs.

#### Quick Start

1. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate dashboard data** (if not already done):
   ```bash
   python3 generate_dashboard_outputs.py
   ```

3. **Launch the dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

4. **Access the dashboard**:
   - The dashboard will automatically open in your browser
   - Default URL: `http://localhost:8501`

#### Dashboard Features

The interactive dashboard provides:

- **Key Metrics**: Total machines, at-risk machines, average RUL, failure probability
- **RUL Overview** (Primary Focus):
  - RUL distribution histogram with threshold markers
  - Predicted vs Actual RUL comparison scatter plot
  - Machines sorted by RUL priority
- **Failure Risk Distribution** (Primary Focus):
  - Risk category pie chart (RED/YELLOW/GREEN)
  - Failure probability distribution histogram
  - Threshold visualization (40% and 70%)
- **Urgent Action Panel**: List of RED category machines requiring immediate attention
- **Machine Details Table**: Comprehensive, sortable, filterable table with all machine information
- **Maintenance Schedule**: Prioritized list with recommended maintenance dates based on RUL

#### Interactive Features

- **Sidebar Filters**: Filter by machine type, risk category, criticality score, and failure flag
- **Sortable Tables**: Sort by RUL, failure probability, machine ID, or risk category
- **Color-Coded Risk Categories**: Visual indicators for RED/YELLOW/GREEN throughout
- **Real-time Updates**: All visualizations update based on selected filters

#### Dashboard Outputs Highlighted

- **RUL Predictions**: Prominently displayed with visualizations and tables
- **Failure Flags**: Clear binary indicators (1/0) for machines requiring urgent action
- **Risk Categories**: Color-coded throughout the dashboard
- **Actionable Insights**: Maintenance schedule based on predictions

For detailed usage instructions, see `DASHBOARD_GUIDE.md`.

### Dashboard Integration (Alternative Tools)

The `dashboard_outputs.csv` file can also be integrated with:
- **Power BI**: Business intelligence reports
- **Tableau**: Advanced visualizations

Key columns in dashboard output:
- `machine_id`
- `snapshot_date`
- `RUL_days_predicted`
- `failure_probability`
- `risk_category`
- `failure_flag` (1 = RED category, 0 = otherwise)

### Running the Interactive Dashboard

The project includes a **Streamlit dashboard** for interactive visualization and exploration of model outputs.

#### Quick Start

1. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate dashboard data** (if not already done):
   ```bash
   python3 generate_dashboard_outputs.py
   ```

3. **Launch the dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

4. **Access the dashboard**:
   - The dashboard will automatically open in your browser
   - Default URL: `http://localhost:8501`

#### Dashboard Features

The interactive dashboard provides:

- **Key Metrics**: Total machines, at-risk machines, average RUL, high failure risk count
- **RUL Overview** (Primary Focus):
  - RUL distribution histogram with risk category color coding
  - Predicted vs Actual RUL comparison scatter plot
  - Visual thresholds at 30 and 60 days
- **Failure Risk Distribution** (Primary Focus):
  - Risk category pie chart (RED/YELLOW/GREEN)
  - Failure probability distribution histogram
  - High-risk threshold visualization (70%)
- **Urgent Action Panel**: List of RED category machines requiring immediate attention
- **Machine Details Table**: Comprehensive, sortable, filterable table with all machine information
- **Maintenance Schedule**: Prioritized list with recommended maintenance dates

#### Interactive Features

- **Sidebar Filters**: Filter by machine type, risk category, criticality score, and RUL range
- **Sortable Tables**: Click column headers to sort
- **Color-Coded Risk Categories**: Visual indicators for RED/YELLOW/GREEN
- **Customizable Views**: Select which columns to display

#### Dashboard Outputs Highlighted

- **RUL Predictions**: Prominently displayed with visualizations and tables
- **Failure Flags**: Clear indicators for machines requiring urgent action
- **Risk Categories**: Color-coded throughout the dashboard
- **Actionable Insights**: Maintenance schedule based on predictions

For detailed usage instructions, see `DASHBOARD_GUIDE.md`.

### Key Features

✅ **Explainable Models**: Linear and Logistic Regression for easy interpretation
✅ **Realistic Data**: Synthetic dataset with logical degradation patterns
✅ **Business-Focused**: Clear risk categories and actionable insights
✅ **Production-Ready**: Clean code structure suitable for deployment
✅ **Viva-Ready**: Well-documented and logically structured

### Model Performance Expectations

- **RUL Model**: MAE < 10 days for machines within 60 days of failure
- **Failure Risk Model**: ROC-AUC > 0.75 with balanced precision/recall

### Cost-Benefit Analysis

The evaluation script includes a cost-benefit analysis comparing:
- Planned maintenance costs
- Prevented downtime savings
- Missed failure costs

### Notes for Viva/Evaluation

1. **Why Two Models?**
   - RUL: Continuous planning metric for scheduling
   - Failure Risk: Binary decision metric for urgent action
   - They are complementary, not contradictory

2. **Why Linear/Logistic Regression?**
   - Highly interpretable (coefficients show feature impact)
   - No black-box complexity
   - Suitable for business stakeholders
   - Fast training and prediction

3. **Data Generation Logic:**
   - Sensor values degrade as failure approaches
   - Correlations preserved between features
   - Realistic failure patterns based on machine criticality

4. **Business Value:**
   - Reduces unplanned downtime
   - Optimizes maintenance resource allocation
   - Enables proactive planning

### Author

Aaryan Rai - AI Engineer and Business Consultant 

### License

This project is for research & development purposes, and all the rights are reserved by the concerned personnel.

