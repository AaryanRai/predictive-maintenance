# Dashboard User Guide
## Delta Industries Predictive Maintenance Dashboard

### Overview

The Streamlit dashboard provides an interactive, presentable interface for evaluating the predictive maintenance project. It prominently displays RUL predictions and failure flags, making it ideal for project evaluation and demonstrations.

### Prerequisites

1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Generate dashboard outputs (if not already done):
```bash
python3 generate_dashboard_outputs.py
```

### Running the Dashboard

Start the dashboard using:
```bash
streamlit run dashboard.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

### Dashboard Features

#### 1. Key Metrics Section
- **Total Machines**: Number of machines currently monitored
- **At Risk (RED)**: Count of machines in RED category requiring urgent action
- **Average RUL**: Mean Remaining Useful Life across all machines
- **Avg Failure Probability**: Average probability of failure within 30 days

#### 2. RUL Overview (Primary Focus)
- **RUL Distribution Histogram**: Visual distribution of predicted RUL values
  - Red dashed line at 30 days (critical threshold)
  - Orange dashed line at 60 days (warning threshold)
- **Predicted vs Actual RUL Scatter Plot**: Comparison of model predictions with actual values
  - Color-coded by risk category
  - Diagonal line shows perfect prediction
  - Hover to see machine details

#### 3. Failure Risk Distribution (Primary Focus)
- **Risk Category Pie Chart**: Breakdown of machines by risk level
  - RED: Urgent action required
  - YELLOW: Schedule maintenance
  - GREEN: Monitor only
- **Failure Probability Distribution**: Histogram showing distribution of failure probabilities
  - Color-coded by risk category
  - Threshold lines at 40% and 70%

#### 4. Urgent Action Required Panel
- **RED Category Machines**: List of all machines requiring immediate attention
- **Key Metrics**: Total RED machines, lowest RUL, highest failure probability
- **Actionable Table**: Machine details sorted by priority
- **Action Recommendations**: Guidance on next steps

#### 5. Machine Details Table
- **Comprehensive View**: All machines with complete information
- **Sortable**: Sort by RUL, failure probability, machine ID, or risk category
- **Color-Coded**: Risk categories highlighted with background colors
- **Columns Include**:
  - Machine ID
  - Machine Type
  - RUL (Days) - Predicted
  - Actual RUL (Days) - For comparison
  - Failure Probability
  - Risk Category
  - Failure Flag (1 = RED, 0 = otherwise)
  - Criticality Score
  - Snapshot Date

#### 6. Maintenance Schedule
- **Prioritized List**: Machines sorted by risk and RUL
- **Recommended Dates**: Calculated based on predicted RUL
- **Days Until Maintenance**: Countdown to recommended maintenance
- **Actionable Planning**: Helps schedule maintenance activities

### Sidebar Filters

Use the sidebar to filter the dashboard:

1. **Machine Type**: Filter by specific machine types (CNC, Lathe, Press, etc.)
2. **Risk Category**: Show only RED, YELLOW, or GREEN machines
3. **Criticality Score**: Filter by machine criticality (1-5)
4. **Failure Flag**: Show only flagged (1) or not flagged (0) machines

All filters can be combined for detailed analysis.

### Key Highlights for Project Evaluation

#### RUL Predictions (Primary Focus)
- Prominently displayed in multiple sections
- Visual comparisons with actual values
- Clear thresholds and color coding
- Sortable and filterable for analysis

#### Failure Flags (Primary Focus)
- Explicit failure_flag column (1/0)
- Risk category color coding
- Dedicated alert panel for urgent cases
- Failure probability prominently displayed

### Interpreting Results

#### Risk Categories

**RED (Urgent Action Required)**
- Failure probability ≥ 70% OR RUL ≤ 30 days
- Action: Schedule maintenance within 1 week
- Prepare spare parts and assign priority technicians

**YELLOW (Schedule Maintenance)**
- Failure probability ≥ 40% OR RUL ≤ 60 days
- Action: Schedule maintenance within 2-4 weeks
- Include in next maintenance cycle

**GREEN (Monitor Only)**
- Low failure probability and RUL > 60 days
- Action: Continue routine monitoring
- No immediate action required

#### RUL Interpretation

- **RUL < 30 days**: Critical - Immediate maintenance required
- **RUL 30-60 days**: High priority - Schedule within 2 weeks
- **RUL 60-90 days**: Medium priority - Plan for next month
- **RUL > 90 days**: Low priority - Routine monitoring

### Dashboard Navigation Tips

1. **Start with Key Metrics**: Get overall fleet health overview
2. **Check Alert Panel**: Identify machines requiring immediate attention
3. **Use Filters**: Narrow down to specific machine types or risk levels
4. **Review Maintenance Schedule**: Plan maintenance activities
5. **Explore Visualizations**: Understand patterns and distributions

### Troubleshooting

**Dashboard won't load:**
- Ensure `outputs/dashboard_outputs.csv` exists
- Run `python3 generate_dashboard_outputs.py` first
- Check that all dependencies are installed

**No data showing:**
- Check sidebar filters - they may be too restrictive
- Reset filters to "All" to see all machines

**Charts not displaying:**
- Ensure plotly is installed: `pip install plotly>=5.17.0`
- Refresh the browser page

### Exporting Data

To export filtered data:
1. Apply desired filters in sidebar
2. Use the Machine Details Table
3. Copy data or use Streamlit's built-in export (if enabled)

### Best Practices for Project Evaluation

1. **Start with Overview**: Show key metrics and overall health
2. **Highlight Critical Cases**: Focus on RED category machines
3. **Demonstrate Interactivity**: Show filtering and sorting capabilities
4. **Explain Visualizations**: Walk through RUL and failure probability charts
5. **Show Actionability**: Demonstrate how dashboard informs maintenance decisions

### Technical Details

- **Framework**: Streamlit
- **Visualization Library**: Plotly (interactive charts)
- **Data Source**: `outputs/dashboard_outputs.csv`
- **Update Frequency**: Static (regenerate CSV to update)

### Future Enhancements

Potential additions for production use:
- Real-time data updates
- Historical trend analysis
- Cost-benefit calculations
- Maintenance cost tracking
- Alert notifications
- Export to PDF reports

---

**For questions or issues, refer to the main README.md or project documentation.**
