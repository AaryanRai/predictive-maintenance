# Predictive Maintenance Project - Problem Definition
## Delta Industries Ltd.

### Business Context

Delta Industries operates multiple critical manufacturing machines across its production facilities. These machines are essential for maintaining production schedules and meeting customer commitments.

### Current Challenges

**Unplanned Machine Failures** cause:
- Production downtime
- Financial losses from missed deliveries
- Emergency maintenance costs
- Customer dissatisfaction

**Current Maintenance Approach:**
- Time-based maintenance (scheduled regardless of actual condition)
- Reactive maintenance (fixing after failure)
- Poor resource allocation (maintaining healthy machines unnecessarily)

### Project Objectives

Build a predictive maintenance system that provides two complementary outputs:

#### 1. Remaining Useful Life (RUL) - Continuous Metric
- **What**: Estimates how many days a machine can operate before failure
- **Use Case**: Long-term planning, scheduling maintenance windows, resource allocation
- **Example**: "Machine M001 has 45 days of useful life remaining"
- **Business Value**: Enables proactive, planned maintenance scheduling

#### 2. Failure Risk Classification (30-day window) - Binary Decision Metric
- **What**: Identifies machines likely to fail within the next 30 days
- **Use Case**: Urgent action prioritization, immediate maintenance dispatch
- **Example**: "Machine M002 has 85% probability of failing in next 30 days"
- **Business Value**: Prevents catastrophic failures, enables emergency response

### Why Both Metrics Are Complementary (Not Contradictory)

**RUL (Continuous Planning Metric):**
- Provides granular timeline for maintenance planning
- Helps optimize maintenance schedules across multiple machines
- Useful for strategic resource allocation
- Example: A machine with 60 days RUL can be scheduled for maintenance next month

**Failure Risk (Binary Decision Metric):**
- Provides clear yes/no decision for urgent action
- Helps prioritize which machines need immediate attention
- Useful for operational decision-making
- Example: A machine with 85% failure probability needs maintenance THIS WEEK

**Together:**
- RUL tells you "when" (planning horizon)
- Failure risk tells you "if urgent" (action threshold)
- A machine with 25 days RUL will have high failure probability → immediate action
- A machine with 90 days RUL will have low failure probability → planned maintenance

### Success Criteria

1. **RUL Prediction**: Mean Absolute Error (MAE) < 10 days for machines within 60 days of failure
2. **Failure Risk**: ROC-AUC > 0.75, with balanced precision and recall
3. **Business Impact**: Enable 30% reduction in unplanned downtime through proactive maintenance

### Data Strategy

- Weekly machine health snapshots
- Sensor data (temperature, vibration, pressure)
- Operational metrics (load, hours since maintenance)
- Historical failure patterns

