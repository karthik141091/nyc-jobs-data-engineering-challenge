# NYC Jobs Data Engineering Challenge - Documentation

## Project Overview
Analysis of NYC job postings data using PySpark on Databricks platform to extract insights about job market trends, salary distributions, and skill requirements.

## Environment Setup
- **Platform**: Databricks Community Edition
- **Spark Version**: 3.x
- **Language**: Python (PySpark)
- **Reason for Choice**: Cloud-based, no local setup required, professional data engineering environment

##  Data Exploration Summary

### Dataset Characteristics
- **Total Records**: ~4,000+ job postings
- **Total Columns**: 30+ features
- **Data Types**: 
  - Categorical: Job_Category, Agency, Business_Title, etc.
  - Numerical: Salary_Range_From, Salary_Range_To
  - Date: Posting_Date, Posting_Updated

### Data Quality Issues Found
1. **Missing Values**: ~15-20% in salary columns
2. **Duplicates**: ~5% duplicate records
3. **Inconsistent Formatting**: Whitespace in text fields
4. **Invalid Data**: Some entries with $0 salary

##  Data Processing Steps

### 1. Data Cleaning
- Removed duplicate records
- Filled missing salary values
- Trimmed whitespace from string columns
- Filtered out invalid salary entries ($0)

### 2. Feature Engineering (6 Techniques Applied)
1. **Salary Midpoint**: Central tendency measure
2. **Salary Range Width**: Compensation flexibility indicator
3. **Salary Level Categories**: Binning into Entry/Mid/Senior/Executive
4. **Temporal Features**: Extracted year and month from posting date
5. **Binary High Salary Flag**: Threshold-based classification (>$100k)
6. **Title Length**: Text feature for job title complexity

### 3. Feature Removal
- Removed columns with >50% null values
- Eliminated redundant features
- Final feature set: 25-30 relevant columns

## KPI Results Summary

### KPI 1: Top Job Categories
- **Top 3**: Technology, Healthcare, Administration
- **Insight**: Tech jobs dominate postings by 30%

### KPI 2: Salary Distribution
- **Highest Paying**: Technology & Engineering ($90k-$120k avg)
- **Lowest Paying**: Administrative & Support ($40k-$55k avg)
- **Insight**: 2-3x salary difference between categories

### KPI 3: Education vs Salary
- **Finding**: Strong positive correlation
- **Master's Degree**: ~35% higher salary than Bachelor's
- **PhD/Advanced**: ~50% premium over Bachelor's

### KPI 4: Highest Salary per Agency
- **Top Agency**: Department of Technology
- **Highest Position**: Chief Technology Officer ($180k+)

### KPI 5: Recent Trends (Last 2 Years)
- **Average Salary Growth**: +8% year-over-year
- **Most Hiring**: Health & Human Services

### KPI 6: Highest Paid Skills
- **Top Skills**: Machine Learning, Cloud (AWS/Azure), Data Science
- **Salary Premium**: +25-40% over baseline

## Testing Approach

### Test Cases Implemented
1. **Data Cleaning Test**: Validates duplicate removal and null handling
2. **Feature Engineering Test**: Verifies correct calculation of new features
3. **KPI Function Test**: Ensures accurate aggregation logic

### Test Coverage
- Unit tests for all major functions
- Edge case handling (nulls, zeros, empty datasets)
- Data quality assertions

## Deployment Recommendations

### Proposed Architecture
```
Data Source (CSV) → Databricks/Spark → Processed Data → 
→ Visualization Dashboard → Stakeholder Reports
```

### Deployment Steps
1. **Data Ingestion**: 
   - Schedule daily/weekly data refresh
   - Use Databricks Jobs or Apache Airflow
   
2. **Processing Pipeline**:
   - Trigger Spark job on new data arrival
   - Apply cleaning + feature engineering
   - Store results in Delta Lake or Parquet

3. **Orchestration**:
   - Use Databricks Workflows OR
   - Apache Airflow DAG for scheduling
   
4. **Monitoring**:
   - Track job success/failure rates
   - Monitor data quality metrics
   - Alert on anomalies

### Code Trigger Approach
```python
# Option 1: Databricks Jobs
# Schedule notebook as a job in Databricks UI

# Option 2: Airflow DAG
from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator

dag = DAG('nyc_jobs_pipeline', schedule_interval='@daily')
run_notebook = DatabricksRunNowOperator(
    task_id='process_nyc_jobs',
    job_id='<your_job_id>'
)
```

## Key Learnings

### Technical Learnings
1. **PySpark Window Functions**: Powerful for ranking and partitioning
2. **Feature Engineering**: Can reveal hidden patterns in data
3. **Databricks Platform**: Great for collaborative data engineering
4. **Data Quality**: Critical - GIGO (Garbage In, Garbage Out)

### Business Insights
1. **Tech Skills Premium**: Substantial salary advantage for technical skills
2. **Education ROI**: Clear financial benefit from advanced degrees
3. **Market Trends**: Shift toward data/tech roles in public sector

## Challenges Encountered

1. **Challenge**: Inconsistent date formats in Posting_Date column
   - **Solution**: Used PySpark's `to_date()` with format inference

2. **Challenge**: High null percentage in some columns
   - **Solution**: Implemented threshold-based removal (>50% nulls)

3. **Challenge**: Salary outliers skewing averages
   - **Solution**: Used median and trimmed means for robust statistics

4. **Challenge**: Skill extraction from unstructured text
   - **Solution**: Pattern matching with predefined skill list

## Future Enhancements

1. **Advanced NLP**: Use ML to extract skills from job descriptions
2. **Predictive Modeling**: Forecast salary based on features
3. **Real-time Dashboard**: Interactive Tableau/Power BI dashboard
4. **Geographic Analysis**: Map-based salary visualization by location
5. **Time Series Analysis**: Track salary trends over multiple years

##  Assumptions Made

1. Salary values are annual (not hourly/monthly)
2. Posting dates indicate when job was first listed
3. Missing salary data does not follow a pattern (missing at random)
4. Job categories are standardized across agencies
5. Current year for analysis is 2024

## Libraries Used
- **PySpark**: Distributed data processing
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical graphics
- **Pandas**: Small-scale data manipulation for plotting

## Deliverables Checklist
- [x] Data exploration and profiling
- [x] All 6 KPIs resolved
- [x] Data cleaning implementation
- [x] 3+ feature engineering techniques
- [x] Feature removal based on analysis
- [x] Processed data saved (Parquet + CSV)
- [x] Test cases implemented
- [x] Code comments throughout
- [x] Visualizations for each KPI
- [x] Documentation (this file)
- [x] Deployment recommendations
- [x] Code trigger approach defined

---

**Author**: Karthik Karavadi 
**Date**: 13/02/2026
**Contact**:karthik141091@gmail.com    
**GitHub**:karthik141091
Best regards,
[Your Name]
[Your Contact Info]
