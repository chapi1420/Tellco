# TellCo Telecom Data Analysis

## Overview

This project analyzes telecom data to uncover customer behavior, engagement patterns, and overall network performance. The analysis aims to provide actionable insights to inform strategic decisions for improving customer satisfaction and operational efficiency.

The project is structured into multiple tasks, each addressing specific aspects of the dataset. The results are showcased through exploratory analysis, visualizations, and machine learning techniques.

---

## Notebook Structure

### 1. **Task 1: User Overview Analysis**
   - **Objective**: Understand customer behavior on the network.
   - **Key Steps**:
     - Aggregate metrics per user:
       - Number of sessions.
       - Total session duration.
       - Total data download and upload.
       - Total data volume.
     - Conduct exploratory data analysis (EDA):
       - Handle missing values and outliers.
       - Segment users into decile classes based on session duration.
       - Perform univariate and bivariate analyses.
       - Visualize key insights using Matplotlib and Seaborn.
     - **Output**: Insights on user behavior and recommendations for targeted marketing.

### 2. **Task 2: User Engagement Analysis**
   - **Objective**: Assess user engagement levels.
   - **Key Steps**:
     - Compute engagement metrics such as session frequency, duration, and data usage.
     - Normalize metrics and classify users into engagement clusters using k-means.
     - Visualize engagement trends and patterns.
     - **Output**: Clustered user segments based on engagement levels.

### 3. **Task 3: Experience Analytics**
   - **Objective**: Evaluate network performance and device usage.
   - **Key Steps**:
     - Aggregate metrics such as average RTT, TCP retransmissions, and throughput.
     - Identify top and bottom-performing devices and users.
     - Analyze throughput distributions across devices.
     - **Output**: Insights into user experience and network performance bottlenecks.

### 4. **Task 4: Satisfaction Analysis**
   - **Objective**: Derive and predict customer satisfaction scores.
   - **Key Steps**:
     - Compute engagement and experience scores.
     - Use regression models to predict satisfaction scores.
     - Perform k-means clustering to identify satisfaction segments.
     - **Output**: Key drivers of customer satisfaction.

### 5. **Task 5: Dashboard Development**
   - **Objective**: Present insights interactively.
   - **Key Steps**:
     - Develop a Streamlit dashboard for visualizing:
       - User Overview
       - Engagement
       - Experience
       - Satisfaction
     - Deploy the dashboard for public access.
     - **Output**: An interactive dashboard summarizing all findings.

---

## Key Features

1. **Data Exploration**:
   - Extensive use of EDA techniques to uncover hidden patterns.
   - Handling of missing data and outliers.

2. **Machine Learning**:
   - Clustering for user segmentation.
   - Regression modeling for satisfaction prediction.

3. **Visualization**:
   - Insights visualized using Matplotlib, Seaborn, and Plotly.

4. **Interactive Dashboard**:
   - User-friendly Streamlit dashboard to display key metrics and trends.

---

## Requirements

- **Python**: 3.8 or later
- **Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `sqlalchemy`
  - `streamlit`
- **Database**:
  - PostgreSQL for storing and querying telecom data.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/chapi1420/tellco-analysis.git
   cd tellco-analysis
