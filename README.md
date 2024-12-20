# TellCo Telecommunications Analysis and Dashboard

## Project Overview

This project is focused on analyzing data from TellCo, a mobile service provider in the Republic of Pefkakia. The goal is to determine the business growth potential and provide actionable insights through comprehensive data analysis, machine learning, and dashboard development. The results will guide recommendations on whether TellCo should be purchased by the client.

## Key Objectives

1. **User Overview Analysis**: Understand customer behavior, top devices, and usage patterns.
2. **User Engagement Analysis**: Evaluate user activity levels and group customers based on engagement metrics.
3. **Experience Analytics**: Analyze network and device performance metrics for better user experience.
4. **Satisfaction Analysis**: Derive satisfaction scores based on engagement and experience.
5. **Dashboard Development**: Build a user-friendly dashboard to visualize insights and KPIs.

## Folder Structure

```plaintext
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── src/
│   ├── __init__.py
│   └── data_processing.py
├── notebooks/
│   ├── analysis.ipynb
│   └── README.md
├── tests/
│   └── test_data_processing.py
├── scripts/
│   ├── run_analysis.py
│   └── generate_dashboard.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .gitignore
Requirements
Python 3.9 or higher
PostgreSQL database
Required libraries (see requirements.txt)
Installation and Setup
Clone this repository:

bash
Copy code
git clone <repository-url>
cd <repository-name>
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up the PostgreSQL database and load the provided SQL schema and data.

Run the main scripts:

Data Preprocessing: src/data_processing.py
Dashboard Development: scripts/generate_dashboard.py
Optional: Build a Docker image and run the container:

bash
Copy code
docker build -t tellco-analysis .
docker run -p 8501:8501 tellco-analysis
Deliverables
Analysis Report: Detailed insights into customer behavior, engagement, and satisfaction.
Dashboard: A Streamlit-based dashboard with interactive visualizations, hosted on [Heroku/Netlify].
Code Repository: Includes reusable code for analysis, modeling, and deployment.
Methodologies
Exploratory Data Analysis (EDA):

Identified top devices and manufacturers.
Segmented users into decile classes based on session durations.
Engagement Analysis:

Normalized metrics and used k-means clustering.
Derived top users and their application-specific engagement.
Experience Analytics:

Analyzed throughput, RTT, and retransmission metrics.
Clustered users into experience groups using k-means.
Satisfaction Analysis:

Calculated engagement and experience scores.
Built a regression model to predict satisfaction.
Visualization:

Developed a Streamlit dashboard with clear navigation and interactive elements.
Key Features
Reusable and Modular Code: Designed for scalability and maintainability.
SQL Integration: Leveraged PostgreSQL for data storage and retrieval.
CI/CD Pipeline: Configured using GitHub Actions.
Dockerized Application: Simplifies deployment and ensures consistency.
Authors and Acknowledgments
Team:

Mahlet
Rediet
Kerod
Elias
Emitinan
Rehmet
This project was completed as part of the 10 Academy Artificial Intelligence Mastery program.

References
Streamlit Documentation
Pandas Documentation
Scikit-learn Documentation
10 Academy Learning Resources
