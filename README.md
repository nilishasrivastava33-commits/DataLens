# DataLens — Automated Data Analysis & Insight Platform

DataLens is a Python web app for exploring CSV datasets. It provides a dataset overview, a column-level data-quality report, basic data cleaning, descriptive statistics, visualizations, rule-based insights, and a Random Forest regression demo.
For the app:
https://datalens-3sxrwwpw38cnrwe5nc4xiw.streamlit.app/
## Features
- Upload a CSV file
- Inspect rows, columns, data types, missing values, duplicates, and unique values
- Remove duplicate rows and fill missing numerical/categorical values
- View descriptive statistics
- Generate histograms, box plots, and a correlation heatmap
- Display automatic rule-based insights
- Train and evaluate a regression model on a selected numerical target
- Download the cleaned CSV

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository, then deploy the repository through Streamlit Community Cloud.

## Notes and limitations
- Cleaning uses simple baseline rules (median for numeric columns, most frequent value for categorical columns). Review these assumptions for each dataset.
- Correlation does not prove causation.
- Machine-learning metrics are demonstrations and depend on the chosen target and available features. High test scores do not guarantee future performance.
- The app processes uploaded CSV data during the current session; it does not persist user uploads to a database.
