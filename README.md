# Zaalima-Development-Internship-Practice

# Customer Churn Prediction & LTV Engine
# Day 1 Progress Report — Data Cleaning & Initial Data Understanding
# 📌 Project Overview

This project focuses on analyzing telecom customer behavior to predict customer churn and understand factors affecting customer retention and lifetime value (LTV).

The goal is to build an end-to-end analytics and machine learning pipeline using:

Python
SQL
PostgreSQL
Machine Learning
FastAPI
Dashboarding Tools



# ✅ Work Completed So Far
# 1️⃣ Project Environment Setup

Created the initial project workflow and folder structure for:

Raw data
Cleaned data
Notebooks
SQL scripts
Dashboard files
API integration


# 2️⃣ Dataset Loading
Loaded the Telco Customer Churn Dataset into Jupyter Notebook using pandas.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 3️⃣ Initial Dataset Inspection
Performed:

df.head()
df.shape
df.info()
df.describe()
df.describe(include="object")
---Initial Findings
Dataset contained customer demographic and subscription information
Total rows before cleaning: 7043
Dataset included:
categorical features
numerical features
target variable (Churn)


# 4️⃣ Data Type Analysis
Checked datatype consistency using:
df.dtypes

Key Observation
TotalCharges was incorrectly stored as a string (str) instead of numeric.


# 5️⃣ Missing Value Detection
Detected hidden blank-space values inside TotalCharges.

(df["TotalCharges"] == " ").sum()
Result
Found 11 blank records


# 6️⃣ Data Cleaning Performed
Converted Blank Spaces to NaN
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)

Removed Missing Rows
df = df.dropna()

Corrected Datatype
df["TotalCharges"] = df["TotalCharges"].astype(float)


# 7️⃣ Post-Cleaning Validation
Performed:

df.isnull().sum()
df.shape
df.dtypes

Final Result
Missing values removed successfully
TotalCharges converted to float64

Final dataset shape:
(7032, 21)


# 8️⃣ Categorical Data Audit
Analyzed all categorical columns using:

for col in df.select_dtypes(include=["object", "string"]).columns:
    print(df[col].unique())
    
Key Business Observations
Contract types:
Month-to-month
One year
Two year
Internet services:
DSL
Fiber optic
No internet service
Payment methods:
Electronic check
Mailed check
Bank transfer
Credit card
Churn categories:
Yes
No


# 9️⃣ Unique Value Analysis
Performed:

df.nunique()
Key Findings
Feature	Unique Values
customerID	7032
Contract	3
PaymentMethod	4
Churn	2
MonthlyCharges	1584
TotalCharges	6530


# 🔟 Target Variable Analysis (Churn)
Performed churn distribution analysis.

df["Churn"].value_counts()

Result
Churn Status	Count
No	5163
Yes	1869

Churn Percentage Analysis
df["Churn"].value_counts(normalize=True) * 100

Result
Churn Status	Percentage
No	73.42%
Yes	26.58%

# 📌 Initial Business Insights
1. Majority customers are retained
2. Approximately 26.5% customers churned
3. Dataset is moderately imbalanced
4. Month-to-month contracts may have higher churn risk (to be verified in upcoming analysis)
5. TotalCharges required cleaning due to hidden blank values


# 👨‍💻 Role Contribution (Data Analyst)
Current responsibilities completed:

1. Data loading
2. Data cleaning
3. Datatype correction
4. Missing value handling
5. Data auditing
6. Initial exploratory analysis
7. Business observation extraction
