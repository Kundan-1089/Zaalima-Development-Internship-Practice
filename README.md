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


# Day 2 Progress Report — Exploratory Data Analysis (EDA) & Business Insights
# 📌 Objective of Day 2

The focus of Day 2 was to perform deep exploratory data analysis (EDA) to identify the major business factors contributing to customer churn.
The analysis concentrated on:

Contract behavior
Internet service usage
Payment behavior
Customer tenure
Pricing impact
Senior citizen behavior
Billing preferences
Numerical feature relationships


# 1️⃣ Churn vs Contract Type Analysis
Crosstab Analysis
pd.crosstab(df["Contract"], df["Churn"])

Percentage Analysis
pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100

Key Findings
Contract Type	Churn Rate
Month-to-month	42.7%
One year	11.2%
Two year	2.8%

# 📌 Business Insight
Customers with month-to-month contracts show extremely high churn behavior compared to long-term contract customers.

# 📌 Business Recommendation
Promote annual and long-term contracts
Offer discounts for yearly subscriptions
Create retention campaigns targeting month-to-month users


# 2️⃣ Churn vs Internet Service Analysis
Crosstab Analysis
pd.crosstab(df["InternetService"], df["Churn"])

Percentage Analysis
pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100

Key Findings
Internet Service	Churn Rate
DSL	19.0%
Fiber optic	41.9%
No internet	7.4%

# 📌 Business Insight
Fiber optic customers exhibit the highest churn rate in the internet service category.

# 📌 Possible Business Reasons
High pricing plans
Service quality dissatisfaction
High customer expectations
Strong market competition

# 📌 Business Recommendation
Improve premium customer support
Launch retention offers for fiber users
Optimize pricing strategy for premium plans


# 3️⃣ Churn vs Payment Method Analysis
Crosstab Analysis
pd.crosstab(df["PaymentMethod"], df["Churn"])

Percentage Analysis
pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100

Key Findings
Payment Method	Churn Rate
Electronic check	45.3%
Mailed check	19.2%
Bank transfer (automatic)	16.7%
Credit card (automatic)	15.2%

# 📌 Business Insight
Electronic check users represent the highest churn-risk payment segment.

# 📌 Business Recommendation
Encourage automatic payment methods
Provide discounts for auto-pay enrollment
Improve digital payment reliability


# 4️⃣ Churn vs Tenure Analysis
Statistical Analysis
df.groupby("Churn")["tenure"].describe()

Key Findings
Customer Group	Average Tenure
Retained Customers	37.6 months
Churned Customers	18.0 months

# 📌 Business Insight
Most churn occurs during the early customer lifecycle stage.
Half of churned customers leave within the first 10 months.

# 📌 Business Recommendation
Improve onboarding experience
Build first-year customer engagement programs
Introduce loyalty campaigns early in the customer lifecycle


# 5️⃣ Churn vs Monthly Charges Analysis
Statistical Analysis
df.groupby("Churn")["MonthlyCharges"].describe()

Key Findings
Customer Group	Average Monthly Charges
Retained Customers	61.3
Churned Customers	74.4

# 📌 Business Insight
Customers paying higher monthly charges are more likely to churn.

# 📌 Business Recommendation
Introduce premium customer retention plans
Offer bundled services and loyalty rewards
Improve value perception for high-paying customers


# 6️⃣ Churn vs Total Charges Analysis
Statistical Analysis
df.groupby("Churn")["TotalCharges"].describe()

Key Findings
Customer Group	Average Total Charges
Retained Customers	2555
Churned Customers	1531

# 📌 Business Insight
Although churned customers pay higher monthly charges, their lower tenure results in lower total lifetime revenue.

# 📌 Business Importance
The company is losing high-potential revenue customers early in the lifecycle.

# 📌 Business Recommendation
Develop early churn detection systems
Improve onboarding for premium customers
Increase customer engagement during the first year


# 7️⃣ Churn vs Senior Citizen Analysis
Crosstab Analysis
pd.crosstab(df["SeniorCitizen"], df["Churn"])

Percentage Analysis
pd.crosstab(df["SeniorCitizen"], df["Churn"], normalize="index") * 100

Key Findings
Customer Group	Churn Rate
Non-Senior Customers	23.7%
Senior Citizens	41.7%

# 📌 Business Insight
Senior citizens are significantly more likely to churn.

# 📌 Business Recommendation
Provide simplified plans for senior customers
Improve support accessibility
Offer senior loyalty discounts


# 8️⃣ Churn vs Paperless Billing Analysis
Crosstab Analysis
pd.crosstab(df["PaperlessBilling"], df["Churn"])

Percentage Analysis
pd.crosstab(df["PaperlessBilling"], df["Churn"], normalize="index") * 100

Key Findings
Paperless Billing	Churn Rate
No	16.4%
Yes	33.6%

# 📌 Business Insight
Paperless billing users show significantly higher churn behavior.

# 📌 Possible Interpretation
Paperless billing may be strongly associated with:

month-to-month contracts
online-first customers
high digital engagement segments

# 📌 Business Recommendation
Improve digital customer engagement
Launch retention email campaigns
Introduce rewards for long-term digital customers


# 9️⃣ Correlation Analysis
Correlation Matrix
df.select_dtypes(include=["int64", "float64"]).corr()

# 📌 Key Correlation Findings
Feature Pair	Correlation
tenure ↔ TotalCharges	0.826
MonthlyCharges ↔ TotalCharges	0.651
tenure ↔ MonthlyCharges	0.247

# 📌 Business Insight
Customer retention (tenure) is the strongest driver of long-term revenue generation.

# 📌 Business Importance
Long-term customers generate significantly higher lifetime value (LTV).


# Emerging High-Risk Customer Persona
Based on EDA findings, the highest churn-risk customers are likely to be:

1. Month-to-month customers
2. Fiber optic users
3. Electronic check users
4. Senior citizens
5. High monthly charge customers
6. Low-tenure customers
7. Paperless billing users


# Technologies Used
1. Python
2. Pandas
3. NumPy
4. Matplotlib
5. Seaborn
6. Jupyter Notebook

# 📌 Day 2 Learning Outcomes
1. Developed business-focused EDA skills
2. Connected multiple customer behavior patterns
3. Generated actionable business recommendations
4. Identified major churn-driving customer segments
5. Built foundation for feature engineering and predictive modeling
