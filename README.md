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


# Customer Churn Prediction & Lifetime Value (LTV) Engine
# Day 3 Progress Report — Feature Engineering, ML Preparation & SMOTE Optimization
# 📌 Objective of Day 3
The primary objective of Day 3 was to transform the cleaned dataset into a machine learning-ready dataset by performing:

Feature Engineering
Categorical Encoding
Feature Scaling
Train-Test Splitting
Baseline Machine Learning Modeling
Class Imbalance Handling using SMOTE
Model Performance Optimization

This phase marked the transition from exploratory analytics to predictive machine learning.

# ✅ Work Completed
# 1️⃣ Feature Engineering

New business-driven features were created using insights obtained during EDA.


# 🔹 Churn Flag Creation
Code
df["Churn_Flag"] = df["Churn"].map({"No": 0, "Yes": 1})

Purpose
Converted the target variable into numeric format for machine learning compatibility.


# 2️⃣ Customer Tenure Segmentation
Created Feature
Tenure_Group

Categories
Tenure Range	Customer Segment
0–12 Months	New Customer
13–24 Months	Early Customer
25–48 Months	Mid-term Customer
49+ Months	Long-term Customer

# 📌 Business Insight
Short-tenure customers showed significantly higher churn behavior.

# 📌 Business Importance
Customer onboarding and first-year retention are critical for churn reduction.


# 3️⃣ Monthly Charges Segmentation
Created Feature
Monthly_Charge_Group

Categories
Monthly Charges	Segment
Low	
Medium	
High	
Very High

# 📌 Business Insight
Customers with higher monthly charges demonstrated higher churn probability.

# 📌 Business Recommendation
Improve value perception for premium plans
Introduce loyalty benefits for high-paying users


# 4️⃣ Customer Value Segmentation
Created Feature
Customer_Value_Segment

Categories
Segment
Low Value
Medium Value
High Value
Premium Value

# 📌 Business Importance
This segmentation helps identify:

high-value customers
premium customers
low-revenue segments

# Useful for:
targeted retention campaigns
LTV optimization
customer prioritization


# 5️⃣ High Risk Customer Flag
Created Feature
High_Risk_Flag
The feature was engineered using multiple high-risk churn indicators identified during EDA.

Risk Conditions Used
Month-to-month contract
Fiber optic internet service
Electronic check payment method
Low tenure customers

# 📌 Result
High-risk flagged customers showed approximately:
71% churn rate

# 📌 Business Importance
This feature can help:

retention teams identify risky customers
prioritize intervention campaigns
reduce future customer loss


# 6️⃣ Categorical Encoding
Machine learning models cannot process text-based categorical data directly.

All categorical features were converted into numeric format using:
Technique Used
Label Encoding

# 📌 Encoded Features Included
gender
Partner
Dependents
Contract
PaymentMethod
InternetService
Tenure_Group
Customer_Value_Segment
and other categorical columns

# 📌 Important Learning
Categorical encoding is a crucial preprocessing step for machine learning model compatibility.


# 7️⃣ Feature Scaling
Technique Used
StandardScaler

Numerical Features Scaled
tenure
MonthlyCharges
TotalCharges

# 📌 Purpose of Scaling
Feature scaling standardizes numerical values so that machine learning algorithms can learn efficiently.

# 📌 Result
After scaling:

Metric	Result
Mean	≈ 0
Standard Deviation	≈ 1


# 8️⃣ Feature & Target Separation
Features (X)
Input variables used for prediction.

Target (y)
Churn_Flag

# 📌 Important ML Concept
Target leakage was prevented by removing:
Churn
Churn_Flag
from the feature dataset before model training.


# 9️⃣ Train-Test Split
Technique Used
train_test_split()

Split Ratio
Dataset	Percentage
Training Data	80%
Testing Data	20%

# 📌 Purpose
Used to evaluate model performance on unseen data.


# 🔟 Baseline Logistic Regression Model
Model Used
Logistic Regression

# 📌 Initial Model Performance
Metric	Score
Accuracy	78.7%
Precision	63.2%
Recall	47.3%
F1-Score	54.1%

# 📌 Key Observation
The dataset suffered from class imbalance, causing poor churn detection recall.

# 📌 Business Concern
The model missed many actual churn customers.

This is problematic because:-
churn customers represent revenue loss
undetected churn reduces retention opportunities


# [11] Class Imbalance Handling using SMOTE
Technique Used
SMOTE (Synthetic Minority Oversampling Technique)

# 📌 Why SMOTE Was Used
The original dataset distribution:

Class	Count
No Churn	5163
Yes Churn	1869
showed significant imbalance.

SMOTE generated synthetic minority-class samples to balance the dataset.

# 📌 Balanced Dataset Result
Class	Count
No Churn	5163
Yes Churn	5163


# [12] Logistic Regression After SMOTE
The Logistic Regression model was retrained on the balanced dataset.

# 📌 Optimized Model Performance
Metric	Before SMOTE	After SMOTE

1. Accuracy	78.7%	79.3%
2. Precision	63.2%	76.5%
3. Recall	47.3%	84.6%
4. F1-Score	54.1%	80.3%

# 🚨 Major Improvement
Recall improved dramatically:
47% → 85%
This means the optimized model can now identify the majority of churn-risk customers successfully.

# 📌 Business Impact
The optimized model can help organizations:

1. detect churn customers early
2. improve customer retention
3. reduce revenue loss
4. build proactive retention strategies

# 📌 Important Machine Learning Learning
This phase demonstrated the importance of:

1. baseline modeling before optimization
2. handling class imbalance
3. balancing precision-recall tradeoffs
4. systematic model improvement


# 🛠️ Technologies & Libraries Used
1. Python
2. Pandas
3. NumPy
4. Scikit-learn
5. imbalanced-learn (SMOTE)
6. Matplotlib
7. Seaborn
8. Jupyter Notebook


# 📌 Files Generated During Day 3
File	Purpose

1. Telco_Customer_churn_feature_engineered.csv	Feature engineered dataset
2. Telco_Customer_churn_ml_ready.csv	Encoded & scaled ML-ready dataset
3. Telco_Customer_churn_smote_ready.csv	Balanced SMOTE dataset


# 📌 Day 3 Learning Outcomes
1. Learned feature engineering techniques
2. Understood categorical encoding & scaling
3. Built ML-ready datasets
4. Trained first classification model
5. Identified class imbalance issues
6. Applied SMOTE balancing
7. Improved churn prediction recall significantly
8. Performed systematic model optimization


# Customer Churn Prediction & Lifetime Value (LTV) Engine
# Day 4 Progress Report — Random Forest Modeling, SMOTE Comparison & Advanced ML Experimentation
# 📌 Objective of Day 4
The primary objective of Day 4 was to perform advanced machine learning experimentation and compare the impact of:
1. different ML algorithms
2. class imbalance handling
3. SMOTE balancing
4. model optimization strategies

This phase focused on understanding whether churn prediction improvements were caused by:
1. better algorithms,
2. balanced datasets,
3. or the combination of both.


# ✅ Work Completed
# 1️⃣ Random Forest Modeling (WITH SMOTE)
A Random Forest Classifier was implemented using the balanced SMOTE dataset generated during Day 3.

# 📌 Model Configuration
Parameters Used

RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 📌 Dataset Used
Balanced dataset created using:
SMOTE (Synthetic Minority Oversampling Technique)

# 📌 Random Forest + SMOTE Performance
Metric	Score
Accuracy	83.6%
Precision	82.3%
Recall	85.6%
F1-Score	83.9%

# 🚨 Major Achievement
The optimized model successfully improved:-

1. churn customer detection
2. model balance
3. prediction reliability
4. overall retention prediction capability

# 📌 Business Importance

The model can now:-
1. identify high-risk customers effectively
2. support proactive retention campaigns
3. reduce customer loss
4. improve customer lifetime value (LTV)


# 2️⃣ Confusion Matrix Analysis (RF + SMOTE)
# Output Summary

Prediction Type	Count
Correct No-Churn Predictions	847
Correct Churn Predictions	881
False Positives	190
False Negatives	148

# 📌 Important Insight
The model correctly identified:-
881 churn customers,
which was a massive improvement compared to earlier models.

# 📌 Business Interpretation
False negatives reduced significantly, meaning:

1. fewer churn customers were missed
2. retention teams can now act earlier
3. customer loss risk decreased substantially


# 3️⃣ Random Forest WITHOUT SMOTE
To scientifically evaluate the impact of class balancing, Random Forest was also trained on the original imbalanced dataset.

# 📌 Random Forest WITHOUT SMOTE Performance

Metric	Score
Accuracy	78.5%
Precision	63.0%
Recall	46.5%
F1-Score	53.5%

# 🚨 Critical Discovery
Despite changing the algorithm from Logistic Regression to Random Forest, performance remained poor on the imbalanced dataset.
This proved that:-
# 📌 The primary issue was class imbalance, not algorithm weakness.

# 📌 Major Analytical Conclusion

Both:
Logistic Regression
Random Forest
performed poorly on the original imbalanced dataset.

# However, after applying SMOTE:
1. recall improved dramatically
2. churn detection improved significantly
3. overall model balance improved

# 📌 Important Machine Learning Insight
This phase demonstrated that:-
# “Balancing the dataset had a greater impact than simply changing the algorithm.”


# 4️⃣ Scientific Model Comparison
The project now includes controlled machine learning experimentation.

# 📌 Model Performance Comparison
# Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	78.7%	63.2%	47.3%	54.1%
Logistic + SMOTE	79.3%	76.5%	84.6%	80.3%
Random Forest WITHOUT SMOTE	78.5%	63.0%	46.5%	53.5%
Random Forest + SMOTE	83.6%	82.3%	85.6%	83.9%


# 🚨 Best Performing Model
# 🏆 Random Forest + SMOTE

This model achieved the best:-
recall
precision
F1-score
overall churn prediction performance

# 📌 Key Learning from Comparison
The experiments proved that:-
1. imbalanced datasets suppress churn detection capability
2. accuracy alone is misleading
3. recall is critical in churn prediction
4. balancing techniques significantly improve minority class detection


# 5️⃣ Business & ML Learnings
# 📌 Important Business Insights
1. Churn prediction systems must prioritize recall
2. Missing churn customers can directly impact revenue
3. High recall models improve customer retention opportunities
4. Data balancing can dramatically improve business outcomes

# 📌 Important Machine Learning Learnings
This phase covered:-
1. advanced classification modeling
2. algorithm comparison
3. class imbalance handling
4. controlled experimentation
5. confusion matrix interpretation
6. recall optimization
7. model evaluation strategy

# 📌 Important Analytical Realization
The project demonstrated a professional ML workflow:-
1. Build baseline model
2. Identify weaknesses
3. Diagnose imbalance issues
4. Apply balancing techniques
5. Compare improvements scientifically
6. Select the best-performing model

# 🛠️ Technologies & Libraries Used
1. Python
2. Pandas
3. NumPy
4. Scikit-learn
5. RandomForestClassifier
6. imbalanced-learn (SMOTE)
7. Matplotlib
8. Seaborn
9. Jupyter Notebook

# 📌 Files Generated During Day 4
File	Purpose
rf_smote_results.csv	RF + SMOTE experiment results
rf_without_smote_results.csv	RF without SMOTE experiment results

# 📌 Day 4 Learning Outcomes
1. Learned Random Forest classification
2. Compared ML algorithms scientifically
3. Understood the real impact of SMOTE
4. Performed controlled experimentation
5. Improved churn detection recall significantly
6. Identified the best-performing model
7. Learned importance of recall over accuracy in churn prediction
