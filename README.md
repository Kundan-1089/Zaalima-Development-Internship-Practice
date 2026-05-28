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


# Customer Churn Prediction & LTV Engine
# Day 5 Progress Report — Feature Importance, Explainable AI & Model Comparison
# 📌 Objective of Day 5

Day 5 focused on understanding why the model predicts customer churn.
The goal was not only to build a predictive model, but also to explain the major factors influencing churn and convert model results into business recommendations.

# ✅ Work Completed
# 1. Feature Importance Analysis

Used the best-performing model:-
Random Forest + SMOTE
to identify the most important churn-driving features.

# Top 10 Important Features
Rank	Feature	Importance
1	MonthlyCharges	0.139
2	tenure	0.136
3	TotalCharges	0.134
4	Contract	0.113
5	OnlineSecurity	0.058
6	TechSupport	0.058
7	PaymentMethod	0.044
8	Tenure_Group	0.031
9	DeviceProtection	0.026
10	OnlineBackup	0.026

# 📌 Key Business Insights
1. Monthly Charges
Customers paying higher monthly charges are more likely to churn.
# Recommendation:
Offer premium retention discounts, bundle offers, and loyalty benefits.


3. Tenure
Short-tenure customers are highly churn-prone.
# Recommendation:
Improve onboarding experience and run first-year retention campaigns.


3. Total Charges
Lifetime customer spending is strongly linked with retention behavior.
# Recommendation:
Identify high-value customers early and protect them with priority retention offers.


4. Contract Type
Month-to-month contracts remain one of the strongest churn indicators.
# Recommendation:
Promote one-year and two-year plans through discounts and incentives.


5. Online Security & Tech Support
Customers without support/security services show higher churn tendency.
# Recommendation:
Bundle support and security services with premium plans.


6. Payment Method
Payment behavior contributes to churn risk.
# Recommendation:
Encourage automatic payment methods and reduce payment friction.


# 2. Model Comparison
Compared all completed machine learning experiments.

# Model	Accuracy	Precision	Recall	F1 Score
1. Logistic Regression	0.786	0.632	0.473	0.541
2. Logistic + SMOTE	0.793	0.765	0.846	0.803
3. Random Forest	0.785	0.630	0.465	0.535
4. Random Forest + SMOTE	0.836	0.822	0.856	0.839

# 🏆 Best Performing Model
Random Forest + SMOTE
# Why it was selected
Highest accuracy: 83.6%
Highest recall: 85.6%
Highest F1-score: 83.9%
Best balance between churn detection and prediction reliability

# 📌 Major ML Learning
The comparison proved that:-

Class imbalance was the primary bottleneck.
Without SMOTE, both Logistic Regression and Random Forest showed poor churn recall.
After SMOTE, recall improved dramatically.

# 📌 Business Impact
The final model can help the business:-

1. detect churn-risk customers early
2. reduce revenue loss
3. improve retention campaigns
4. prioritize high-risk customer segments
5. support customer lifetime value strategy

# 📁 Files Generated
# File	Purpose
1. feature_importance_results.csv	Stores feature importance ranking
2. model_comparison_results.csv	Stores all model comparison metrics

# 🛠️ Tools & Libraries Used
1. Python
2. Pandas
3. NumPy
4. Scikit-learn
5. Random Forest
6. Matplotlib
7. Seaborn
8. Jupyter Notebook


# Customer Churn Prediction & LTV Engine
# Day 6 Progress Report — XGBoost Modeling & Final Master Model Comparison
# 📌 Objective of Day 6

Day 6 focused on testing an advanced machine learning algorithm, XGBoost, and comparing it with previous models to finalize the best churn prediction model.

# The main goals were:
1. Train XGBoost without SMOTE
2. Train XGBoost with SMOTE
3. Compare XGBoost with Logistic Regression and Random Forest
4. Build final master model comparison
5. Select the best production-ready model


# ✅ Work Completed
# 1. XGBoost Without SMOTE
XGBoost was first trained on the original imbalanced ML-ready dataset.

# Performance
Metric	Score
1. Accuracy	0.783
2. Precision	0.617
3. Recall	0.487
4. F1 Score	0.544

# Confusion Matrix Summary
# Prediction Type	Count
1. Correct No-Churn Predictions	920
2. Correct Churn Predictions	182
3. False Positives	113
4. False Negatives	192

# Key Insight
Even an advanced boosting algorithm struggled on the imbalanced dataset.
This confirmed again that class imbalance was the major bottleneck, not just model selection.


# 2. XGBoost With SMOTE
XGBoost was then trained on the SMOTE-balanced dataset.

# Performance
# Metric	Score
1. Accuracy	0.809
2. Precision	0.780
3. Recall	0.859
4. F1 Score	0.818

# Confusion Matrix Summary
# Prediction Type	Count
1. Correct No-Churn Predictions	788
2. Correct Churn Predictions	884
3. False Positives	249
4. False Negatives	145

# Key Insight
After applying SMOTE, XGBoost recall improved significantly from 48.7% to 85.9%.
This means the model became much better at identifying customers likely to churn.


# 3. Final Master Model Comparison
All six completed experiments were compared together.

# Model	Accuracy	Precision	Recall	F1 Score
1. Logistic Regression	0.786	0.632	0.473	0.541
2. Logistic + SMOTE	0.793	0.765	0.846	0.803
3. Random Forest	0.785	0.630	0.465	0.535
4. Random Forest + SMOTE	0.836	0.822	0.856	0.839
5. XGBoost	0.783	0.617	0.486	0.544
6. XGBoost + SMOTE	0.809	0.780	0.859	0.817

# 🏆 Final Best Model
Random Forest + SMOTE

# Final Best Model Performance
# Metric	Score
1. Accuracy	0.836
2. Precision	0.822
3. Recall	0.856
4. F1 Score	0.839

# 📌 Why Random Forest + SMOTE Was Selected
Although XGBoost + SMOTE achieved slightly higher recall, Random Forest + SMOTE was selected as the best overall model because it had:-

1. highest accuracy
2. highest precision
3. highest F1-score
4. strong recall
5. best overall balance between churn detection and false alarms

# 📌 Major Machine Learning Conclusion
The complete experiment proved that:-
Class imbalance was the primary performance bottleneck.

Without SMOTE, all three algorithms had weak recall:

# Model	Recall Without SMOTE
1. Logistic Regression	0.473
2. Random Forest	0.465
3. XGBoost	0.486

After SMOTE, recall improved strongly across all models:

# Model	Recall With SMOTE
1. Logistic + SMOTE	0.846
2. Random Forest + SMOTE	0.856
3. XGBoost + SMOTE	0.859

# 📌 Business Impact
The final model can help the business:-

1. identify churn-risk customers early
2. reduce missed churn cases
3. prioritize high-risk customers for retention campaigns
4. improve customer lifetime value
5. support data-driven marketing strategies

# 📌 Final Analytical Insights
1. Accuracy alone was not enough for churn prediction.
2. Recall and F1-score were more important because missing churn customers causes revenue loss.
3. SMOTE significantly improved minority class detection.
4. Random Forest + SMOTE provided the most balanced production-ready performance.
5. XGBoost + SMOTE was useful when maximum churn detection recall is the priority.

# 📁 Files Generated During Day 6
# File	Purpose
1. xgb_without_smote_results.csv	XGBoost without SMOTE experiment results
2. xgb_smote_results.csv	XGBoost with SMOTE experiment results
3. final_master_model_comparison.csv	Final comparison of all six ML experiments

# 🛠️ Tools & Libraries Used
1. Python
2. Pandas
3. NumPy
4. Scikit-learn
5. XGBoost
6. Matplotlib
7. Seaborn
8. Jupyter Notebook


# Customer Churn Prediction & LTV Engine
# 🚀 Day 7 — Power BI Dashboarding & Business Intelligence Layer
# 📌 Project Phase Overview

Day 7 focused on transforming the cleaned telecom churn dataset, machine learning outputs, and LTV analysis into a complete interactive Business Intelligence solution using Power BI.

The objective was to convert analytical findings into executive-level dashboards capable of supporting real-world business decision-making for telecom customer retention, churn reduction, revenue optimization, and customer segmentation.

# 🚀 Power BI Dashboards Developed
# ✅ Dashboard 1 — Executive Overview Dashboard
# Objective

Provide a high-level summary of overall business performance, churn status, customer base distribution, and revenue intelligence.

# Key Components
1. Total Customers KPI
2. Churned Customers KPI
3. Churn Rate KPI
4. Average Customer LTV KPI
5. Total Estimated Lifetime Revenue KPI
6. Customer Churn Distribution Donut Chart
7. Contract Type Distribution Analysis
8. Customer LTV Segment Distribution
9. Interactive Business Slicers

# Business Insights
1. Overall churn rate reached 26.58%
2. Month-to-month contracts dominate the customer base
3. Total customer lifetime revenue exceeded 16M+
4. Customer segments were evenly distributed across value tiers


# 🚀 Dashboard 2 — Customer Churn Intelligence Dashboard
# Objective

Identify customer groups most likely to churn and discover major churn-driving business factors.

# Key Components
1. High-Risk Customer KPI
2. Churn by Contract Type
3. Churn by Internet Service
4. Churn by Senior Citizen Status
5. Churn by Payment Method
6. Churn Trend by Tenure
7. Customer Retention Distribution
8. Interactive Filters & Slicers

# Business Insights
1. Month-to-month customers showed highest churn risk
2. Fiber optic customers displayed elevated churn tendency
3. Electronic check users had highest churn behavior
4. Senior citizens demonstrated higher churn sensitivity
5. Long-term customers showed significantly lower churn probability


# 🚀 Dashboard 3 — LTV & Revenue Intelligence Dashboard
# Objective

Analyze customer lifetime value patterns and identify the most profitable customer segments.

# Key Components
1. Total Customer Lifetime Revenue KPI
2. Premium Segment Revenue KPI
3. Premium Customer Count KPI
4. Average Monthly Revenue KPI
5. Average Customer LTV KPI
6. Revenue Growth by Customer Tenure
7. Revenue by Contract Type
8. LTV Segment Distribution
9. Churn Distribution Across LTV Segments

# Business Insights
1. Premium customer segment generated highest business revenue
2. Two-year contract customers produced strongest revenue stability
3. Customer revenue increased significantly with tenure growth
4. High-value customers demonstrated lower churn behavior
5. Premium segment revenue exceeded 10M+


# 🚀 Dashboard 4 — ML Model Performance & Feature Importance Dashboard
# Objective

Compare machine learning model performance and visualize the strongest churn-driving features.

# Models Evaluated
1. Logistic Regression
2. Logistic Regression + SMOTE
3. Random Forest
4. Random Forest + SMOTE
5. XGBoost
6. XGBoost + SMOTE

# Key Components
1. Best Model Accuracy KPI
2. Best Model Precision KPI
3. Best Model Recall KPI
4. Best Model F1 Score KPI
5. Total Models Evaluated KPI
6. Final Model Comparison Visualization
7. Recall Score Comparison
8. Top Churn Drivers Visualization
9. Feature Importance Analysis

# Final ML Results
# Model	Accuracy	F1 Score
1. Random Forest + SMOTE	83.6%	83.9%
2. XGBoost + SMOTE	80.9%	81.7%
3. Logistic + SMOTE	79.3%	80.3%

# Key ML Insights
1. SMOTE balancing significantly improved recall performance
2. Random Forest + SMOTE achieved best overall results
3. MonthlyCharges, tenure, TotalCharges, and Contract were top churn-driving features
4. ML feature importance aligned strongly with business dashboard insights

# 🚀 Advanced Features Implemented
# ✅ DAX Measures
Custom DAX measures were created for:-
1. Churn Rate
2. Total Estimated LTV
3. Premium Segment Revenue
4. Average LTV
5. High-Risk Customers
6. Premium Customers
7. Revenue KPIs

# 🚀 Interactive Dashboard Features
# Implemented Functionality
1. Cross-filtering
2. Dynamic slicers
3. Segment filtering
4. Contract-based filtering
5. Internet service filtering
6. Payment method filtering
7. LTV segment filtering

# 🚀 Business Impact of the Solution
The final Business Intelligence ecosystem enables telecom businesses to:-
1. Predict customer churn risk
2. Identify high-value customers
3. Optimize retention campaigns
4. Reduce customer acquisition cost
5. Improve long-term customer profitability
6. Monitor revenue concentration patterns
7. Support executive-level decision-making

# 🚀 Used Files & Assets Saved
# Cleaned Dataset Files
1. powerbi_customer_churn_dataset.csv
2. Telco_customer_churn_cleaned.csv

# ML Performance Files
1. final_master_model_comparison.csv
2. feature_importance_results.csv

# Power BI Deliverables
1. Executive Overview Dashboard
2. Churn Intelligence Dashboard
3. LTV & Revenue Intelligence Dashboard
4. ML Performance Dashboard


# 🚀 Final Project Storytelling Summary
1. This project evolved from raw telecom customer data into a complete end-to-end business intelligence and predictive analytics ecosystem.

2. Using machine learning, customer segmentation, churn analysis, lifetime value estimation, and interactive Power BI dashboards, the solution identified high-risk customers, premium revenue segments, and the strongest churn-driving business factors.

3. The final system enables telecom organizations to make data-driven retention decisions, improve customer loyalty, optimize lifetime revenue, and proactively reduce churn through advanced analytics and executive-level dashboard reporting.

