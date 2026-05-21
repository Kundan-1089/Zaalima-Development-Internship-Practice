CREATE DATABASE telco_churn;
USE telco_churn;

CREATE TABLE customers (
customerID VARCHAR(20) PRIMARY KEY,
gender VARCHAR(10),
SeniorCitizen INT,
Partner VARCHAR(5),
Dependents VARCHAR(5),
tenure INT,
PhoneService VARCHAR(5),
MultipleLines VARCHAR(20),
InternetService VARCHAR(20),
OnlineSecurity VARCHAR(20),
OnlineBackup VARCHAR(20),
DeviceProtection VARCHAR(20),
TechSupport VARCHAR(20),
StreamingTV VARCHAR(20),
StreamingMovies VARCHAR(20),
Contract VARCHAR(20),
PaperlessBilling VARCHAR(5),
PaymentMethod VARCHAR(30),
MonthlyCharges DECIMAL(10,2),
TotalCharges DECIMAL(10,2),
Churn VARCHAR(5)
);

SELECT * FROM customers LIMIT 20;

