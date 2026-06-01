-- TABLE_Creation --
DROP TABLE IF EXISTS customer_churn;

CREATE TABLE customer_churn (
    customer_id VARCHAR(50),
    gender VARCHAR(20),
    senior_citizen INT,
    partner VARCHAR(10),
    dependents VARCHAR(10),
    tenure INT,
    phone_service VARCHAR(10),
    multiple_lines VARCHAR(50),
    internet_service VARCHAR(50),
    online_security VARCHAR(50),
    online_backup VARCHAR(50),
    device_protection VARCHAR(50),
    tech_support VARCHAR(50),
    streaming_tv VARCHAR(50),
    streaming_movies VARCHAR(50),
    contract VARCHAR(50),
    paperless_billing VARCHAR(10),
    payment_method VARCHAR(100),
    monthly_charges NUMERIC(10,2),
    total_charges NUMERIC(10,2),
    churn VARCHAR(10),
    estimated_ltv NUMERIC(12,2),
    ltv_segment VARCHAR(50)
);


-- CROSS CHECK
SELECT * FROM customer_churn;

SELECT COUNT(*) FROM customer_churn;



-- Churn_Analysis_Queries --

-- 1. Total customers
SELECT COUNT(*) AS total_customers
FROM customer_churn;


-- 2. Churn distribution
SELECT 
    churn,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM customer_churn
GROUP BY churn
ORDER BY customer_count DESC;


-- 3. Churn by contract type
SELECT 
    contract,
    churn,
    COUNT(*) AS customer_count
FROM customer_churn
GROUP BY contract, churn
ORDER BY contract, churn;


-- 4. Churn rate by contract type
SELECT
    contract,
    ROUND(
        SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) AS churn_rate_percentage
FROM customer_churn
GROUP BY contract
ORDER BY churn_rate_percentage DESC;


-- 5. Churn rate by internet service
SELECT
    internet_service,
    ROUND(
        SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) AS churn_rate_percentage
FROM customer_churn
GROUP BY internet_service
ORDER BY churn_rate_percentage DESC;


-- 6. Churn rate by payment method
SELECT
    payment_method,
    ROUND(
        SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) AS churn_rate_percentage
FROM customer_churn
GROUP BY payment_method
ORDER BY churn_rate_percentage DESC;


-- 7. Average tenure by churn
SELECT
    churn,
    ROUND(AVG(tenure), 2) AS avg_tenure
FROM customer_churn
GROUP BY churn;


-- 8. Average monthly charges by churn
SELECT
    churn,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customer_churn
GROUP BY churn;



-- Business_Insights_Queries --

-- 1. Overall revenue intelligence
SELECT
    ROUND(SUM(estimated_ltv), 2) AS total_estimated_ltv,
    ROUND(AVG(estimated_ltv), 2) AS average_ltv,
    ROUND(AVG(monthly_charges), 2) AS average_monthly_charges,
    ROUND(AVG(tenure), 2) AS average_tenure
FROM customer_churn;


-- 2. LTV segment distribution
SELECT
    ltv_segment,
    COUNT(*) AS customer_count
FROM customer_churn
GROUP BY ltv_segment
ORDER BY customer_count DESC;


-- 3. Average LTV by segment
SELECT
    ltv_segment,
    ROUND(AVG(estimated_ltv), 2) AS avg_ltv
FROM customer_churn
GROUP BY ltv_segment
ORDER BY avg_ltv ASC;


-- 4. Churn rate by LTV segment
SELECT
    ltv_segment,
    ROUND(
        SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) AS churn_rate_percentage
FROM customer_churn
GROUP BY ltv_segment
ORDER BY churn_rate_percentage DESC;


-- 5. Premium customer count and revenue
SELECT
    COUNT(*) AS premium_customers,
    ROUND(SUM(estimated_ltv), 2) AS premium_segment_revenue,
    ROUND(AVG(estimated_ltv), 2) AS avg_premium_ltv
FROM customer_churn
WHERE ltv_segment = 'Premium Value';


-- 6. Premium churn customers
SELECT
    COUNT(*) AS premium_churn_customers,
    ROUND(SUM(estimated_ltv), 2) AS revenue_at_risk
FROM customer_churn
WHERE ltv_segment = 'Premium Value'
  AND churn = 'Yes';


-- 7. Top 10 highest LTV customers
SELECT
    customer_id,
    estimated_ltv,
    monthly_charges,
    tenure,
    contract,
    churn
FROM customer_churn
ORDER BY estimated_ltv DESC
LIMIT 10;


-- 8. High-risk customer logic validation
SELECT
    COUNT(*) AS high_risk_customers
FROM customer_churn
WHERE churn = 'Yes'
  AND contract = 'Month-to-month';


-- 9. Revenue by contract type
SELECT
    contract,
    COUNT(*) AS customer_count,
    ROUND(SUM(estimated_ltv), 2) AS total_ltv,
    ROUND(AVG(estimated_ltv), 2) AS avg_ltv
FROM customer_churn
GROUP BY contract
ORDER BY total_ltv DESC;


-- 10. Final business summary
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate,
    ROUND(SUM(estimated_ltv), 2) AS total_estimated_ltv,
    ROUND(AVG(estimated_ltv), 2) AS average_ltv
FROM customer_churn;

