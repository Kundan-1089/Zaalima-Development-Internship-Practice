from fastapi import FastAPI

# CREATE FASTAPI APP

app = FastAPI(

    title="Customer Churn Prediction API",

    description="A basic FastAPI application for customer churn prediction and customer lifetime value estimation.",

    version="1.0"

)

# HOME ROUTE

@app.get("/")

def home():

    return {

        "message": "Customer Churn Prediction API is Running Successfully"

    }

# CUSTOMER CHURN PREDICTION ENDPOINT

@app.get("/predict")

def predict_churn(

        tenure: int,

        monthly_charges: float

):

    # SIMPLE CHURN LOGIC

    if tenure < 12 and monthly_charges > 70:

        prediction = "High Churn Risk"

    else:

        prediction = "Low Churn Risk"

    # SIMPLE LTV CALCULATION

    estimated_ltv = tenure * monthly_charges

    return {

        "tenure": tenure,

        "monthly_charges": monthly_charges,

        "prediction": prediction,

        "estimated_ltv": estimated_ltv

    }