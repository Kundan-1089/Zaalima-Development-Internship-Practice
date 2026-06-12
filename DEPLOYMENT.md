# API Deployment

Run API:

uvicorn app:app --reload

Swagger:
http://127.0.0.1:8000/docs

Docker Build:
docker build -t customer-churn-api .

Docker Run:
docker run -p 8000:8000 customer-churn-api
