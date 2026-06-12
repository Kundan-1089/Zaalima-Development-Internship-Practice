## Docker Deployment

Build Docker Image:
docker build -t customer-churn-api .

Run Docker Container:
docker run -p 8000:8000 customer-churn-api

API Endpoint:
http://localhost:8000

Swagger Documentation:
http://localhost:8000/docs
