# Quick Start Guide - Financial Pipeline API

## 🚀 Quick Setup (3 steps)

### 1. Install Dependencies
```bash
pip install fastapi uvicorn boto3 python-multipart pydantic aiofiles python-dotenv
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

### 3. Start the API
```bash
python financial_pipeline_api.py
```

The API will be available at: http://localhost:8000

## 📚 Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Test the API

### Option 1: Use the test script
```bash
python test_api.py
```

### Option 2: Manual testing with curl

#### Start a web search job:
```bash
curl -X POST "http://localhost:8000/process/websearch" \
  -H "Content-Type: application/json" \
  -d '{
    "country_name": "India",
    "entity_id": "NTPC_001",
    "plant_name": "NTPC Vindhyachal",
    "target_years": [2023, 2024]
  }'
```

#### Check job status:
```bash
curl "http://localhost:8000/status/{job_id}"
```

#### Get results:
```bash
curl "http://localhost:8000/results/India/NTPC_001"
```

## 🐳 Deploy with Docker

### Simple Docker:
```bash
docker build -t financial-api .
docker run -p 8000:8000 financial-api
```

### Docker Compose:
```bash
docker-compose up -d
```

## 🛠️ Using the Deployment Script

The `deploy.sh` script automates common tasks:

```bash
# Setup environment and dependencies
./deploy.sh setup

# Start development server
./deploy.sh dev

# Start production server
./deploy.sh prod

# Deploy with Docker
./deploy.sh docker
```

## 📊 API Response Examples

### Successful Job Creation:
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Web search processing started for NTPC Vindhyachal",
  "data": {
    "country_name": "India",
    "entity_id": "NTPC_001",
    "plant_name": "NTPC Vindhyachal"
  }
}
```

### Job Status Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "message": "Processing completed successfully. 3 files uploaded to S3."
}
```

### Results Response:
```json
{
  "success": true,
  "country_name": "India",
  "entity_id": "NTPC_001",
  "results": {
    "financial.json": "https://aws-demo-22.s3.us-east-1.amazonaws.com/India/NTPC_001/financial.json?...",
    "financial_assumptions.json": "https://aws-demo-22.s3.us-east-1.amazonaws.com/India/NTPC_001/financial_assumptions.json?...",
    "final_output.json": "https://aws-demo-22.s3.us-east-1.amazonaws.com/India/NTPC_001/final_output.json?..."
  }
}
```

## ⚠️ Important Notes

1. **Environment Variables**: Make sure to set your AWS credentials in `.env`
2. **S3 Bucket**: Ensure the S3 bucket exists and you have write permissions
3. **API Keys**: Gemini and OpenAI keys are optional for the mock implementation
4. **File Limits**: Maximum 50MB per PDF file upload
5. **Production**: Use proper authentication and HTTPS in production

## 🔍 Health Check
```bash
curl http://localhost:8000/health
```

This will show the status of all services including S3 connectivity.

## 📝 Logs
- Console output shows real-time processing
- File logs written to `financial_pipeline.log`
- Docker logs: `docker logs <container_name>`

## 🆘 Troubleshooting

### API won't start:
- Check if port 8000 is available
- Verify all dependencies are installed
- Check `.env` file configuration

### S3 errors:
- Verify AWS credentials in `.env`
- Check bucket name and permissions
- Test AWS CLI access: `aws s3 ls`

### Job processing fails:
- Check logs for detailed error messages
- Verify file formats (PDF only)
- Check file size limits

## 🔗 Integration

To integrate with your existing pipeline, replace the mock `CompleteIntegratedPipeline` class with your actual implementation in `financial_pipeline_api.py`.

The integration points are:
- `process_web_search()` method
- `process_uploaded_files()` method

Happy processing! 🎉