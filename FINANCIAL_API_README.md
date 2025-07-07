# Financial Pipeline API

A comprehensive single-file FastAPI application that wraps financial processing pipeline and provides REST endpoints for both web search-based and file upload-based processing, with automatic S3 storage of results.

## Features

- **Single File Architecture**: Everything contained in one Python file (`financial_pipeline_api.py`)
- **Asynchronous Processing**: Background tasks for long-running financial data processing jobs
- **File Upload Support**: Multiple PDF file uploads with validation
- **Web Search Integration**: Automatic PDF downloading via existing pipeline
- **S3 Integration**: Automatic upload of all three JSON outputs to AWS S3
- **Job Tracking**: In-memory job status tracking with UUIDs
- **Error Handling**: Comprehensive error responses with proper HTTP codes
- **Input Validation**: Pydantic models for all requests
- **CORS Support**: Cross-origin resource sharing enabled
- **API Documentation**: Auto-generated Swagger/OpenAPI docs at `/docs`

## API Endpoints

### Core Processing Endpoints

1. **POST /process/websearch** - Process using web search
   - Downloads PDFs via web search and processes them
   - Parameters: `country_name`, `entity_id`, `plant_name`, `target_years` (optional)

2. **POST /process/upload** - Process uploaded files
   - Processes user-uploaded PDF files
   - Parameters: `country_name`, `entity_id`, `plant_name`, file uploads

3. **GET /status/{job_id}** - Check processing status
   - Returns current status of processing job with progress percentage

4. **GET /results/{country_name}/{entity_id}** - Get processing results
   - Returns pre-signed URLs to S3 stored results

5. **GET /** - Root endpoint with API documentation
   - Basic API information and endpoint listing

### Utility Endpoints

- **GET /health** - Health check endpoint
- **GET /jobs** - List all jobs (for monitoring)
- **DELETE /jobs/{job_id}** - Cancel a job

## S3 Storage Structure

Results are stored in S3 following this structure:
```
aws-demo-22/
├── {country_name}/
│   └── {entity_id}/
│       ├── financial.json           # Multi-year financial data
│       ├── financial_assumptions.json  # Latest year assumptions
│       └── final_output.json        # Combined output
```

## Installation

### Prerequisites

- Python 3.8+
- AWS account with S3 access
- API keys for Gemini and OpenAI (optional, for enhanced processing)

### Dependencies

Install the required dependencies:

```bash
pip install fastapi==0.104.1 uvicorn==0.24.0 boto3==1.34.0 python-multipart==0.0.6 pydantic==2.5.0 aiofiles==23.2.0 python-dotenv==1.0.0
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### Environment Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Update `.env` with your actual values:
```env
# API Keys for external services
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# AWS Configuration for S3 storage
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=aws-demo-22

# Server Configuration (optional)
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

## Running the API

### Development Mode

```bash
python financial_pipeline_api.py
```

### Production Mode

```bash
uvicorn financial_pipeline_api:app --host 0.0.0.0 --port 8000
```

### With Docker (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY financial_pipeline_api.py .
COPY .env .

EXPOSE 8000

CMD ["python", "financial_pipeline_api.py"]
```

Build and run:
```bash
docker build -t financial-pipeline-api .
docker run -p 8000:8000 financial-pipeline-api
```

## Usage Examples

### Web Search Processing

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

Response:
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Web search processing started for NTPC Vindhyachal. Use job_id to check status.",
  "data": {
    "country_name": "India",
    "entity_id": "NTPC_001",
    "plant_name": "NTPC Vindhyachal",
    "target_years": [2023, 2024]
  }
}
```

### File Upload Processing

```bash
curl -X POST "http://localhost:8000/process/upload" \
  -F "country_name=India" \
  -F "entity_id=NTPC_001" \
  -F "plant_name=NTPC Vindhyachal" \
  -F "files=@report_2023.pdf" \
  -F "files=@report_2024.pdf"
```

### Check Status

```bash
curl -X GET "http://localhost:8000/status/{job_id}"
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "method": "websearch",
  "country_name": "India",
  "entity_id": "NTPC_001",
  "plant_name": "NTPC Vindhyachal",
  "created_at": "2024-01-15T10:30:00.000Z",
  "updated_at": "2024-01-15T10:32:00.000Z",
  "progress": 75,
  "message": "Processing complete, uploading to S3..."
}
```

### Get Results

```bash
curl -X GET "http://localhost:8000/results/India/NTPC_001"
```

Response:
```json
{
  "success": true,
  "country_name": "India",
  "entity_id": "NTPC_001",
  "results": {
    "financial.json": "https://aws-demo-22.s3.us-east-1.amazonaws.com/India/NTPC_001/financial.json?AWSAccessKeyId=...",
    "financial_assumptions.json": "https://aws-demo-22.s3.us-east-1.amazonaws.com/India/NTPC_001/financial_assumptions.json?AWSAccessKeyId=...",
    "final_output.json": "https://aws-demo-22.s3.us-east-1.amazonaws.com/India/NTPC_001/final_output.json?AWSAccessKeyId=..."
  },
  "generated_at": "2024-01-15T10:35:00.000Z"
}
```

## Testing

Run the test suite to verify all endpoints:

```bash
python test_api.py
```

The test script will:
- Test all API endpoints
- Create sample processing jobs
- Wait for completion
- Verify S3 upload functionality
- Test error handling

## API Documentation

Once the server is running, access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input data, file validation errors
- **404 Not Found**: Job not found, results not found
- **500 Internal Server Error**: S3 connectivity issues, processing failures

All errors return a structured response:
```json
{
  "success": false,
  "error": "Error description",
  "details": "Additional error details",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Monitoring and Logging

- Logs are written to both console and `financial_pipeline.log`
- Health check endpoint at `/health` for monitoring
- Job listing endpoint at `/jobs` for debugging
- Automatic cleanup of old completed jobs

## Security Considerations

For production deployment:

1. **Environment Variables**: Never commit API keys to version control
2. **CORS**: Configure `allow_origins` appropriately for your domain
3. **Rate Limiting**: Consider adding rate limiting middleware
4. **Authentication**: Add authentication for sensitive endpoints
5. **HTTPS**: Use HTTPS in production
6. **File Validation**: The API validates file types and sizes
7. **S3 Permissions**: Use least-privilege IAM policies for S3 access

## Pipeline Integration

The API includes a placeholder `CompleteIntegratedPipeline` class that should be replaced with your actual financial processing pipeline. The integration points are:

- `process_web_search()` - For web search based processing
- `process_uploaded_files()` - For file upload based processing

## Limitations and Notes

- Current implementation includes mock processing for demonstration
- Maximum file upload size is 50MB per file
- Jobs are stored in memory (consider using Redis for production)
- S3 pre-signed URLs expire after 1 hour by default
- The API is designed to be stateless except for job tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Support

For issues and questions:
- Check the logs in `financial_pipeline.log`
- Use the `/health` endpoint to diagnose connectivity issues
- Verify environment variables are set correctly
- Test S3 connectivity separately if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.