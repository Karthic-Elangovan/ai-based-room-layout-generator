#!/usr/bin/env python3
"""
Financial Pipeline API - Comprehensive single-file FastAPI application

This module provides REST endpoints for financial processing pipeline with 
both web search-based and file upload-based processing, with automatic S3 storage.

Dependencies required:
    fastapi==0.104.1
    uvicorn==0.24.0
    boto3==1.34.0
    python-multipart==0.0.6
    pydantic==2.5.0
    aiofiles==23.2.0
    python-dotenv==1.0.0

Environment Variables:
    GEMINI_API_KEY=your_gemini_key
    OPENAI_API_KEY=your_openai_key
    AWS_ACCESS_KEY_ID=your_aws_key
    AWS_SECRET_ACCESS_KEY=your_aws_secret
    AWS_REGION=your_aws_region
    S3_BUCKET_NAME=aws-demo-22
"""

import asyncio
import json
import logging
import os
import tempfile
import traceback
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from contextlib import asynccontextmanager

import aiofiles
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from python_dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Models
# =============================================================================

class JobStatus(str, Enum):
    """Job processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingMethod(str, Enum):
    """Processing method enumeration"""
    WEBSEARCH = "websearch"
    UPLOAD = "upload"

class WebSearchRequest(BaseModel):
    """Request model for web search processing"""
    country_name: str = Field(..., min_length=1, max_length=100, description="Name of the country")
    entity_id: str = Field(..., min_length=1, max_length=100, description="Unique entity identifier")
    plant_name: str = Field(..., min_length=1, max_length=200, description="Name of the plant/facility")
    target_years: Optional[List[int]] = Field(
        None, 
        description="Target years for processing (optional)",
        example=[2023, 2024]
    )

    @validator('target_years')
    def validate_years(cls, v):
        if v is not None:
            current_year = datetime.now().year
            for year in v:
                if not isinstance(year, int) or year < 1900 or year > current_year + 10:
                    raise ValueError(f"Invalid year: {year}. Must be between 1900 and {current_year + 10}")
        return v

class ProcessingStatus(BaseModel):
    """Processing status response model"""
    job_id: str
    status: JobStatus
    method: ProcessingMethod
    country_name: str
    entity_id: str
    plant_name: str
    created_at: datetime
    updated_at: datetime
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    message: Optional[str] = None
    error: Optional[str] = None

class ApiResponse(BaseModel):
    """Generic API response model"""
    success: bool
    job_id: Optional[str] = None
    message: str
    data: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ResultsResponse(BaseModel):
    """Results response model"""
    success: bool = True
    country_name: str
    entity_id: str
    results: Dict[str, str] = Field(description="URLs to S3 stored results")
    generated_at: datetime

# =============================================================================
# S3 Service Integration
# =============================================================================

class S3Service:
    """S3 service for uploading and managing financial data files"""
    
    def __init__(self):
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'aws-demo-22')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=self.region
            )
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 service initialized successfully for bucket: {self.bucket_name}")
        except (NoCredentialsError, ClientError) as e:
            logger.error(f"Failed to initialize S3 service: {e}")
            raise

    async def upload_file(self, file_path: str, s3_key: str, content_type: str = 'application/json') -> str:
        """Upload a file to S3 and return the URL"""
        try:
            with open(file_path, 'rb') as file:
                self.s3_client.upload_fileobj(
                    file,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': content_type}
                )
            
            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"File uploaded successfully to S3: {url}")
            return url
            
        except ClientError as e:
            logger.error(f"Failed to upload file to S3: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"S3 upload failed: {str(e)}"
            )

    async def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """Generate a presigned URL for S3 object"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate download URL: {str(e)}"
            )

    def get_s3_key(self, country_name: str, entity_id: str, filename: str) -> str:
        """Generate S3 key following the required structure"""
        return f"{country_name}/{entity_id}/{filename}"

# =============================================================================
# Background Task Manager
# =============================================================================

class TaskManager:
    """Background task manager for job status tracking and processing"""
    
    def __init__(self):
        self.jobs: Dict[str, ProcessingStatus] = {}
        self.lock = asyncio.Lock()

    async def create_job(self, method: ProcessingMethod, country_name: str, 
                        entity_id: str, plant_name: str) -> str:
        """Create a new processing job"""
        job_id = str(uuid.uuid4())
        
        async with self.lock:
            self.jobs[job_id] = ProcessingStatus(
                job_id=job_id,
                status=JobStatus.PENDING,
                method=method,
                country_name=country_name,
                entity_id=entity_id,
                plant_name=plant_name,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                progress=0
            )
        
        logger.info(f"Created new job: {job_id}")
        return job_id

    async def update_job(self, job_id: str, status: Optional[JobStatus] = None,
                        progress: Optional[int] = None, message: Optional[str] = None,
                        error: Optional[str] = None):
        """Update job status and progress"""
        async with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job {job_id} not found"
                )
            
            job = self.jobs[job_id]
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            
            job.updated_at = datetime.now()
            logger.info(f"Updated job {job_id}: status={job.status}, progress={job.progress}%")

    async def get_job(self, job_id: str) -> ProcessingStatus:
        """Get job status"""
        async with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job {job_id} not found"
                )
            return self.jobs[job_id]

    async def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        async with self.lock:
            to_remove = [
                job_id for job_id, job in self.jobs.items()
                if job.updated_at < cutoff_time and job.status in [JobStatus.COMPLETED, JobStatus.FAILED]
            ]
            
            for job_id in to_remove:
                del self.jobs[job_id]
                logger.info(f"Cleaned up old job: {job_id}")

# =============================================================================
# Financial Pipeline Integration (Placeholder)
# =============================================================================

class CompleteIntegratedPipeline:
    """
    Placeholder for the existing financial processing pipeline
    This would import and integrate the actual CompleteIntegratedPipeline class
    """
    
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.gemini_api_key or not self.openai_api_key:
            logger.warning("API keys not found - pipeline functionality will be limited")

    async def process_web_search(self, country_name: str, entity_id: str, 
                               plant_name: str, target_years: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Process financial data using web search
        This would integrate with the actual pipeline for web search functionality
        """
        logger.info(f"Processing web search for {plant_name} in {country_name}")
        
        # Simulate processing steps
        await asyncio.sleep(2)  # Simulate PDF download time
        
        # Mock financial data structure
        financial_data = {
            "multi_year_data": {
                "revenue": [1000000, 1200000, 1150000],
                "expenses": [800000, 950000, 920000],
                "years": target_years or [2021, 2022, 2023]
            },
            "metadata": {
                "country": country_name,
                "entity": entity_id,
                "plant": plant_name,
                "processed_at": datetime.now().isoformat(),
                "method": "websearch"
            }
        }
        
        financial_assumptions = {
            "latest_year": target_years[-1] if target_years else 2023,
            "growth_rate": 0.05,
            "discount_rate": 0.08,
            "assumptions": {
                "revenue_growth": "5% annually based on market trends",
                "cost_inflation": "3% annually based on industry standards"
            }
        }
        
        final_output = {
            "summary": {
                "total_revenue": sum(financial_data["multi_year_data"]["revenue"]),
                "total_expenses": sum(financial_data["multi_year_data"]["expenses"]),
                "profit_margin": 0.15
            },
            "financial_data": financial_data,
            "assumptions": financial_assumptions
        }
        
        return {
            "financial.json": financial_data,
            "financial_assumptions.json": financial_assumptions,
            "final_output.json": final_output
        }

    async def process_uploaded_files(self, files: List[UploadFile], country_name: str,
                                   entity_id: str, plant_name: str) -> Dict[str, Any]:
        """
        Process uploaded PDF files
        This would integrate with the actual pipeline for file processing
        """
        logger.info(f"Processing {len(files)} uploaded files for {plant_name}")
        
        # Simulate file processing
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type: {file.filename}. Only PDF files are supported."
                )
            
            # Simulate PDF processing time
            await asyncio.sleep(1)
        
        # Mock processing similar to web search
        return await self.process_web_search(country_name, entity_id, plant_name)

# =============================================================================
# FastAPI Application Setup
# =============================================================================

# Global instances
task_manager = TaskManager()
s3_service = None
pipeline = CompleteIntegratedPipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    global s3_service
    
    # Startup
    logger.info("Starting Financial Pipeline API...")
    try:
        s3_service = S3Service()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Financial Pipeline API...")
    await task_manager.cleanup_old_jobs(max_age_hours=0)  # Clean all jobs on shutdown
    logger.info("Application shutdown completed")

# Create FastAPI application
app = FastAPI(
    title="Financial Pipeline API",
    description="Comprehensive API for financial processing pipeline with web search and file upload capabilities",
    version="1.0.0",
    contact={
        "name": "Financial Pipeline Team",
        "email": "support@financialpipeline.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            details=f"HTTP {exc.status_code}"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc)
        ).dict()
    )

# =============================================================================
# Background Processing Functions
# =============================================================================

async def process_websearch_background(job_id: str, request: WebSearchRequest):
    """Background task for web search processing"""
    try:
        await task_manager.update_job(job_id, JobStatus.PROCESSING, 10, "Starting web search...")
        
        # Process using pipeline
        results = await pipeline.process_web_search(
            request.country_name,
            request.entity_id,
            request.plant_name,
            request.target_years
        )
        
        await task_manager.update_job(job_id, progress=50, message="Processing complete, uploading to S3...")
        
        # Upload results to S3
        urls = {}
        for filename, data in results.items():
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(data, tmp_file, indent=2, default=str)
                tmp_file_path = tmp_file.name
            
            try:
                s3_key = s3_service.get_s3_key(request.country_name, request.entity_id, filename)
                url = await s3_service.upload_file(tmp_file_path, s3_key)
                urls[filename] = url
                
                # Update progress
                progress = 50 + (len(urls) / len(results)) * 50
                await task_manager.update_job(job_id, progress=int(progress))
                
            finally:
                # Clean up temporary file
                Path(tmp_file_path).unlink(missing_ok=True)
        
        await task_manager.update_job(
            job_id, 
            JobStatus.COMPLETED, 
            100, 
            f"Processing completed successfully. {len(urls)} files uploaded to S3."
        )
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        await task_manager.update_job(
            job_id,
            JobStatus.FAILED,
            0,
            error=f"Processing failed: {str(e)}"
        )

async def process_upload_background(job_id: str, files_data: List[Dict], 
                                  country_name: str, entity_id: str, plant_name: str):
    """Background task for file upload processing"""
    try:
        await task_manager.update_job(job_id, JobStatus.PROCESSING, 10, "Processing uploaded files...")
        
        # Recreate UploadFile objects from data
        upload_files = []
        temp_files = []
        
        for file_data in files_data:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.write(file_data['content'])
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Create a mock UploadFile for processing
            class MockUploadFile:
                def __init__(self, filename, content):
                    self.filename = filename
                    self.content = content
                    
            upload_files.append(MockUploadFile(file_data['filename'], file_data['content']))
        
        try:
            # Process files using pipeline
            results = await pipeline.process_uploaded_files(
                upload_files,
                country_name,
                entity_id,
                plant_name
            )
            
            await task_manager.update_job(job_id, progress=50, message="File processing complete, uploading to S3...")
            
            # Upload results to S3
            urls = {}
            for filename, data in results.items():
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    json.dump(data, tmp_file, indent=2, default=str)
                    tmp_file_path = tmp_file.name
                
                try:
                    s3_key = s3_service.get_s3_key(country_name, entity_id, filename)
                    url = await s3_service.upload_file(tmp_file_path, s3_key)
                    urls[filename] = url
                    
                    # Update progress
                    progress = 50 + (len(urls) / len(results)) * 50
                    await task_manager.update_job(job_id, progress=int(progress))
                    
                finally:
                    # Clean up temporary file
                    Path(tmp_file_path).unlink(missing_ok=True)
            
            await task_manager.update_job(
                job_id,
                JobStatus.COMPLETED,
                100,
                f"Processing completed successfully. {len(urls)} files uploaded to S3."
            )
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)
                
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        await task_manager.update_job(
            job_id,
            JobStatus.FAILED,
            0,
            error=f"Processing failed: {str(e)}"
        )

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    Root endpoint with API documentation and health check
    """
    return {
        "message": "Financial Pipeline API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "process_websearch": "POST /process/websearch",
            "process_upload": "POST /process/upload",
            "get_status": "GET /status/{job_id}",
            "get_results": "GET /results/{country_name}/{entity_id}"
        },
        "description": "API for financial processing pipeline with web search and file upload capabilities"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test S3 connection
        s3_service.s3_client.list_objects_v2(Bucket=s3_service.bucket_name, MaxKeys=1)
        s3_status = "healthy"
    except Exception as e:
        s3_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "s3": s3_status,
            "task_manager": "healthy",
            "pipeline": "healthy"
        },
        "active_jobs": len(task_manager.jobs)
    }

@app.post("/process/websearch", response_model=ApiResponse)
async def process_websearch(
    request: WebSearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Process financial data using web search
    
    Downloads PDFs via web search and processes them to generate financial analysis.
    The processing is done asynchronously in the background.
    """
    try:
        # Create processing job
        job_id = await task_manager.create_job(
            ProcessingMethod.WEBSEARCH,
            request.country_name,
            request.entity_id,
            request.plant_name
        )
        
        # Start background processing
        background_tasks.add_task(process_websearch_background, job_id, request)
        
        return ApiResponse(
            success=True,
            job_id=job_id,
            message=f"Web search processing started for {request.plant_name}. Use job_id to check status.",
            data={
                "country_name": request.country_name,
                "entity_id": request.entity_id,
                "plant_name": request.plant_name,
                "target_years": request.target_years
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start websearch processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processing: {str(e)}"
        )

@app.post("/process/upload", response_model=ApiResponse)
async def process_upload(
    background_tasks: BackgroundTasks,
    country_name: str = Form(..., description="Name of the country"),
    entity_id: str = Form(..., description="Unique entity identifier"), 
    plant_name: str = Form(..., description="Name of the plant/facility"),
    files: List[UploadFile] = File(..., description="PDF files to process")
):
    """
    Process uploaded PDF files
    
    Processes user-uploaded PDF files to generate financial analysis.
    The processing is done asynchronously in the background.
    """
    try:
        # Validate input
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided"
            )
        
        # Validate file types and size
        max_file_size = 50 * 1024 * 1024  # 50MB
        files_data = []
        
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type: {file.filename}. Only PDF files are supported."
                )
            
            content = await file.read()
            if len(content) > max_file_size:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File too large: {file.filename}. Maximum size is 50MB."
                )
            
            files_data.append({
                'filename': file.filename,
                'content': content
            })
        
        # Create processing job
        job_id = await task_manager.create_job(
            ProcessingMethod.UPLOAD,
            country_name,
            entity_id,
            plant_name
        )
        
        # Start background processing
        background_tasks.add_task(
            process_upload_background,
            job_id,
            files_data,
            country_name,
            entity_id,
            plant_name
        )
        
        return ApiResponse(
            success=True,
            job_id=job_id,
            message=f"File upload processing started for {plant_name}. {len(files)} files queued for processing.",
            data={
                "country_name": country_name,
                "entity_id": entity_id,
                "plant_name": plant_name,
                "files_count": len(files),
                "files": [f["filename"] for f in files_data]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start upload processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processing: {str(e)}"
        )

@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_status(job_id: str):
    """
    Check processing status
    
    Returns the current status of a processing job including progress and any error messages.
    """
    try:
        return await task_manager.get_job(job_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job status: {str(e)}"
        )

@app.get("/results/{country_name}/{entity_id}", response_model=ResultsResponse)
async def get_results(country_name: str, entity_id: str):
    """
    Get processing results
    
    Returns URLs to S3 stored results for the specified country and entity.
    """
    try:
        # Generate S3 keys for the three required files
        files = ["financial.json", "financial_assumptions.json", "final_output.json"]
        results = {}
        
        for filename in files:
            s3_key = s3_service.get_s3_key(country_name, entity_id, filename)
            
            # Check if file exists and generate presigned URL
            try:
                s3_service.s3_client.head_object(Bucket=s3_service.bucket_name, Key=s3_key)
                url = await s3_service.generate_presigned_url(s3_key)
                results[filename] = url
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    # File doesn't exist
                    continue
                else:
                    raise
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No results found for country '{country_name}' and entity '{entity_id}'"
            )
        
        return ResultsResponse(
            country_name=country_name,
            entity_id=entity_id,
            results=results,
            generated_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve results: {str(e)}"
        )

# =============================================================================
# Additional Utility Endpoints
# =============================================================================

@app.get("/jobs", response_model=Dict[str, Any])
async def list_jobs(status_filter: Optional[JobStatus] = None, limit: int = 100):
    """
    List all jobs (for debugging/monitoring)
    """
    jobs = list(task_manager.jobs.values())
    
    if status_filter:
        jobs = [job for job in jobs if job.status == status_filter]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply limit
    jobs = jobs[:limit]
    
    return {
        "total_jobs": len(jobs),
        "jobs": [job.dict() for job in jobs]
    }

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a job (mark as failed)
    """
    try:
        job = await task_manager.get_job(job_id)
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in {job.status} status"
            )
        
        await task_manager.update_job(job_id, JobStatus.FAILED, error="Job cancelled by user")
        
        return {"success": True, "message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )

# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting Financial Pipeline API on {host}:{port}")
    
    uvicorn.run(
        "financial_pipeline_api:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )