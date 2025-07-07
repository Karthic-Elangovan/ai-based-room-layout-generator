#!/usr/bin/env python3
"""
Test script for Financial Pipeline API

This script tests the main endpoints of the Financial Pipeline API.
Run the API server first with: python financial_pipeline_api.py
"""

import asyncio
import requests
import json
import time
import os
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

class APITester:
    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def test_root_endpoint(self):
        """Test the root endpoint"""
        print("🔍 Testing root endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            data = response.json()
            print(f"✅ Root endpoint successful: {data['message']}")
            return True
        except Exception as e:
            print(f"❌ Root endpoint failed: {e}")
            return False

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
            print(f"✅ Health check successful: {data['status']}")
            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

    def test_websearch_endpoint(self):
        """Test the websearch processing endpoint"""
        print("🔍 Testing websearch endpoint...")
        try:
            payload = {
                "country_name": "India",
                "entity_id": "NTPC_001",
                "plant_name": "NTPC Vindhyachal",
                "target_years": [2023, 2024]
            }
            
            response = self.session.post(
                f"{self.base_url}/process/websearch",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            if data['success'] and data['job_id']:
                print(f"✅ Websearch processing started: Job ID {data['job_id']}")
                return data['job_id']
            else:
                print(f"❌ Websearch processing failed: {data}")
                return None
                
        except Exception as e:
            print(f"❌ Websearch endpoint failed: {e}")
            return None

    def test_status_endpoint(self, job_id):
        """Test the status endpoint"""
        print(f"🔍 Testing status endpoint for job {job_id}...")
        try:
            response = self.session.get(f"{self.base_url}/status/{job_id}")
            response.raise_for_status()
            data = response.json()
            print(f"✅ Status check successful: {data['status']} ({data['progress']}%)")
            print(f"   Message: {data.get('message', 'N/A')}")
            return data
        except Exception as e:
            print(f"❌ Status endpoint failed: {e}")
            return None

    def test_upload_endpoint(self):
        """Test the file upload endpoint"""
        print("🔍 Testing upload endpoint...")
        try:
            # Create a dummy PDF file for testing
            dummy_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n177\n%%EOF"
            
            files = {
                'files': ('test_report.pdf', dummy_pdf_content, 'application/pdf')
            }
            
            data = {
                'country_name': 'India',
                'entity_id': 'NTPC_002',
                'plant_name': 'NTPC Test Plant'
            }
            
            response = self.session.post(
                f"{self.base_url}/process/upload",
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
            
            if result['success'] and result['job_id']:
                print(f"✅ Upload processing started: Job ID {result['job_id']}")
                return result['job_id']
            else:
                print(f"❌ Upload processing failed: {result}")
                return None
                
        except Exception as e:
            print(f"❌ Upload endpoint failed: {e}")
            return None

    def wait_for_job_completion(self, job_id, timeout=30):
        """Wait for a job to complete"""
        print(f"⏳ Waiting for job {job_id} to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_data = self.test_status_endpoint(job_id)
            if not status_data:
                return False
                
            if status_data['status'] == 'completed':
                print(f"✅ Job {job_id} completed successfully!")
                return True
            elif status_data['status'] == 'failed':
                print(f"❌ Job {job_id} failed: {status_data.get('error', 'Unknown error')}")
                return False
                
            time.sleep(2)  # Wait 2 seconds before checking again
        
        print(f"⏰ Job {job_id} did not complete within {timeout} seconds")
        return False

    def test_results_endpoint(self, country_name, entity_id):
        """Test the results endpoint"""
        print(f"🔍 Testing results endpoint for {country_name}/{entity_id}...")
        try:
            response = self.session.get(f"{self.base_url}/results/{country_name}/{entity_id}")
            
            if response.status_code == 404:
                print(f"ℹ️  No results found for {country_name}/{entity_id} (expected for test)")
                return True
            
            response.raise_for_status()
            data = response.json()
            print(f"✅ Results endpoint successful: {len(data['results'])} files found")
            for filename, url in data['results'].items():
                print(f"   📄 {filename}: {url}")
            return True
            
        except Exception as e:
            print(f"❌ Results endpoint failed: {e}")
            return False

    def test_jobs_endpoint(self):
        """Test the jobs listing endpoint"""
        print("🔍 Testing jobs endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/jobs")
            response.raise_for_status()
            data = response.json()
            print(f"✅ Jobs endpoint successful: {data['total_jobs']} jobs found")
            return True
        except Exception as e:
            print(f"❌ Jobs endpoint failed: {e}")
            return False

    def run_full_test(self):
        """Run the complete test suite"""
        print("🚀 Starting Financial Pipeline API Test Suite")
        print("=" * 50)
        
        # Basic endpoint tests
        if not self.test_root_endpoint():
            return False
            
        if not self.test_health_endpoint():
            print("⚠️  Health check failed - API may not be properly configured")
        
        # Test websearch processing
        websearch_job_id = self.test_websearch_endpoint()
        if websearch_job_id:
            self.wait_for_job_completion(websearch_job_id, timeout=60)
            # Test results after processing
            self.test_results_endpoint("India", "NTPC_001")
        
        # Test upload processing
        upload_job_id = self.test_upload_endpoint()
        if upload_job_id:
            self.wait_for_job_completion(upload_job_id, timeout=60)
            # Test results after processing
            self.test_results_endpoint("India", "NTPC_002")
        
        # Test utility endpoints
        self.test_jobs_endpoint()
        
        print("=" * 50)
        print("✅ Test suite completed!")

def main():
    """Main test function"""
    print("Financial Pipeline API Test Script")
    print("Make sure the API server is running on http://localhost:8000")
    print("Start server with: python financial_pipeline_api.py")
    print()
    
    # Check if API server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print("❌ API server responded with error")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API server. Make sure it's running.")
        print("Start with: python financial_pipeline_api.py")
        return
    
    # Run tests
    tester = APITester()
    tester.run_full_test()

if __name__ == "__main__":
    main()