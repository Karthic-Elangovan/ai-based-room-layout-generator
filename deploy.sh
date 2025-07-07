#!/bin/bash

# Financial Pipeline API Deployment Script
# This script helps deploy the Financial Pipeline API

set -e

echo "🚀 Financial Pipeline API Deployment Script"
echo "=============================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install dependencies
install_dependencies() {
    echo "📦 Installing Python dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ requirements.txt not found"
        exit 1
    fi
}

# Function to setup environment
setup_environment() {
    echo "🔧 Setting up environment..."
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            echo "📋 Copying .env.example to .env"
            cp .env.example .env
            echo "⚠️  Please edit .env file with your actual credentials before running the API"
        else
            echo "❌ No .env.example file found"
            exit 1
        fi
    else
        echo "✅ .env file already exists"
    fi
}

# Function to test API
test_api() {
    echo "🧪 Testing API..."
    if [ -f "test_api.py" ]; then
        echo "Running basic syntax check..."
        python -m py_compile financial_pipeline_api.py
        echo "✅ API file compiled successfully"
        echo "ℹ️  To run full tests, start the API first and then run: python test_api.py"
    else
        echo "⚠️  test_api.py not found, skipping tests"
    fi
}

# Function to start API in development mode
start_dev() {
    echo "🔥 Starting API in development mode..."
    echo "API will be available at: http://localhost:8000"
    echo "API documentation: http://localhost:8000/docs"
    echo "Press Ctrl+C to stop"
    python financial_pipeline_api.py
}

# Function to start API in production mode
start_prod() {
    echo "🏭 Starting API in production mode..."
    if command_exists uvicorn; then
        uvicorn financial_pipeline_api:app --host 0.0.0.0 --port 8000 --workers 4
    else
        echo "❌ uvicorn not found. Installing..."
        pip install uvicorn
        uvicorn financial_pipeline_api:app --host 0.0.0.0 --port 8000 --workers 4
    fi
}

# Function to deploy with Docker
deploy_docker() {
    echo "🐳 Deploying with Docker..."
    if command_exists docker; then
        if [ -f "Dockerfile" ]; then
            echo "Building Docker image..."
            docker build -t financial-pipeline-api .
            echo "Starting container..."
            docker run -d -p 8000:8000 --name financial-api financial-pipeline-api
            echo "✅ Docker container started successfully"
            echo "API available at: http://localhost:8000"
        else
            echo "❌ Dockerfile not found"
            exit 1
        fi
    else
        echo "❌ Docker not found. Please install Docker first"
        exit 1
    fi
}

# Function to deploy with Docker Compose
deploy_compose() {
    echo "🐳 Deploying with Docker Compose..."
    if command_exists docker-compose; then
        if [ -f "docker-compose.yml" ]; then
            echo "Starting services..."
            docker-compose up -d
            echo "✅ Services started successfully"
            echo "API available at: http://localhost:8000"
        else
            echo "❌ docker-compose.yml not found"
            exit 1
        fi
    else
        echo "❌ Docker Compose not found. Please install Docker Compose first"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup         Setup environment and install dependencies"
    echo "  dev           Start API in development mode"
    echo "  prod          Start API in production mode"
    echo "  docker        Deploy using Docker"
    echo "  compose       Deploy using Docker Compose"
    echo "  test          Run tests"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup      # Setup environment"
    echo "  $0 dev        # Start development server"
    echo "  $0 docker     # Deploy with Docker"
}

# Main script logic
case "${1:-}" in
    setup)
        setup_environment
        install_dependencies
        test_api
        echo ""
        echo "🎉 Setup completed successfully!"
        echo "Next steps:"
        echo "1. Edit .env file with your credentials"
        echo "2. Run: $0 dev (for development) or $0 prod (for production)"
        ;;
    dev)
        start_dev
        ;;
    prod)
        start_prod
        ;;
    docker)
        deploy_docker
        ;;
    compose)
        deploy_compose
        ;;
    test)
        test_api
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        echo "⚠️  No command specified"
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_help
        exit 1
        ;;
esac