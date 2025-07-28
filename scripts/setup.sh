#!/bin/bash

# ===================================
# Amazon Fashion Search Engine
# Quick Setup Script
# ===================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_step() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    print_header "Amazon Fashion Search Engine - Setup"
    
    # Check prerequisites
    echo "🔍 Checking prerequisites..."
    
    if ! command_exists "python3"; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_step "Python $python_version found"
    
    if ! command_exists "node"; then
        print_error "Node.js is required but not installed."
        exit 1
    fi
    
    node_version=$(node --version)
    print_step "Node.js $node_version found"
    
    # Check for uv package manager
    if command_exists "uv"; then
        print_step "uv package manager found"
        USE_UV=true
    else
        print_warning "uv not found, using pip instead"
        USE_UV=false
    fi
    
    # Environment setup
    print_header "Environment Setup"
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.template" ]; then
            cp .env.template .env
            print_step "Created .env file from template"
            print_warning "Please edit .env file and add your OPENAI_API_KEY"
        else
            print_error ".env.template not found"
            exit 1
        fi
    else
        print_step ".env file already exists"
    fi
    
    # Check for OpenAI API key
    if grep -q "your_openai_api_key_here" .env; then
        print_warning "Please set your OPENAI_API_KEY in .env file"
        echo "You can get an API key from: https://platform.openai.com/api-keys"
    fi
    
    # Python dependencies
    print_header "Installing Python Dependencies"
    
    if [ "$USE_UV" = true ]; then
        print_step "Installing with uv..."
        uv install
    else
        print_step "Installing with pip..."
        pip install -e .
    fi
    
    # Frontend dependencies
    print_header "Installing Frontend Dependencies"
    
    cd frontend
    
    if [ -f "yarn.lock" ]; then
        print_step "Installing with yarn..."
        yarn install
    elif [ -f "package-lock.json" ]; then
        print_step "Installing with npm..."
        npm install
    else
        print_step "Installing with npm..."
        npm install
    fi
    
    cd ..
    
    # Data check
    print_header "Checking Data Files"
    
    if [ -f "data/processed/processed_products.parquet" ] && [ -f "data/embeddings/embeddings.npy" ]; then
        print_step "Preloaded data found - ready to use!"
    else
        print_warning "Preloaded data not found"
        echo "Options:"
        echo "  1. Use preloaded data (fastest): Download data files"
        echo "  2. Generate sample data: python services/data-pipeline/main.py --rebuild"
        echo "  3. Generate full data: python services/data-pipeline/main.py --rebuild --full"
    fi
    
    # Final instructions
    print_header "Setup Complete!"
    
    echo "🚀 To start the application:"
    echo ""
    echo "Option 1: Manual start (recommended for development)"
    echo "  Terminal 1: python services/search-api/main.py"
    echo "  Terminal 2: cd frontend && npm run dev"
    echo ""
    echo "Option 2: Docker (recommended for production)"
    echo "  docker-compose up"
    echo ""
    echo "📖 Access the application:"
    echo "  Frontend: http://localhost:3000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  Health:   http://localhost:8000/health"
    echo ""
    print_step "Setup completed successfully!"
}

# Run main function
main "$@"