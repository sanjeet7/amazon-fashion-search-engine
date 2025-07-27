#!/bin/bash

# Fashion Search Frontend Startup Script
echo "🚀 Starting Fashion Search Frontend..."
echo "=================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    echo "❌ Please run this script from the frontend directory."
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies."
        exit 1
    fi
fi

# Check if API is running
echo "🔍 Checking if API is running on http://localhost:8000..."
if curl -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is running and healthy!"
else
    echo "⚠️  Warning: API doesn't seem to be running on localhost:8000"
    echo "   Make sure to start the search API first:"
    echo "   cd services/search-api && python main.py"
    echo ""
    echo "   Continuing anyway... you can start the API later."
fi

echo ""
echo "🌐 Starting development server..."
echo "   Frontend will be available at: http://localhost:3000"
echo "   API should be running at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

# Start the development server
npm start