#!/bin/bash

# Test script for Fashion Search Frontend
echo "🧪 Testing Fashion Search Frontend Setup"
echo "======================================="

# Check if we're in the workspace root
if [ ! -d "frontend" ]; then
    echo "❌ Please run this script from the workspace root directory."
    exit 1
fi

cd frontend

# Check Node.js version
echo "📋 Checking Node.js version..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "✅ Node.js: $NODE_VERSION"
    
    # Check if version is 16+
    MAJOR_VERSION=$(echo $NODE_VERSION | sed 's/v\([0-9]*\).*/\1/')
    if [ "$MAJOR_VERSION" -lt 16 ]; then
        echo "⚠️  Warning: Node.js 16+ recommended (you have $NODE_VERSION)"
    fi
else
    echo "❌ Node.js not found. Please install Node.js 16+"
    exit 1
fi

# Check npm
echo "📋 Checking npm..."
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo "✅ npm: $NPM_VERSION"
else
    echo "❌ npm not found"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install --silent

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Check if all required files exist
echo "📋 Checking required files..."
REQUIRED_FILES=(
    "package.json"
    "src/App.js"
    "src/index.js"
    "src/services/api.js"
    "src/components/Header.js"
    "src/components/SearchBar.js"
    "src/components/ProductResults.js"
    "src/components/StatusIndicator.js"
    "public/index.html"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Test build
echo "🏗️  Testing production build..."
npm run build --silent

if [ $? -eq 0 ]; then
    echo "✅ Production build successful"
    echo "📁 Build output in: frontend/build/"
else
    echo "❌ Production build failed"
    exit 1
fi

# Check API connection (optional)
echo "🔍 Testing API connection..."
if curl -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is running and reachable"
else
    echo "⚠️  API not running (this is optional for frontend testing)"
fi

echo ""
echo "🎉 Frontend setup test completed successfully!"
echo ""
echo "To start the development server:"
echo "  cd frontend"
echo "  npm start"
echo ""
echo "Or use the convenience script:"
echo "  ./frontend/start-frontend.sh"
echo ""
echo "The frontend will be available at: http://localhost:3000"