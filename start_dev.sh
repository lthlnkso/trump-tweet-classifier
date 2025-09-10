#!/bin/bash
# Development server script for Trump Tweet Classifier
# Runs on port 8001 to avoid conflicts with production (port 8000)

set -e

echo "🔧 Starting Trump Tweet Classifier Development Server"
echo "===================================================="

# Activate virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found! Please run: python3 -m venv venv"
    exit 1
fi

# Check if model exists
MODEL_FILE=$(ls -t models/trump_classifier_*.joblib 2>/dev/null | head -1)
if [ -z "$MODEL_FILE" ]; then
    echo "❌ No trump_classifier model files found in models/ directory!"
    echo "📋 Available models:"
    ls -la models/*.joblib 2>/dev/null || echo "   No .joblib files found"
    exit 1
else
    echo "✅ Using model: $MODEL_FILE"
    export MODEL_PATH="$MODEL_FILE"
fi

# Set development environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export ENVIRONMENT="development"
export LOG_LEVEL="INFO"

echo "🚀 Starting development server on port 8001..."
echo "   • Development server: http://localhost:8001"
echo "   • API documentation: http://localhost:8001/docs"
echo "   • Metrics dashboard: http://localhost:8001/metrics"
echo "   • Stop with: Ctrl+C"
echo ""
echo "📊 Note: Production server continues on port 8000"
echo ""

# Start uvicorn development server with auto-reload
uvicorn api:app \
    --host 0.0.0.0 \
    --port 8001 \
    --reload \
    --reload-include="*.py" \
    --reload-include="frontend/*.html" \
    --reload-include="frontend/*.js" \
    --log-level info


