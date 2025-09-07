#!/bin/bash

# Start script for Trump Tweet Classifier API

echo "🚀 Starting Trump Tweet Classifier API..."
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if model exists
MODEL_FILE="models/trump_classifier_20250906_214735.joblib"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Model file not found: $MODEL_FILE"
    echo "Please train a model first using: python train_classifier.py"
    exit 1
fi

echo "✅ Model found: $MODEL_FILE"
echo "🔄 Starting API server on http://0.0.0.0:8000"
echo "📚 API documentation available at: http://0.0.0.0:8000/docs"
echo "❤️  Health check: http://0.0.0.0:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"

# Start the API server
python api.py

