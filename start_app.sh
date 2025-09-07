#!/bin/bash

# Start Trump Tweet Classifier MVP
# This script starts both the API service and opens the frontend

echo "🚀 Starting Trump Tweet Classifier MVP..."
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create it first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if model file exists
MODEL_FILE="models/trump_classifier_20250907_082520.joblib"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Model file not found: $MODEL_FILE"
    echo "Please train the model first or check the path."
    exit 1
fi

# Start the API in the background
echo "🔧 Starting API service on http://localhost:8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait a moment for the API to start
echo "⏳ Waiting for API to start..."
sleep 3

# Check if API is running
if kill -0 $API_PID 2>/dev/null; then
    echo "✅ API service started successfully (PID: $API_PID)"
else
    echo "❌ Failed to start API service"
    exit 1
fi

# Open frontend in default browser
echo "🌐 Opening frontend in browser..."
FRONTEND_URL="http://localhost:8000"

# Try different browsers in order of preference
if command -v google-chrome &> /dev/null; then
    google-chrome "$FRONTEND_URL" &
elif command -v firefox &> /dev/null; then
    firefox "$FRONTEND_URL" &
elif command -v chromium-browser &> /dev/null; then
    chromium-browser "$FRONTEND_URL" &
elif command -v xdg-open &> /dev/null; then
    xdg-open "$FRONTEND_URL" &
else
    echo "⚠️  Could not detect browser. Please manually open: $FRONTEND_URL"
fi

echo ""
echo "🎉 Trump Tweet Classifier MVP is now running!"
echo "========================================"
echo "🔗 Frontend Application: http://localhost:8000"
echo "🔗 API Documentation: http://localhost:8000/docs"
echo "🔗 API Health Check: http://localhost:8000/health"
echo "🔗 API Info: http://localhost:8000/api"
echo ""
echo "💡 Tips:"
echo "   - Try typing 'The FAKE NEWS media is TERRIBLE!' for a high score"
echo "   - Use ALL CAPS and exclamation points for better Trump scores"
echo "   - Check the Network tab if the frontend can't connect to API"
echo ""
echo "🛑 To stop the service:"
echo "   - Press Ctrl+C to stop this script"
echo "   - Or run: kill $API_PID"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if kill -0 $API_PID 2>/dev/null; then
        kill $API_PID
        echo "✅ API service stopped"
    fi
    echo "👋 Goodbye!"
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Keep the script running and show logs
echo "📋 API Logs (Press Ctrl+C to stop):"
echo "====================================="
wait $API_PID
