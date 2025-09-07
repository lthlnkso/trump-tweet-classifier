#!/bin/bash
# Production deployment script for Trump Tweet Classifier
# This script starts the application with production-ready configuration

set -e

echo "🚀 Starting Trump Tweet Classifier in Production Mode"
echo "=================================================="

# Activate virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found! Please run: python3 -m venv venv"
    exit 1
fi

# Install production dependencies
echo "📋 Installing production dependencies..."
pip install gunicorn --quiet

# Create necessary directories
mkdir -p logs data

# Set production environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export ENVIRONMENT="production"
export LOG_LEVEL="INFO"

# Check if model exists and get the latest one
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

# Backup database before starting
if [ -f "data/trump_classifier.db" ]; then
    echo "💾 Creating database backup..."
    cp data/trump_classifier.db "data/backup_$(date +%Y%m%d_%H%M%S).db"
fi

echo "🚀 Starting production server with 4 workers..."
echo "   • Server will run on http://0.0.0.0:8000"
echo "   • Access logs: logs/access.log"  
echo "   • Error logs: logs/error.log"
echo "   • Stop with: pkill -f gunicorn"

# Start with gunicorn for better concurrency
gunicorn api:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --log-level info \
    --pid gunicorn.pid \
    --daemon

# Wait a moment for startup
sleep 2

# Check if server started successfully
if pgrep -f gunicorn > /dev/null; then
    echo "✅ Production server started successfully!"
    echo "📊 Monitor with: tail -f logs/access.log"
    echo "🔍 Health check: curl http://localhost:8000/health"
    echo "🌐 Frontend: http://localhost:8000"
    echo ""
    echo "📈 Performance Tips:"
    echo "   • Expected capacity: 200-400 concurrent users"
    echo "   • Monitor logs for performance issues"
    echo "   • Use Cloudflare for additional CDN/security"
else
    echo "❌ Failed to start production server"
    echo "📋 Check error logs: cat logs/error.log"
    exit 1
fi
