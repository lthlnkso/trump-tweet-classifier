# 🛠️ Developer Guide

This document describes the technical architecture, patterns, and implementation details of the Trump Tweet Classifier project. Use this as your guide for understanding, maintaining, and extending the codebase.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Core Components](#core-components)
4. [API Structure](#api-structure)
5. [Database Schema](#database-schema)
6. [Logging System](#logging-system)
7. [Frontend Architecture](#frontend-architecture)
8. [ML Pipeline](#ml-pipeline)
9. [File Organization](#file-organization)
10. [Development Patterns](#development-patterns)
11. [Extension Points](#extension-points)
12. [Common Tasks](#common-tasks)

## 🎯 Project Overview

**Type**: ML-powered web application  
**Framework**: FastAPI (backend) + Vanilla JS (frontend)  
**Database**: SQLite3 with direct SQL queries  
**ML Stack**: XGBoost + sentence-transformers + custom feature engineering  
**Deployment**: Single-file API server with static file serving  

### Key Features
- Real-time text classification with confidence scoring
- Persistent analytics and user feedback collection
- Viral sharing system with dynamic OpenGraph generation
- Responsive UI with animations and mobile optimization
- Comprehensive logging and error handling

## 🏗️ Architecture & Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │───▶│   FastAPI        │───▶│   ML Model      │
│   (Static)      │    │   Router         │    │   (XGBoost)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │              ┌─────────▼─────────┐             │
         │              │   Database        │             │
         │              │   (SQLite)        │             │
         │              └───────────────────┘             │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Browser       │    │   Logging        │    │   Image Gen     │
│   Clipboard     │    │   System         │    │   (PIL)         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Request Flow
1. **User Input** → Frontend form submission
2. **API Request** → `/classify` endpoint with text + session_id
3. **ML Processing** → Vectorization + XGBoost prediction
4. **Database Log** → Store submission, user data, results
5. **Response** → JSON with classification, confidence, submission_id
6. **UI Update** → Modal display with results and sharing options

## 🔧 Core Components

### FastAPI Application (`api.py`)
**Purpose**: Main application server handling all HTTP requests

**Key Patterns**:
- **Lifespan Context Manager**: Database initialization and cleanup
- **Dependency Injection**: Request logging, IP extraction
- **Static File Serving**: Jinja2 templates for frontend
- **Error Handling**: Try-catch with detailed logging

**Critical Functions**:
```python
@app.post("/classify")  # Main classification endpoint
@app.post("/feedback")  # User feedback collection
@app.post("/share")     # Viral sharing link creation
@app.get("/share/{hash}") # Share page rendering
@app.get("/admin/stats") # Analytics dashboard
```

### Database Layer (`database.py`)
**Purpose**: SQLite persistence with automatic schema management

**Design Principles**:
- **Direct SQL**: No ORM, uses sqlite3 directly for simplicity
- **Migration System**: `_apply_migrations()` handles schema evolution
- **Connection Management**: Context managers for proper cleanup
- **Data Integrity**: Transactions for multi-table operations

**Schema Pattern**:
```sql
-- Core tables
submissions    -- Every classification request
feedback       -- User rating and comments  
usage_stats    -- Aggregated analytics

-- Extension pattern for new features
CREATE TABLE new_feature (
    id INTEGER PRIMARY KEY,
    submission_id TEXT REFERENCES submissions(id),
    -- feature-specific columns
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Logging System (`logging_config.py`)
**Purpose**: Comprehensive application monitoring

**Log Structure**:
- **File Rotation**: Hourly files with 2-week retention
- **Log Levels**: INFO for normal operations, ERROR for exceptions
- **Structured Format**: Timestamp, module, function, message
- **Performance Tracking**: Request timing, database operations

**Usage Pattern**:
```python
from logging_config import setup_logging, log_request
logger = setup_logging()

# In endpoints
log_request(request, "classify", {"user_count": 42})
logger.info(f"Classification completed: {result}")
```

## 🌐 API Structure

### Endpoint Categories

#### **Core Functionality**
- `POST /classify` - Text classification (main feature)
- `GET /health` - Service health check
- `GET /` - Frontend application

#### **User Engagement**  
- `POST /feedback` - Collect user ratings
- `POST /share` - Create shareable links
- `GET /share/{hash}` - Render share pages

#### **Administrative**
- `GET /admin/stats` - Usage analytics
- Static file serving for frontend assets

### Request/Response Patterns

**Standard Request Model**:
```python
class ClassificationRequest(BaseModel):
    text: str = Field(..., max_length=600)
    session_id: Optional[str] = None
```

**Standard Response Model**:
```python
class ClassificationResponse(BaseModel):
    classification: int        # 0 or 1
    confidence: float         # 0.0 to 1.0
    level: str               # "CERTIFIED Trump", etc.
    submission_id: str       # For feedback/sharing
```

**Error Handling**:
- All endpoints use try-catch with logging
- Return 500 with generic message for security
- Log detailed errors for debugging

## 🗄️ Database Schema

### Core Tables

#### `submissions`
```sql
CREATE TABLE submissions (
    id TEXT PRIMARY KEY,           -- UUID for submissions
    user_ip TEXT,                 -- Client IP (hashed for privacy)
    user_agent TEXT,              -- Browser identification
    session_id TEXT,              -- Frontend session tracking
    text TEXT NOT NULL,           -- User input text
    classification INTEGER,        -- Model output (0/1)
    confidence REAL,              -- Model confidence (0.0-1.0)
    level TEXT,                   -- Human-readable level
    processing_time_ms INTEGER,   -- Performance metric
    created_at TIMESTAMP,         -- When submitted
    -- Sharing system columns
    is_public BOOLEAN DEFAULT 0,  -- Can be shared publicly
    share_hash TEXT,              -- Unique sharing identifier
    share_title TEXT,             -- OpenGraph title
    share_description TEXT        -- OpenGraph description
);
```

#### `feedback`
```sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    submission_id TEXT REFERENCES submissions(id),
    user_ip TEXT,
    agree BOOLEAN NOT NULL,       -- True = "Spot On!", False = "Nah..."
    message TEXT,                 -- Optional user comment
    created_at TIMESTAMP
);
```

#### `usage_stats`
```sql
CREATE TABLE usage_stats (
    id INTEGER PRIMARY KEY,
    date DATE,                    -- Daily aggregation
    total_submissions INTEGER,
    unique_users INTEGER,
    avg_confidence REAL,
    trump_percentage REAL,        -- % classified as Trump-like
    created_at TIMESTAMP
);
```

### Extension Pattern

**Adding New Features**:
1. Create new table with `submission_id` foreign key
2. Add migration in `_apply_migrations()`
3. Create data access methods in `database.py`
4. Add API endpoints in `api.py`

## 📊 Logging System

### Log File Structure
```
logs/
├── app_2024_01_15_14.log      # Hourly rotation
├── app_2024_01_15_15.log
└── ...                        # 2-week retention
```

### Log Categories

#### **Request Logging**
```python
# Format: timestamp - module - level - function - message
2024-01-15 14:30:22,123 - api - INFO - classify_text - Classification request from 192.168.1.100
```

#### **Performance Logging**
```python
log_performance("classification", processing_time_ms, {"confidence": 0.85})
```

#### **Database Operations**
```python
log_database_operation("INSERT", "submissions", {"success": True, "duration_ms": 15})
```

#### **Error Logging**
```python
logger.error(f"Classification failed: {str(e)}", exc_info=True)
```

### Monitoring Patterns
- **Request Tracking**: Every API call logged with timing
- **Error Aggregation**: Failed requests with stack traces
- **Performance Metrics**: Database query timing, ML inference time
- **User Behavior**: Classification patterns, feedback rates

## 🎨 Frontend Architecture

### File Structure
```
frontend/
├── index.html          # Main application
├── share.html          # Viral sharing pages
├── script.js           # Interactive functionality
├── images/             # Static assets
│   ├── TrumpChallenge.png
│   ├── CERTIFIEDTrump.png
│   └── share/          # Dynamic share images
└── README.md
```

### JavaScript Architecture

**No Framework Approach**: Vanilla JS with modern browser APIs

**Core Patterns**:
```javascript
// Global state management
let currentSubmissionId = null;
let currentShareUrl = null;

// API communication
async function classifyTweet() {
    const response = await fetch('/classify', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text, session_id})
    });
}

// UI state management  
function displayResult(result) {
    // Update modal with classification
    // Reset animations
    // Store result for sharing/feedback
}
```

**Responsive Design**:
- **Mobile-first**: Bootstrap 5 grid system
- **Viewport units**: `vh`/`vw` for consistent sizing
- **Breakpoints**: Desktop vs mobile layouts
- **Touch-friendly**: Large buttons, accessible controls

### Animation System
- **CSS Keyframes**: Floating effects, glowing text
- **JavaScript Triggers**: Modal display, feedback success
- **Performance**: GPU-accelerated transforms
- **Accessibility**: Respects `prefers-reduced-motion`

## 🤖 ML Pipeline

### Model Architecture

**XGBoost Classifier** with enhanced feature engineering:

```python
# Feature combination
embeddings = sentence_transformer.encode(text)     # 384-dim semantic
readability = textstat.flesch_reading_ease(text)   # Readability score  
caps_percent = len(re.findall(r'[A-Z]', text)) / len(text)  # CAPS usage

features = np.concatenate([embeddings, [readability, caps_percent]])
```

### Training Pipeline (`train_classifier.py`)

**Command-line Interface**:
```bash
python3 train_classifier.py --classifier xgboost --features
python3 compare_models.py  # Model comparison
```

**Evaluation Metrics**:
- Accuracy, Precision, Recall, F1-Score
- ROC AUC, Average Precision
- Per-class performance breakdown
- Confusion matrix visualization

### Vectorization (`vectorize.py`)

**TextVectorizer Class**:
```python
vectorizer = TextVectorizer(
    model_name='all-MiniLM-L6-v2',  # sentence-transformers model
    include_text_features=True       # Add readability + CAPS
)

features = vectorizer.vectorize([text])  # Returns combined feature vector
```

### Prediction Pipeline
1. **Text Input** → Preprocessing (cleanup, normalization)
2. **Feature Extraction** → Embeddings + text statistics  
3. **Model Inference** → XGBoost probability prediction
4. **Post-processing** → Convert to confidence score and level

## 📁 File Organization

### Root Level
```
├── api.py              # Main FastAPI application
├── database.py         # SQLite persistence layer
├── logging_config.py   # Logging configuration
├── image_generator.py  # Dynamic image creation
├── requirements.txt    # Python dependencies
├── dev.md             # This file
└── README.md          # User documentation
```

### ML Components
```
├── train_classifier.py    # Model training with metrics
├── vectorize.py           # Feature extraction
├── compare_models.py      # Model comparison utility
├── predict.py            # Standalone prediction script
└── models/               # Trained model storage
    ├── .gitkeep
    └── *.joblib          # Serialized models (excluded from git)
```

### Data & Logs
```
├── data/                 # Database files
├── logs/                 # Application logs (auto-created)
└── *.csv                # Training data (excluded from git)
```

### Scripts & Utilities
```
├── start_api.sh          # Server startup script
├── start_app.sh          # Alternative startup
├── client.py            # API testing client
└── create_test_subset.py # Data preparation
```

## 🔄 Development Patterns

### Code Organization Principles

#### **Separation of Concerns**
- **API Logic**: Route handling, request validation
- **Business Logic**: Classification, scoring, level assignment
- **Data Access**: Database operations, logging
- **Presentation**: Frontend UI, animations, sharing

#### **Error Handling Strategy**
```python
try:
    # Operation
    result = perform_operation()
    logger.info(f"Operation successful: {result}")
    return {"status": "success", "data": result}
except SpecificException as e:
    logger.error(f"Known error in {function_name}: {e}")
    return {"status": "error", "message": "User-friendly message"}
except Exception as e:
    logger.error(f"Unexpected error in {function_name}: {e}", exc_info=True)
    return {"status": "error", "message": "Something went wrong"}
```

#### **Database Patterns**
```python
# Transaction pattern
def complex_operation(data):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO table1 ...")
            cursor.execute("INSERT INTO table2 ...")
            conn.commit()
            log_database_operation("complex_operation", "SUCCESS")
        except Exception as e:
            conn.rollback()
            log_database_operation("complex_operation", "FAILED", str(e))
            raise
```

#### **API Response Patterns**
```python
# Consistent response structure
{
    "status": "success|error",
    "data": {...},           # Only on success
    "message": "...",        # Error message or success note
    "metadata": {            # Optional: timing, pagination, etc.
        "processing_time_ms": 150,
        "submission_id": "abc123"
    }
}
```

### Testing Patterns

**Test File Naming**: `test_*.py` (excluded from git)

**Common Test Scenarios**:
```python
# API endpoint testing
def test_classify_endpoint():
    response = client.post("/classify", json={"text": "Test tweet"})
    assert response.status_code == 200
    assert "confidence" in response.json()

# Database operations
def test_log_submission():
    submission_id = db.log_submission(...)
    assert submission_id is not None
    
# ML pipeline  
def test_vectorizer():
    features = vectorizer.vectorize(["Test text"])
    assert features.shape[1] == expected_feature_count
```

## 🚀 Extension Points

### Adding New Endpoints

1. **Define Models** (if needed):
```python
class NewFeatureRequest(BaseModel):
    field1: str
    field2: Optional[int] = None

class NewFeatureResponse(BaseModel):
    result: str
    metadata: dict
```

2. **Create Database Schema**:
```python
# In database.py _apply_migrations()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS new_feature (
        id INTEGER PRIMARY KEY,
        submission_id TEXT REFERENCES submissions(id),
        feature_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
```

3. **Add API Endpoint**:
```python
@app.post("/new-feature", response_model=NewFeatureResponse)
async def new_feature_endpoint(
    request: NewFeatureRequest,
    http_request: Request
):
    try:
        # Process request
        result = process_new_feature(request)
        
        # Log to database
        db.log_new_feature(result)
        
        # Log request
        log_request(http_request, "new_feature", {"success": True})
        
        return NewFeatureResponse(result=result)
    except Exception as e:
        logger.error(f"New feature failed: {e}")
        raise HTTPException(status_code=500, detail="Feature unavailable")
```

### Adding New ML Models

1. **Extend Vectorizer**:
```python
# In vectorize.py
def vectorize_for_new_model(self, texts):
    # Different feature extraction
    return features
```

2. **Create Training Script**:
```python
# new_model_trainer.py
class NewModelClassifier:
    def train(self, X, y):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic  
        return predictions
```

3. **Integrate into API**:
```python
# Load multiple models
trump_classifier = load_model("trump_model.joblib")
new_classifier = load_model("new_model.joblib")

@app.post("/classify-advanced")
async def advanced_classify(request: ClassificationRequest):
    # Use multiple models
    trump_result = trump_classifier.predict(features)
    new_result = new_classifier.predict(features)
    return {"trump": trump_result, "new_feature": new_result}
```

### Adding New Frontend Features

1. **Extend JavaScript**:
```javascript
// In script.js
function newFeature() {
    // New UI functionality
}

// Add event listeners
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('new-button').addEventListener('click', newFeature);
});
```

2. **Add HTML Elements**:
```html
<!-- In index.html -->
<div class="new-feature-section">
    <button id="new-button" class="btn btn-primary">New Feature</button>
</div>
```

3. **Style with CSS**:
```css
.new-feature-section {
    /* Responsive styling */
}
```

### Database Schema Evolution

**Migration Pattern**:
```python
def _apply_migrations(self):
    cursor = self.connection.cursor()
    
    # Version 1: Add new column
    try:
        cursor.execute("ALTER TABLE submissions ADD COLUMN new_field TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Version 2: Add new table  
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS new_table (
            id INTEGER PRIMARY KEY,
            related_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Version 3: Add index
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_new_table_related 
        ON new_table(related_id)
    """)
```

## 🔧 Common Tasks

### Local Development

**Start Development Server**:
```bash
# With auto-reload
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Or use script
./start_api.sh
```

**Database Operations**:
```bash
# View database
sqlite3 data/trump_classifier.db
.tables
.schema submissions
SELECT COUNT(*) FROM submissions;
```

**Log Monitoring**:
```bash
# Real-time log watching
tail -f logs/app_$(date +%Y_%m_%d_%H).log

# Error analysis
grep ERROR logs/*.log | head -20
```

### Model Development

**Train New Model**:
```bash
# Ensure training data exists
ls -la *.csv

# Train with different configurations
python3 train_classifier.py --classifier xgboost --features
python3 train_classifier.py --classifier logistic --no-features

# Compare results
python3 compare_models.py
```

**Test Model Performance**:
```bash
# Standalone prediction
python3 predict.py "Your test tweet here"

# API testing
python3 client.py
```

### Deployment

**Production Server**:
```bash
# Multi-worker production setup
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# With process management
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker
```

**Database Backup**:
```bash
# Backup database
cp data/trump_classifier.db data/backup_$(date +%Y%m%d_%H%M%S).db

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

### Performance Monitoring

**Check Resource Usage**:
```bash
# Database size
du -h data/trump_classifier.db

# Log directory size  
du -h logs/

# Memory usage
ps aux | grep python
```

**Analyze Performance**:
```python
# Database query performance
import sqlite3
conn = sqlite3.connect('data/trump_classifier.db')
conn.execute("EXPLAIN QUERY PLAN SELECT * FROM submissions WHERE created_at > ?", (date,))
```

### Troubleshooting

**Common Issues**:

1. **Database Lock**: 
   ```bash
   # Check for hanging connections
   lsof | grep trump_classifier.db
   ```

2. **Model Loading Errors**:
   ```bash
   # Verify model file
   ls -la models/*.joblib
   python3 -c "import joblib; print(joblib.load('models/model.joblib'))"
   ```

3. **Frontend Issues**:
   ```bash
   # Check static file serving
   curl http://localhost:8000/static/script.js
   ```

4. **Performance Issues**:
   ```bash
   # Check log for slow requests
   grep "processing_time_ms" logs/*.log | sort -k5 -n | tail -10
   ```

---

## 📝 Notes for Future Development

### Architecture Decisions Made
- **Single-file API**: Keep it simple, FastAPI handles everything
- **Direct SQL**: No ORM to avoid complexity for simple operations  
- **Vanilla JS**: No frontend framework to minimize dependencies
- **SQLite**: Perfect for MVP, can migrate to PostgreSQL later
- **File-based logging**: Simple rotation, easy to parse

### Known Technical Debt
- **Large CSV files**: Need proper data management solution
- **No rate limiting**: Should add for production deployment
- **Basic authentication**: Admin endpoints need protection
- **Error boundaries**: Frontend needs better error handling
- **Mobile UX**: Some animations could be optimized

### Future Considerations
- **Containerization**: Add Docker for easier deployment
- **API versioning**: Plan for backwards compatibility
- **Real-time features**: WebSocket support for live updates
- **Analytics dashboard**: More sophisticated admin interface
- **A/B testing**: Framework for experimenting with UI changes

This development guide should serve as your north star for understanding and extending the Trump Tweet Classifier. Update this file as the project evolves!
