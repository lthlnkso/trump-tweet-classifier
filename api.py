"""
FastAPI service for Trump Tweet Classification

This service provides a REST API for classifying tweets as Trump or non-Trump
using a pre-trained machine learning model with sentence embeddings.

Features:
- Efficient model loading at startup (no reload per request)
- Fast classification endpoint
- Comprehensive error handling
- API documentation via OpenAPI/Swagger
- Health check endpoint
"""

import os
import joblib
import logging
import time
import uuid
import hashlib
import urllib.parse
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

# Import our custom modules
from database import db
from logging_config import log_request, log_performance, log_database_operation, get_client_ip
from image_generator import image_gen
from analytics import TrumpAnalytics

# Configure logging
logger = logging.getLogger(__name__)

# Setup templates
templates = Jinja2Templates(directory="frontend")

# Global model storage
model_store = {}

class ClassificationRequest(BaseModel):
    """Request model for tweet classification."""
    text: str = Field(
        ..., 
        description="Tweet text to classify",
        min_length=1,
        max_length=600,  # Updated to match frontend limit
        example="The FAKE NEWS media is the enemy of the American People!"
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for tracking user sessions"
    )

class ClassificationResponse(BaseModel):
    """Response model for tweet classification."""
    text: str = Field(description="Original input text")
    prediction: str = Field(description="Classification result: 'Trump' or 'Non-Trump'")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")
    is_trump: bool = Field(description="Boolean indicator if classified as Trump tweet")
    trump_level: str = Field(description="Trump-o-meter level based on confidence")
    trump_score: int = Field(description="Trump score out of 100")
    submission_id: int = Field(description="Unique ID for this submission (for feedback)")

class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    submission_id: Optional[int] = Field(
        None,
        description="ID of the submission being rated (optional)"
    )
    agrees_with_rating: bool = Field(
        ...,
        description="Whether user agrees with the classification"
    )
    feedback_message: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional feedback message from user"
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for tracking"
    )

class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool = Field(description="Whether feedback was recorded successfully")
    feedback_id: int = Field(description="Unique ID for the feedback record")
    message: str = Field(description="Confirmation message")

class ShareRequest(BaseModel):
    """Request model for creating a shareable link."""
    submission_id: int = Field(description="ID of the submission to share")
    
class ShareResponse(BaseModel):
    """Response model for share link creation."""
    success: bool = Field(description="Whether share link was created successfully")
    share_url: str = Field(description="The shareable URL")
    share_hash: str = Field(description="Unique hash for the share")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded")
    model_name: Optional[str] = Field(description="Name of the loaded model")

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str = Field(description="Name of the loaded model")
    training_timestamp: str = Field(description="When the model was trained")
    model_path: str = Field(description="Path to the model file")
    vector_dimensions: int = Field(description="Dimensionality of text embeddings")

def load_model(model_path: str) -> Dict[str, Any]:
    """
    Load the trained classifier model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dictionary containing model components
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is invalid
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        model = joblib.load(model_path)
        
        # Validate model components
        required_components = ['classifier', 'scaler', 'vectorizer']
        for component in required_components:
            if component not in model:
                raise ValueError(f"Model file is missing required component: {component}")
        
        logger.info(f"Model loaded successfully: {model.get('model_name', 'Unknown')}")
        logger.info(f"Training timestamp: {model.get('training_timestamp', 'Unknown')}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise ValueError(f"Failed to load model: {e}")

def generate_share_hash(submission_id: int, timestamp: str) -> str:
    """Generate a unique share hash for a submission."""
    data = f"{submission_id}_{timestamp}_{uuid.uuid4().hex[:8]}"
    return hashlib.md5(data.encode()).hexdigest()[:12]

def create_share_urls(base_url: str, share_hash: str, share_text: str, trump_score: int) -> Dict[str, str]:
    """Create social media share URLs."""
    share_url = f"{base_url}/share/{share_hash}"
    
    # Customize share text based on score
    if trump_score >= 85:
        share_text = f"🔥 I scored {trump_score}% on the Trump Tweet Challenge! I'm basically Certified Trump! Can you beat my score?"
    elif trump_score >= 70:
        share_text = f"🎯 I scored {trump_score}% on the Trump Tweet Challenge! Not bad for a non-politician! Think you can do better?"
    elif trump_score >= 50:
        share_text = f"🤔 I scored {trump_score}% on the Trump Tweet Challenge. I'm getting there... Can you write more like 45-47?"
    else:
        share_text = f"😅 I scored {trump_score}% on the Trump Tweet Challenge. Definitely not Trump material! Can you do better?"
    
    twitter_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}&url={urllib.parse.quote(share_url)}"
    facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={urllib.parse.quote(share_url)}"
    
    return {
        "twitter": twitter_url,
        "facebook": facebook_url,
        "share_url": share_url
    }

def get_trump_level(confidence: float, is_trump: bool) -> tuple[str, int]:
    """
    Determine Trump-o-meter level based on confidence score.
    
    Args:
        confidence: Model confidence (0.0 to 1.0)
        is_trump: Whether classified as Trump
        
    Returns:
        Tuple of (level_name, score_out_of_100)
    """
    # Convert confidence to a score out of 100
    if is_trump:
        score = int(confidence * 100)
    else:
        # For non-Trump classifications, invert the confidence
        score = int((1 - confidence) * 100)
    
    # Trump-o-meter levels (generous scoring as requested)
    if score >= 95:
        return "Certified Trump 🏆", score
    elif score >= 85:
        return "Donald Trump Jr. 👔", score
    elif score >= 70:
        return "Eric Trump 🏗️", score
    elif score >= 55:
        return "Tiffany Trump 💎", score
    elif score >= 40:
        return "Trump Supporter 🇺🇸", score
    elif score >= 25:
        return "Trump Curious 🤔", score
    else:
        return "Definitely Not Trump 🚫", score

def classify_text(text: str, user_ip: str, user_agent: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify a tweet text using the loaded model and log to database.
    
    Args:
        text: Tweet text to classify
        user_ip: Client IP address
        user_agent: Client user agent
        session_id: Optional session identifier
        
    Returns:
        Dictionary with classification results including submission_id
        
    Raises:
        ValueError: If model is not loaded or classification fails
    """
    start_time = time.time()
    
    if 'model' not in model_store:
        raise ValueError("Model not loaded")
    
    model = model_store['model']
    
    try:
        # Extract model components
        classifier = model['classifier']
        scaler = model['scaler']
        vectorizer = model['vectorizer']
        
        # Vectorize the text
        vector = vectorizer.vectorize(text)
        
        # Scale the features
        vector_scaled = scaler.transform(vector.reshape(1, -1))
        
        # Make prediction
        prediction = classifier.predict(vector_scaled)[0]
        probabilities = classifier.predict_proba(vector_scaled)[0]
        confidence = probabilities.max()
        
        # Convert to human-readable format
        is_trump = bool(prediction == 1)
        prediction_label = "Trump" if is_trump else "Non-Trump"
        
        # Get Trump-o-meter level
        trump_level, trump_score = get_trump_level(confidence, is_trump)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log to database
        try:
            submission_id = db.log_submission(
                user_ip=user_ip,
                user_agent=user_agent,
                text_content=text,
                classification=prediction_label,
                confidence=float(confidence),
                trump_level=trump_level,
                trump_score=trump_score,
                processing_time_ms=processing_time_ms,
                session_id=session_id
            )
            log_database_operation("INSERT", "submissions", processing_time_ms)
        except Exception as db_error:
            logger.error(f"Failed to log submission to database: {db_error}")
            log_database_operation("INSERT", "submissions", error=str(db_error))
            submission_id = -1  # Fallback ID
        
        # Log performance
        log_performance("classification", processing_time_ms, {
            "text_length": len(text),
            "trump_score": trump_score,
            "confidence": f"{confidence:.3f}"
        })
        
        return {
            "text": text,
            "prediction": prediction_label,
            "confidence": float(confidence),
            "is_trump": is_trump,
            "trump_level": trump_level,
            "trump_score": trump_score,
            "submission_id": submission_id
        }
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error during classification: {e}")
        log_performance("classification_error", processing_time_ms, {"error": str(e)})
        raise ValueError(f"Classification failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown.
    Loads the model at startup to avoid loading it on every request.
    """
    # Startup
    default_model_path = "models/trump_classifier_20250907_082520.joblib"
    model_path = os.getenv("MODEL_PATH", default_model_path)
    
    try:
        logger.info("Starting up Trump Tweet Classifier API...")
        model = load_model(model_path)
        model_store['model'] = model
        model_store['model_path'] = model_path
        logger.info("Model loaded successfully at startup")
        
        # Initialize database and analytics
        db.init_db()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model at startup: {e}")
        # Continue startup but mark model as not loaded
        model_store['model'] = None
        model_store['error'] = str(e)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Trump Tweet Classifier API...")
    model_store.clear()

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Trump Tweet Classifier API",
    description="API service for classifying tweets as written by Donald Trump or not using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    """Serve the frontend application."""
    return FileResponse('frontend/index.html')

@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API information endpoint."""
    return {
        "message": "Trump Tweet Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "frontend": "/"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = 'model' in model_store and model_store['model'] is not None
    model_name = None
    
    if model_loaded:
        model_name = model_store['model'].get('model_name', 'Unknown')
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_name=model_name
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if 'model' not in model_store or model_store['model'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = model_store['model']
    
    # Get vector dimensions from the vectorizer
    try:
        # Create a test vector to get dimensions
        test_vector = model['vectorizer'].vectorize("test")
        vector_dimensions = len(test_vector)
    except:
        vector_dimensions = 1024  # Default for static-retrieval-mrl-en-v1
    
    return ModelInfo(
        model_name=model.get('model_name', 'Unknown'),
        training_timestamp=model.get('training_timestamp', 'Unknown'),
        model_path=model_store.get('model_path', 'Unknown'),
        vector_dimensions=vector_dimensions
    )

@app.post("/classify", response_model=ClassificationResponse)
async def classify_tweet(request: ClassificationRequest, http_request: Request):
    """
    Classify a tweet as Trump or non-Trump.
    
    This endpoint takes tweet text and returns a classification along with
    a confidence score and logs the submission to database.
    """
    start_time = time.time()
    client_ip = get_client_ip(http_request)
    user_agent = http_request.headers.get("User-Agent", "Unknown")
    
    if 'model' not in model_store or model_store['model'] is None:
        error_msg = model_store.get('error', 'Model not loaded')
        response_time_ms = (time.time() - start_time) * 1000
        log_request(http_request, response_time_ms, 503)
        raise HTTPException(status_code=503, detail=f"Service unavailable: {error_msg}")
    
    try:
        result = classify_text(
            text=request.text,
            user_ip=client_ip,
            user_agent=user_agent,
            session_id=request.session_id
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        log_request(http_request, response_time_ms, 200)
        
        return ClassificationResponse(**result)
        
    except ValueError as e:
        response_time_ms = (time.time() - start_time) * 1000
        log_request(http_request, response_time_ms, 400)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Unexpected error during classification: {e}")
        log_request(http_request, response_time_ms, 500)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, http_request: Request):
    """
    Submit user feedback on a classification result.
    
    This endpoint allows users to indicate whether they agree with the 
    classification and provide optional feedback messages.
    """
    start_time = time.time()
    client_ip = get_client_ip(http_request)
    
    try:
        feedback_id = db.log_feedback(
            submission_id=request.submission_id,
            user_ip=client_ip,
            agrees_with_rating=request.agrees_with_rating,
            feedback_message=request.feedback_message,
            session_id=request.session_id
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        log_database_operation("INSERT", "feedback", response_time_ms)
        log_request(http_request, response_time_ms, 200)
        
        logger.info(f"Feedback received - ID: {feedback_id}, Agrees: {request.agrees_with_rating}, IP: {client_ip}")
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Thank you for your feedback!"
        )
        
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error submitting feedback: {e}")
        log_database_operation("INSERT", "feedback", error=str(e))
        log_request(http_request, response_time_ms, 500)
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@app.post("/classify/batch")
async def classify_batch(texts: list[str]):
    """
    Classify multiple tweets at once for better performance.
    
    Note: This endpoint accepts a list of strings for batch processing.
    """
    if 'model' not in model_store or model_store['model'] is None:
        error_msg = model_store.get('error', 'Model not loaded')
        raise HTTPException(status_code=503, detail=f"Service unavailable: {error_msg}")
    
    if len(texts) > 100:  # Reasonable batch size limit
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    try:
        results = []
        for text in texts:
            result = classify_text(text)
            results.append(result)
        
        return {"results": results, "count": len(results)}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during batch classification: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/share", response_model=ShareResponse)
async def create_share_link(request: ShareRequest, http_request: Request):
    """
    Create a shareable link for a submission result.
    """
    start_time = time.time()
    
    try:
        # Generate share hash and metadata
        timestamp = datetime.now().isoformat()
        share_hash = generate_share_hash(request.submission_id, timestamp)
        
        # Get submission details from database
        submission = db.get_recent_submissions(limit=1000)  # This is inefficient, we'll improve it
        target_submission = None
        for sub in submission:
            if sub['id'] == request.submission_id:
                target_submission = sub
                break
        
        if not target_submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Create share metadata
        trump_score = target_submission['trump_score']
        trump_level = target_submission['trump_level']
        
        share_title = f"I scored {trump_score}% on the Trump Tweet Challenge! 🎯"
        share_description = f"Level: {trump_level} - Can you write like 45-47? Test your Trump tweeting skills!"
        
        # Make submission shareable
        success = db.make_submission_shareable(
            submission_id=request.submission_id,
            share_hash=share_hash,
            share_title=share_title,
            share_description=share_description
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create share link")
        
        # Generate share image
        try:
            image_path = image_gen.generate_share_image(
                trump_score=trump_score,
                trump_level=trump_level,
                classification=target_submission['classification'],
                text_preview=target_submission['text_content'][:100],
                share_hash=share_hash
            )
            logger.info(f"Generated share image: {image_path}")
        except Exception as img_error:
            logger.warning(f"Failed to generate share image: {img_error}")
        
        # Create share URL using request headers to get the actual host
        scheme = http_request.headers.get("x-forwarded-proto", "http" if "localhost" in str(http_request.base_url) else "https")
        host = http_request.headers.get("host", http_request.client.host if hasattr(http_request, 'client') else "localhost:8000")
        
        # Construct the proper base URL
        base_url = f"{scheme}://{host}"
        share_url = f"{base_url}/share/{share_hash}"
        
        logger.info(f"Generated share URL: {share_url}")  # Debug log
        
        response_time_ms = (time.time() - start_time) * 1000
        log_request(http_request, response_time_ms, 200)
        
        return ShareResponse(
            success=True,
            share_url=share_url,
            share_hash=share_hash
        )
        
    except HTTPException:
        raise
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error creating share link: {e}")
        log_request(http_request, response_time_ms, 500)
        raise HTTPException(status_code=500, detail="Failed to create share link")

@app.get("/share/{share_hash}", response_class=HTMLResponse)
async def view_shared_result(share_hash: str, request: Request):
    """
    Display a shared result page with OpenGraph metadata.
    """
    try:
        # Get submission data
        submission = db.get_submission_by_share_hash(share_hash)
        if not submission:
            raise HTTPException(status_code=404, detail="Shared result not found")
        
        # Prepare template data
        base_url = str(request.base_url).rstrip('/')
        share_url = f"{base_url}/share/{share_hash}"
        
        # Create social share URLs
        share_urls = create_share_urls(base_url, share_hash, "", submission['trump_score'])
        
        # Determine prediction color
        prediction_color = "#32CD32" if submission['classification'] == "Trump" else "#FF6B6B"
        
        # Format date
        created_date = datetime.fromisoformat(submission['created_at']).strftime("%B %d, %Y")
        
        template_data = {
            "request": request,
            "trump_score": submission['trump_score'],
            "trump_level": submission['trump_level'],
            "text_content": submission['text_content'],
            "classification": submission['classification'],
            "confidence": f"{submission['confidence'] * 100:.1f}",
            "prediction_color": prediction_color,
            "created_date": created_date,
            "share_hash": share_hash,
            "twitter_share_url": share_urls['twitter'],
            "facebook_share_url": share_urls['facebook'],
            
            # OpenGraph data
            "og_title": submission.get('share_title', f"I scored {submission['trump_score']}% on the Trump Tweet Challenge!"),
            "og_description": submission.get('share_description', f"Level: {submission['trump_level']} - Can you write like 45-47?"),
            "og_image": f"{base_url}/static/images/share/trump_score_{share_hash}_{submission['trump_level'].lower().replace(' ', '_')}.png",
            "og_url": share_url,
            "page_title": f"Trump Tweet Challenge - {submission['trump_score']}% Score"
        }
        
        return templates.TemplateResponse("share.html", template_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error displaying shared result: {e}")
        raise HTTPException(status_code=500, detail="Failed to load shared result")

@app.get("/admin/stats")
async def get_admin_stats():
    """
    Get administrative statistics (for monitoring/debugging).
    Note: In production, this should be protected with authentication.
    """
    try:
        stats = {
            "submissions": db.get_submission_stats(days=7),
            "feedback": db.get_feedback_stats(days=7),
            "sharing": db.get_sharing_stats(),
            "recent_submissions": db.get_recent_submissions(limit=10)
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.get("/metrics", response_class=HTMLResponse)
async def metrics_dashboard(request: Request):
    """
    Comprehensive metrics dashboard with visualizations.
    Note: In production, this should be protected with authentication.
    """
    try:
        analytics = TrumpAnalytics()
        
        # Get basic statistics
        basic_stats = analytics.get_basic_stats()
        
        # Generate all plots
        plots = {}
        
        # Hourly submissions plot
        try:
            plots['hourly_submissions'] = analytics.get_hourly_submissions(days=7)
        except Exception as e:
            logger.warning(f"Failed to generate hourly submissions plot: {e}")
            plots['hourly_submissions'] = None
        
        # Trump level distribution
        try:
            plots['trump_levels'] = analytics.get_trump_level_distribution()
        except Exception as e:
            logger.warning(f"Failed to generate Trump levels plot: {e}")
            plots['trump_levels'] = None
        
        # Confidence analysis
        try:
            plots['confidence_analysis'] = analytics.get_confidence_analysis()
        except Exception as e:
            logger.warning(f"Failed to generate confidence analysis plot: {e}")
            plots['confidence_analysis'] = None
        
        # Geographic analysis
        try:
            plots['geographic_analysis'] = analytics.get_geographic_analysis()
        except Exception as e:
            logger.warning(f"Failed to generate geographic analysis plot: {e}")
            plots['geographic_analysis'] = None
        
        # Feedback analysis
        try:
            plots['feedback_analysis'] = analytics.get_feedback_analysis()
        except Exception as e:
            logger.warning(f"Failed to generate feedback analysis plot: {e}")
            plots['feedback_analysis'] = None
        
        # Sharing analysis
        try:
            plots['sharing_analysis'] = analytics.get_sharing_analysis()
        except Exception as e:
            logger.warning(f"Failed to generate sharing analysis plot: {e}")
            plots['sharing_analysis'] = None
        
        # Render the dashboard template
        context = {
            "request": request,
            "basic_stats": basic_stats,
            "plots": plots
        }
        
        return templates.TemplateResponse("metrics.html", context)
        
    except Exception as e:
        logger.error(f"Error generating metrics dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics dashboard")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
