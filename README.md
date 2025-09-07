# 🎯 Trump Tweet Classifier

**The "Post Like Trump" Challenge** - An AI-powered web app that analyzes your writing style and gives you a "Trump Score" from 0-100! Can you write like the 45th-47th President?

## 🚀 Features

### 🤖 AI Classification
- **XGBoost Machine Learning Model** trained on real tweet data
- **Advanced Feature Engineering**: Sentence embeddings + readability + CAPS percentage
- **Trump-o-meter Levels**: From "Tiffany Trump" (0-20) to "CERTIFIED Trump" (95-100)
- **Confidence Scoring**: See how certain the AI is about your Trump-ness

### 🎨 Beautiful UI
- **Animated Results Modal** with floating effects and pastel colors
- **Responsive Design** - works perfectly on mobile and desktop  
- **Trump Challenge Images** - visual feedback based on your score
- **Fun Animations** - keeps things engaging and Trump-y

### 📊 Analytics & Persistence
- **SQLite Database** - tracks all submissions and user feedback
- **Comprehensive Logging** - hourly rotation, 2-week retention
- **User Feedback System** - "Spot On!" or "Nah..." with optional messages
- **Usage Statistics** - admin dashboard with user metrics

### 🔗 Viral Sharing System
- **Shareable Results** - each score gets a unique share link
- **Rich OpenGraph Tags** - beautiful previews on social media
- **Dynamic Images** - auto-generated share images based on your score
- **Social Media Integration** - direct Twitter/Facebook sharing
- **Reliable Clipboard** - works on HTTP, mobile, any browser

## 🛠️ Tech Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **XGBoost** - Gradient boosting machine learning
- **sentence-transformers** - Advanced NLP embeddings
- **SQLite3** - Lightweight database for persistence
- **Pillow** - Dynamic image generation
- **Jinja2** - Template rendering

### Frontend  
- **Vanilla JavaScript** - No framework bloat
- **Bootstrap 5** - Responsive CSS framework
- **Font Awesome** - Beautiful icons
- **CSS Animations** - Custom floating and glow effects

### ML Pipeline
- **textstat** - Readability analysis
- **scikit-learn** - Model evaluation and metrics
- **joblib** - Model serialization
- **Custom vectorizer** - Combines multiple feature types

## 🏃‍♂️ Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/lthlnkso/trump-tweet-classifier.git
cd trump-tweet-classifier
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Get Training Data
The training data CSV files are too large for GitHub (150MB+). You'll need to:
- **Option A**: Use your own tweet dataset in CSV format with columns: `text`, `is_trump` (boolean)
- **Option B**: Contact the maintainer for access to the training data
- **Option C**: Use a smaller sample dataset for testing

### 3. Train a Model (Optional)
```bash
# Train XGBoost with enhanced features (requires training data)
python3 train_classifier.py --classifier xgboost --features

# Compare different models
python3 compare_models.py
```

### 4. Start the Server
```bash
# Option 1: Direct FastAPI
python3 api.py

# Option 2: Using start script
chmod +x start_api.sh
./start_api.sh
```

### 5. Open Your Browser
Navigate to `http://localhost:8000` and start posting like Trump! 🎯

## 📁 Project Structure

```
TT/
├── 🤖 ML Core
│   ├── vectorize.py           # Text feature extraction
│   ├── train_classifier.py    # Model training with metrics
│   └── compare_models.py      # Model comparison utility
│
├── 🌐 Web App
│   ├── api.py                 # FastAPI backend
│   ├── database.py            # SQLite persistence layer
│   ├── logging_config.py      # Comprehensive logging
│   └── image_generator.py     # Dynamic share images
│
├── 🎨 Frontend
│   ├── frontend/
│   │   ├── index.html         # Main app interface
│   │   ├── share.html         # Viral sharing page
│   │   ├── script.js          # Interactive functionality
│   │   └── images/            # Trump challenge images
│
├── 📊 Models & Data
│   ├── models/                # Trained model files (.joblib)
│   ├── logs/                  # Application logs (hourly rotation)
│   └── trump_classifier.db    # SQLite database
│
└── 🛠️ Config
    ├── requirements.txt       # Python dependencies
    ├── start_api.sh          # Server startup script
    └── test_*.py             # Comprehensive test suite
```

## 🎮 Usage Examples

### Basic Classification
```python
from vectorize import TextVectorizer
from train_classifier import TrumpTweetClassifier

# Load trained model
classifier = TrumpTweetClassifier.load_model('models/trump_classifier.joblib')

# Classify text
result = classifier.predict("The fake news media is totally out of control!")
print(f"Trump Score: {result['confidence']:.1f}%")
print(f"Level: {result['level']}")
```

### API Usage
```bash
# Classify text via API
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Tremendous success! The best deal ever made!", "session_id": "test123"}'

# Submit feedback
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{"submission_id": "abc123", "agree": true, "message": "Spot on!"}'

# Create shareable link
curl -X POST "http://localhost:8000/share" \
  -H "Content-Type: application/json" \
  -d '{"submission_id": "abc123"}'
```

## 🎯 Trump-o-meter Levels

| Score Range | Level | Image | Description |
|-------------|-------|-------|-------------|
| 95-100 | **CERTIFIED Trump** | `CERTIFIEDTrump.png` | You've mastered the art of the deal! |
| 80-94 | **Donald Trump Jr.** | `trump-thumbs-up.png` | Following in dad's footsteps! |
| 65-79 | **Eric Trump** | `eric-trump.png` | Strong Trump energy! |
| 50-64 | **Trump Supporter** | `trump-supporter.png` | You're getting there! |
| 35-49 | **Tiffany Trump** | `tiffany-trump.png` | Some Trump vibes... |
| 0-34 | **Not Trump** | `trump-disappointed.png` | Try again! |

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
PORT=8000
HOST=0.0.0.0

# Database
DATABASE_URL=sqlite:///trump_classifier.db

# Logging
LOG_LEVEL=INFO
LOG_RETENTION_DAYS=14
```

### Model Training Options
```bash
# Train different classifiers
python3 train_classifier.py --classifier logistic    # Logistic Regression
python3 train_classifier.py --classifier xgboost     # XGBoost (recommended)

# Feature engineering options
python3 train_classifier.py --features               # Include text features
python3 train_classifier.py --no-features            # Embeddings only
```

## 📊 Performance Metrics

Our best model (XGBoost with enhanced features) achieves:
- **Accuracy**: ~85%
- **Precision**: 0.82 (Trump), 0.87 (Non-Trump)
- **Recall**: 0.86 (Trump), 0.84 (Non-Trump)
- **F1-Score**: 0.84 (Trump), 0.85 (Non-Trump)

## 🚀 Deployment

### Local Development
```bash
# Development server with auto-reload
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
# Production server
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Coming Soon)
```dockerfile
# Future: Docker deployment option
FROM python:3.9-slim
# ... Docker configuration
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

## 📈 Roadmap

- [ ] **Docker deployment** configuration
- [ ] **Real-time updates** with WebSockets
- [ ] **User accounts** and personal statistics
- [ ] **Tournament mode** - compete with friends
- [ ] **Advanced analytics** dashboard
- [ ] **Mobile app** (React Native)
- [ ] **Multi-language** support
- [ ] **Celebrity classifier** expansion

## 🐛 Known Issues

- Clipboard API requires HTTPS (workaround: text field with copy buttons)
- Large model files not included in repo (train your own or request access)
- Rate limiting not implemented (add for production use)

## 📜 License

MIT License - Feel free to use this for educational purposes, build upon it, or deploy your own Trump classifier!

## 🙏 Acknowledgments

- **OpenAI** for inspiring AI applications
- **Trump Twitter Archive** for training data
- **Bootstrap & Font Awesome** for beautiful UI components
- **The Python ML community** for amazing tools

---

**Made with 💻 and a sense of humor by [@lthlnkso](https://github.com/lthlnkso)**

*"This is going to be tremendous, believe me!" 🎯*
