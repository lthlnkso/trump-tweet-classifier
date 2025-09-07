"""
Script to load a trained Trump tweet classifier and make predictions.

Usage:
    python predict.py "Your tweet text here"
    python predict.py --model models/your_model.joblib "Your tweet text here"
"""

import joblib
import argparse
import os
from typing import Tuple


def load_model(model_path: str) -> dict:
    """
    Load a trained classifier model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dictionary containing model components
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    required_components = ['classifier', 'scaler', 'vectorizer']
    for component in required_components:
        if component not in model:
            raise ValueError(f"Model file is missing required component: {component}")
    
    print(f"Model loaded: {model.get('model_name', 'Unknown')}")
    print(f"Training timestamp: {model.get('training_timestamp', 'Unknown')}")
    
    return model


def predict_tweet(model: dict, text: str) -> Tuple[str, float]:
    """
    Predict whether a tweet is from Trump.
    
    Args:
        model: Loaded model dictionary
        text: Tweet text to classify
        
    Returns:
        Tuple of (prediction_label, confidence)
    """
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
    
    # Convert to human-readable label
    label = "Trump" if prediction == 1 else "Non-Trump"
    
    return label, confidence


def find_latest_model(models_dir: str = "models") -> str:
    """
    Find the most recently created model file.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Path to the latest model file
    """
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    # Sort by modification time and return the latest
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    latest_model = os.path.join(models_dir, model_files[0])
    
    return latest_model


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict Trump tweets using trained classifier')
    parser.add_argument(
        'text',
        type=str,
        help='Tweet text to classify'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model file (defaults to latest model in models/ directory)'
    )
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        model_path = find_latest_model()
    
    # Load model
    model = load_model(model_path)
    
    # Make prediction
    prediction, confidence = predict_tweet(model, args.text)
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Text: '{args.text}'")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    
    if prediction == "Trump":
        print("\n🟠 This tweet appears to be written by Donald Trump")
    else:
        print("\n🔵 This tweet appears to be written by someone other than Donald Trump")


if __name__ == "__main__":
    main()

