"""
Simple client script for testing the Trump Tweet Classifier API.

Usage:
    python client.py "Your tweet text here"
    python client.py --batch "Tweet 1" "Tweet 2" "Tweet 3"
    python client.py --health
"""

import requests
import json
import argparse
import sys
from typing import List


class TrumpClassifierClient:
    """Client for interacting with the Trump Tweet Classifier API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API service
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> dict:
        """Check the health of the API service."""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to API service. Is it running?"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to API service. Is it running?"}
        except Exception as e:
            return {"error": str(e)}
    
    def classify(self, text: str) -> dict:
        """
        Classify a single tweet.
        
        Args:
            text: Tweet text to classify
            
        Returns:
            Classification result
        """
        try:
            response = requests.post(
                f"{self.base_url}/classify",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to API service. Is it running?"}
        except Exception as e:
            return {"error": str(e)}
    
    def classify_batch(self, texts: List[str]) -> dict:
        """
        Classify multiple tweets at once.
        
        Args:
            texts: List of tweet texts to classify
            
        Returns:
            Batch classification results
        """
        try:
            response = requests.post(
                f"{self.base_url}/classify/batch",
                json=texts,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to API service. Is it running?"}
        except Exception as e:
            return {"error": str(e)}


def print_classification_result(result: dict, is_batch: bool = False):
    """Pretty print classification results."""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    if is_batch:
        print(f"\n📊 Batch Classification Results ({result['count']} tweets):")
        print("=" * 60)
        for i, res in enumerate(result['results'], 1):
            emoji = "🟠" if res['is_trump'] else "🔵"
            print(f"{i}. {emoji} {res['prediction']} (confidence: {res['confidence']:.4f})")
            print(f"   Text: '{res['text'][:80]}{'...' if len(res['text']) > 80 else ''}'")
            print()
    else:
        emoji = "🟠" if result['is_trump'] else "🔵"
        print(f"\n{emoji} Classification Result:")
        print("=" * 40)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Text: '{result['text']}'")


def main():
    """Main client function."""
    parser = argparse.ArgumentParser(description='Trump Tweet Classifier API Client')
    parser.add_argument(
        'texts',
        nargs='*',
        help='Tweet text(s) to classify'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple texts as a batch'
    )
    parser.add_argument(
        '--health',
        action='store_true',
        help='Check API health status'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Get model information'
    )
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='API base URL (default: http://localhost:8000)'
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = TrumpClassifierClient(args.url)
    
    # Handle health check
    if args.health:
        health = client.health_check()
        if "error" in health:
            print(f"❌ Health Check Failed: {health['error']}")
            sys.exit(1)
        
        status_emoji = "✅" if health['status'] == 'healthy' else "❌"
        model_emoji = "✅" if health['model_loaded'] else "❌"
        
        print(f"\n{status_emoji} API Health Status:")
        print("=" * 25)
        print(f"Status: {health['status']}")
        print(f"Model Loaded: {model_emoji} {health['model_loaded']}")
        if health.get('model_name'):
            print(f"Model Name: {health['model_name']}")
        return
    
    # Handle model info
    if args.info:
        info = client.get_model_info()
        if "error" in info:
            print(f"❌ Model Info Failed: {info['error']}")
            sys.exit(1)
        
        print(f"\n📋 Model Information:")
        print("=" * 30)
        print(f"Model Name: {info['model_name']}")
        print(f"Training Time: {info['training_timestamp']}")
        print(f"Model Path: {info['model_path']}")
        print(f"Vector Dimensions: {info['vector_dimensions']}")
        return
    
    # Handle text classification
    if not args.texts:
        print("Error: Please provide text to classify or use --health/--info")
        parser.print_help()
        sys.exit(1)
    
    if args.batch or len(args.texts) > 1:
        # Batch classification
        result = client.classify_batch(args.texts)
        print_classification_result(result, is_batch=True)
    else:
        # Single classification
        result = client.classify(args.texts[0])
        print_classification_result(result)


if __name__ == "__main__":
    main()

