"""
Module for converting text into vector embeddings using SentenceTransformers.

This module uses the static-retrieval-mrl-en-v1 model to generate embeddings
for text input, combined with additional text features like readability metrics
and capitalization percentage for downstream classification tasks.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import textstat
import re
from typing import Union, List


class TextVectorizer:
    """
    A class for vectorizing text using the static-retrieval-mrl-en-v1 model
    combined with additional text features.
    """
    
    def __init__(self, include_text_features: bool = True):
        """
        Initialize the vectorizer with the specified model.
        
        Args:
            include_text_features: Whether to include readability and caps features
        """
        self.model_name = "static-retrieval-mrl-en-v1"
        self.model = None
        self.include_text_features = include_text_features
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _extract_text_features(self, text: str) -> np.ndarray:
        """
        Extract additional text features beyond embeddings.
        
        Args:
            text: Input text
            
        Returns:
            numpy.ndarray: Array of text features [readability_score, caps_percent]
        """
        # Readability score using Flesch Reading Ease
        try:
            readability = textstat.flesch_reading_ease(text)
        except:
            readability = 0.0  # Default if calculation fails
        
        # Percentage of uppercase characters
        if len(text) > 0:
            caps_count = sum(1 for c in text if c.isupper())
            caps_percent = caps_count / len(text) * 100
        else:
            caps_percent = 0.0
        
        return np.array([readability, caps_percent], dtype=np.float32)
    
    def vectorize(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text into vector embeddings combined with text features.
        
        Args:
            text: A string or list of strings to vectorize
            
        Returns:
            numpy.ndarray: The embedding vector(s) with optional text features
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please initialize the vectorizer first.")
        
        # Handle single string input
        single_input = isinstance(text, str)
        if single_input:
            text = [text]
        
        # Generate embeddings
        embeddings = self.model.encode(text, convert_to_numpy=True)
        
        # Add text features if enabled
        if self.include_text_features:
            text_features_list = []
            for txt in text:
                features = self._extract_text_features(txt)
                text_features_list.append(features)
            
            text_features = np.array(text_features_list)
            
            # Combine embeddings with text features
            combined_features = np.concatenate([embeddings, text_features], axis=1)
        else:
            combined_features = embeddings
        
        # Return single vector if input was single string, otherwise return array
        if single_input:
            return combined_features[0]
        return combined_features


def vectorize_text(text: str) -> np.ndarray:
    """
    Convenience function to vectorize a single text string.
    
    Args:
        text: The text string to vectorize
        
    Returns:
        numpy.ndarray: The embedding vector for the input text
    """
    vectorizer = TextVectorizer()
    return vectorizer.vectorize(text)


# Example usage
if __name__ == "__main__":
    # Test the vectorizer
    sample_text = "This is a sample tweet to test our vectorizer."
    
    # Method 1: Using the class
    vectorizer = TextVectorizer()
    vector = vectorizer.vectorize(sample_text)
    print(f"Vector shape: {vector.shape}")
    print(f"First 5 dimensions: {vector[:5]}")
    
    # Method 2: Using the convenience function
    vector2 = vectorize_text(sample_text)
    print(f"Vectors are identical: {np.array_equal(vector, vector2)}")

