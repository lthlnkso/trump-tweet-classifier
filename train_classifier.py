"""
Trump Tweet Classifier Training Script

This script handles the complete pipeline for training a logistic regression classifier
to distinguish between Trump and non-Trump tweets using sentence embeddings.

Features:
- Load CSV data with tweet text and labels
- 80/20 train/test split
- Text vectorization using sentence-transformers
- Logistic regression training
- Model evaluation with confusion matrix and metrics
- Model persistence to disk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
import argparse
from datetime import datetime
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from vectorize import TextVectorizer


class TrumpTweetClassifier:
    """
    A classifier for distinguishing Trump tweets from non-Trump tweets.
    """
    
    def __init__(self, random_state: int = 42, classifier_type: str = 'xgboost'):
        """
        Initialize the classifier.
        
        Args:
            random_state: Random seed for reproducibility
            classifier_type: Type of classifier ('xgboost' or 'logistic')
        """
        self.random_state = random_state
        self.classifier_type = classifier_type
        self.vectorizer = None
        self.scaler = None
        self.classifier = None
        self.is_trained = False
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load training data from CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Ensure we have the required columns
        if 'text' not in df.columns or 'is_trump' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'is_trump' columns")
        
        # Remove any rows with missing text
        df = df.dropna(subset=['text'])
        
        print(f"Loaded {len(df)} samples")
        print(f"Trump tweets: {df['is_trump'].sum()}")
        print(f"Non-Trump tweets: {(df['is_trump'] == False).sum()}")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets and vectorize the text.
        
        Args:
            df: Input dataframe
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"\nSplitting data into {int((1-test_size)*100)}/{int(test_size*100)} train/test...")
        
        # Split the data
        X_text = df['text'].values
        y = df['is_trump'].values.astype(int)  # Convert to int for classifier
        
        X_text_train, X_text_test, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {len(X_text_train)} samples")
        print(f"Test set: {len(X_text_test)} samples")
        
        # Initialize vectorizer
        print("\nVectorizing text data...")
        self.vectorizer = TextVectorizer()
        
        # Vectorize training data
        print("Vectorizing training set...")
        X_train = self.vectorizer.vectorize(X_text_train.tolist())
        
        # Vectorize test data
        print("Vectorizing test set...")
        X_test = self.vectorizer.vectorize(X_text_test.tolist())
        
        print(f"Vector dimensions: {X_train.shape[1]}")
        
        # Scale the features
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the classifier (XGBoost or Logistic Regression).
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.classifier_type == 'xgboost':
            print("\nTraining XGBoost classifier...")
            self.classifier = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            )
        else:
            print("\nTraining logistic regression classifier...")
            self.classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            )
        
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        print("Training completed!")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained classifier with comprehensive metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("\nEvaluating classifier...")
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=[0, 1]
        )
        
        # ROC and PR metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, target_names=['Non-Trump', 'Trump'])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            # Overall metrics
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            
            # Per-class metrics
            'non_trump_precision': precision_per_class[0],
            'non_trump_recall': recall_per_class[0],
            'non_trump_f1': f1_per_class[0],
            'non_trump_support': support_per_class[0],
            
            'trump_precision': precision_per_class[1],
            'trump_recall': recall_per_class[1],
            'trump_f1': f1_per_class[1],
            'trump_support': support_per_class[1],
            
            # Raw data
            'confusion_matrix': cm,
            'classification_report': class_report,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            
            # Model info
            'classifier_type': self.classifier_type,
            'feature_count': X_test.shape[1]
        }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None) -> None:
        """
        Plot and optionally save confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Non-Trump', 'Trump'],
            yticklabels=['Non-Trump', 'Trump']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def print_detailed_metrics(self, results: Dict[str, Any]) -> None:
        """
        Print comprehensive evaluation metrics.
        
        Args:
            results: Results dictionary from evaluate()
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)
        print(f"Classifier Type: {results['classifier_type'].upper()}")
        print(f"Feature Count: {results['feature_count']}")
        print("-"*60)
        
        # Overall metrics
        print("OVERALL METRICS:")
        print(f"  Accuracy:      {results['accuracy']:.4f}")
        print(f"  ROC AUC:       {results['roc_auc']:.4f}")
        print(f"  Avg Precision: {results['avg_precision']:.4f}")
        print(f"  F1-Score:      {results['f1_score']:.4f}")
        print()
        
        # Per-class metrics
        print("PER-CLASS METRICS:")
        print("                  Precision  Recall   F1-Score  Support")
        print(f"  Non-Trump:        {results['non_trump_precision']:.4f}   {results['non_trump_recall']:.4f}    {results['non_trump_f1']:.4f}    {results['non_trump_support']}")
        print(f"  Trump:            {results['trump_precision']:.4f}   {results['trump_recall']:.4f}    {results['trump_f1']:.4f}    {results['trump_support']}")
        print()
        
        # Confusion matrix breakdown
        cm = results['confusion_matrix']
        print("CONFUSION MATRIX BREAKDOWN:")
        print(f"  True Negatives (Non-Trump correctly classified): {cm[0, 0]}")
        print(f"  False Positives (Non-Trump misclassified as Trump): {cm[0, 1]}")
        print(f"  False Negatives (Trump misclassified as Non-Trump): {cm[1, 0]}")
        print(f"  True Positives (Trump correctly classified): {cm[1, 1]}")
        print()
        
        # Error analysis
        total_errors = cm[0, 1] + cm[1, 0]
        print("ERROR ANALYSIS:")
        print(f"  Total Errors: {total_errors}")
        print(f"  Non-Trump misclassified as Trump: {cm[0, 1]} ({cm[0, 1]/total_errors*100:.1f}% of errors)")
        print(f"  Trump misclassified as Non-Trump: {cm[1, 0]} ({cm[1, 0]/total_errors*100:.1f}% of errors)")
    
    def save_model(self, model_dir: str, model_name: str = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model_dir: Directory to save the model
            model_name: Optional model name (will generate timestamp-based name if not provided)
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"trump_classifier_{timestamp}"
        
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        
        # Create model package
        model_package = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'vectorizer': self.vectorizer,
            'random_state': self.random_state,
            'model_name': model_name,
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Save model
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model_package, model_path)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    def predict(self, text: str) -> Tuple[int, float]:
        """
        Predict whether a text is from Trump.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Vectorize and scale the text
        vector = self.vectorizer.vectorize(text)
        vector_scaled = self.scaler.transform(vector.reshape(1, -1))
        
        # Make prediction
        prediction = self.classifier.predict(vector_scaled)[0]
        confidence = self.classifier.predict_proba(vector_scaled)[0].max()
        
        return int(prediction), float(confidence)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Trump Tweet Classifier')
    parser.add_argument(
        '--data', 
        type=str, 
        default='test_subset.csv',
        help='Path to training data CSV file'
    )
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Fraction of data to use for testing'
    )
    parser.add_argument(
        '--model-dir', 
        type=str, 
        default='models',
        help='Directory to save trained model'
    )
    parser.add_argument(
        '--model-name', 
        type=str, 
        default=None,
        help='Name for the saved model'
    )
    parser.add_argument(
        '--save-plots', 
        action='store_true',
        help='Save evaluation plots'
    )
    parser.add_argument(
        '--classifier', 
        type=str, 
        choices=['xgboost', 'logistic'], 
        default='xgboost',
        help='Type of classifier to use'
    )
    parser.add_argument(
        '--no-text-features', 
        action='store_true',
        help='Disable additional text features (readability, caps percentage)'
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = TrumpTweetClassifier(classifier_type=args.classifier)
    
    # Initialize vectorizer with or without text features
    include_text_features = not args.no_text_features
    print(f"Using text features: {include_text_features}")
    print(f"Classifier type: {args.classifier}")
    
    # Load data
    df = classifier.load_data(args.data)
    
    # Override the vectorizer initialization to use our text features setting
    from vectorize import TextVectorizer
    classifier.vectorizer = TextVectorizer(include_text_features=include_text_features)
    
    # Prepare data (but skip vectorizer initialization since we set it manually)
    print(f"\nSplitting data into {int((1-args.test_size)*100)}/{int(args.test_size*100)} train/test...")
    
    # Split the data
    X_text = df['text'].values
    y = df['is_trump'].values.astype(int)
    
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text, y, test_size=args.test_size, random_state=classifier.random_state, stratify=y
    )
    
    print(f"Training set: {len(X_text_train)} samples")
    print(f"Test set: {len(X_text_test)} samples")
    
    # Vectorize training data
    print("\nVectorizing training set...")
    X_train = classifier.vectorizer.vectorize(X_text_train.tolist())
    
    # Vectorize test data
    print("Vectorizing test set...")
    X_test = classifier.vectorizer.vectorize(X_text_test.tolist())
    
    print(f"Vector dimensions: {X_train.shape[1]}")
    
    # Scale the features
    print("Scaling features...")
    classifier.scaler = StandardScaler()
    X_train = classifier.scaler.fit_transform(X_train)
    X_test = classifier.scaler.transform(X_test)
    
    # Train classifier
    classifier.train(X_train, y_train)
    
    # Evaluate classifier
    results = classifier.evaluate(X_test, y_test)
    
    # Print comprehensive results
    classifier.print_detailed_metrics(results)
    
    # Plot confusion matrix
    if args.save_plots:
        plot_path = os.path.join(args.model_dir, 'confusion_matrix.png')
        classifier.plot_confusion_matrix(results['confusion_matrix'], save_path=plot_path)
    else:
        classifier.plot_confusion_matrix(results['confusion_matrix'])
    
    # Save model
    model_path = classifier.save_model(args.model_dir, args.model_name)
    
    # Test prediction with sample
    sample_trump_text = "The FAKE NEWS media is the enemy of the American People!"
    sample_non_trump_text = "Just had a great coffee this morning, feeling good!"
    
    pred1, conf1 = classifier.predict(sample_trump_text)
    pred2, conf2 = classifier.predict(sample_non_trump_text)
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    print(f"Text: '{sample_trump_text}'")
    print(f"Prediction: {'Trump' if pred1 else 'Non-Trump'} (confidence: {conf1:.4f})")
    print()
    print(f"Text: '{sample_non_trump_text}'")
    print(f"Prediction: {'Trump' if pred2 else 'Non-Trump'} (confidence: {conf2:.4f})")
    
    print(f"\nTraining completed! Model saved to: {model_path}")


if __name__ == "__main__":
    main()

