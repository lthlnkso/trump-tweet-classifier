#!/usr/bin/env python3
"""
Model Comparison Utility for Trump Tweet Classification

This script systematically compares different model configurations to help
find the best performing classifier. It tests different combinations of:
- Classifier types (XGBoost vs Logistic Regression)
- Feature sets (with/without text features)
- Saves results for easy comparison

Usage:
    python compare_models.py --data test_subset.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

from train_classifier import TrumpTweetClassifier
from vectorize import TextVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ModelComparator:
    """
    Class for systematically comparing different model configurations.
    """
    
    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the model comparator.
        
        Args:
            data_path: Path to the training data CSV
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.results = []
        
        # Load and prepare base data once
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        
        if 'text' not in self.df.columns or 'is_trump' not in self.df.columns:
            raise ValueError("CSV must contain 'text' and 'is_trump' columns")
        
        self.df = self.df.dropna(subset=['text'])
        print(f"Loaded {len(self.df)} samples")
        print(f"Trump tweets: {self.df['is_trump'].sum()}")
        print(f"Non-Trump tweets: {(self.df['is_trump'] == False).sum()}")
        
        # Split data once
        X_text = self.df['text'].values
        y = self.df['is_trump'].values.astype(int)
        
        self.X_text_train, self.X_text_test, self.y_train, self.y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(self.X_text_train)} samples")
        print(f"Test set: {len(self.X_text_test)} samples")
    
    def run_single_experiment(self, classifier_type: str, include_text_features: bool) -> Dict[str, Any]:
        """
        Run a single experiment with specified configuration.
        
        Args:
            classifier_type: 'xgboost' or 'logistic'
            include_text_features: Whether to include readability/caps features
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {classifier_type.upper()} + {'Enhanced Features' if include_text_features else 'Embeddings Only'}")
        print(f"{'='*60}")
        
        # Initialize classifier
        classifier = TrumpTweetClassifier(
            random_state=self.random_state,
            classifier_type=classifier_type
        )
        
        # Initialize vectorizer with appropriate features
        vectorizer = TextVectorizer(include_text_features=include_text_features)
        classifier.vectorizer = vectorizer
        
        # Vectorize data
        print("Vectorizing training set...")
        X_train = vectorizer.vectorize(self.X_text_train.tolist())
        
        print("Vectorizing test set...")
        X_test = vectorizer.vectorize(self.X_text_test.tolist())
        
        print(f"Vector dimensions: {X_train.shape[1]}")
        
        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        classifier.scaler = scaler
        
        # Train classifier
        classifier.train(X_train_scaled, self.y_train)
        
        # Evaluate
        results = classifier.evaluate(X_test_scaled, self.y_test)
        
        # Print results
        classifier.print_detailed_metrics(results)
        
        # Prepare experiment summary
        experiment_result = {
            'timestamp': datetime.now().isoformat(),
            'classifier_type': classifier_type,
            'include_text_features': include_text_features,
            'feature_count': X_train.shape[1],
            'test_size': self.test_size,
            
            # Key metrics
            'accuracy': results['accuracy'],
            'roc_auc': results['roc_auc'],
            'avg_precision': results['avg_precision'],
            'f1_score': results['f1_score'],
            'precision': results['precision'],
            'recall': results['recall'],
            
            # Per-class metrics
            'non_trump_precision': results['non_trump_precision'],
            'non_trump_recall': results['non_trump_recall'],
            'non_trump_f1': results['non_trump_f1'],
            'trump_precision': results['trump_precision'],
            'trump_recall': results['trump_recall'],
            'trump_f1': results['trump_f1'],
            
            # Support
            'non_trump_support': int(results['non_trump_support']),
            'trump_support': int(results['trump_support']),
            
            # Confusion matrix
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
        
        return experiment_result
    
    def run_comparison(self) -> List[Dict[str, Any]]:
        """
        Run comparison across all model configurations.
        
        Returns:
            List of experiment results
        """
        configurations = [
            ('xgboost', True),   # XGBoost with enhanced features
            ('xgboost', False),  # XGBoost with embeddings only
            ('logistic', True),  # Logistic with enhanced features
            ('logistic', False), # Logistic with embeddings only
        ]
        
        results = []
        
        for classifier_type, include_text_features in configurations:
            try:
                result = self.run_single_experiment(classifier_type, include_text_features)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                print(f"Error in experiment {classifier_type} + {'enhanced' if include_text_features else 'base'}: {e}")
                continue
        
        return results
    
    def print_comparison_summary(self) -> None:
        """Print a summary comparison of all experiments."""
        if not self.results:
            print("No results to compare!")
            return
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Create comparison table
        df_results = pd.DataFrame(self.results)
        
        # Add configuration description
        df_results['config'] = df_results.apply(
            lambda x: f"{x['classifier_type'].upper()}+{'Enhanced' if x['include_text_features'] else 'Base'}", 
            axis=1
        )
        
        # Select key metrics for comparison
        comparison_cols = [
            'config', 'accuracy', 'roc_auc', 'f1_score', 
            'trump_precision', 'trump_recall', 'non_trump_precision', 'non_trump_recall'
        ]
        
        comparison_df = df_results[comparison_cols].round(4)
        
        print("\nKEY METRICS COMPARISON:")
        print(comparison_df.to_string(index=False))
        
        # Find best performers
        print(f"\n{'='*60}")
        print("BEST PERFORMERS:")
        print(f"{'='*60}")
        
        best_accuracy = df_results.loc[df_results['accuracy'].idxmax()]
        best_f1 = df_results.loc[df_results['f1_score'].idxmax()]
        best_roc_auc = df_results.loc[df_results['roc_auc'].idxmax()]
        best_trump_precision = df_results.loc[df_results['trump_precision'].idxmax()]
        best_trump_recall = df_results.loc[df_results['trump_recall'].idxmax()]
        
        print(f"Best Accuracy:       {best_accuracy['config']} ({best_accuracy['accuracy']:.4f})")
        print(f"Best F1-Score:       {best_f1['config']} ({best_f1['f1_score']:.4f})")
        print(f"Best ROC AUC:        {best_roc_auc['config']} ({best_roc_auc['roc_auc']:.4f})")
        print(f"Best Trump Precision: {best_trump_precision['config']} ({best_trump_precision['trump_precision']:.4f})")
        print(f"Best Trump Recall:   {best_trump_recall['config']} ({best_trump_recall['trump_recall']:.4f})")
        
        # Feature impact analysis
        print(f"\n{'='*60}")
        print("FEATURE IMPACT ANALYSIS:")
        print(f"{'='*60}")
        
        for classifier in ['xgboost', 'logistic']:
            base_result = df_results[(df_results['classifier_type'] == classifier) & 
                                   (~df_results['include_text_features'])]
            enhanced_result = df_results[(df_results['classifier_type'] == classifier) & 
                                       (df_results['include_text_features'])]
            
            if len(base_result) > 0 and len(enhanced_result) > 0:
                base_f1 = base_result['f1_score'].iloc[0]
                enhanced_f1 = enhanced_result['f1_score'].iloc[0]
                improvement = enhanced_f1 - base_f1
                
                print(f"{classifier.upper()}: F1 improvement with enhanced features: {improvement:+.4f}")
    
    def save_results(self, output_path: str) -> None:
        """
        Save comparison results to JSON file.
        
        Args:
            output_path: Path to save results JSON
        """
        if not self.results:
            print("No results to save!")
            return
        
        results_data = {
            'metadata': {
                'data_path': self.data_path,
                'test_size': self.test_size,
                'random_state': self.random_state,
                'total_experiments': len(self.results),
                'comparison_timestamp': datetime.now().isoformat()
            },
            'experiments': self.results
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description='Compare Trump Tweet Classifier Configurations')
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
        '--output', 
        type=str, 
        default=None,
        help='Path to save comparison results JSON'
    )
    parser.add_argument(
        '--single', 
        nargs=2,
        metavar=('CLASSIFIER', 'FEATURES'),
        help='Run single experiment: CLASSIFIER={xgboost,logistic} FEATURES={true,false}'
    )
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ModelComparator(
        data_path=args.data,
        test_size=args.test_size
    )
    
    if args.single:
        # Run single experiment
        classifier_type = args.single[0]
        include_features = args.single[1].lower() == 'true'
        
        if classifier_type not in ['xgboost', 'logistic']:
            raise ValueError("Classifier must be 'xgboost' or 'logistic'")
        
        result = comparator.run_single_experiment(classifier_type, include_features)
        comparator.results = [result]
    else:
        # Run full comparison
        print("Running comprehensive model comparison...")
        comparator.run_comparison()
    
    # Print summary
    comparator.print_comparison_summary()
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"model_comparison_{timestamp}.json"
    
    comparator.save_results(output_path)


if __name__ == "__main__":
    main()
