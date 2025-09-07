# Trump Tweet Classifier Experimentation Guide

This guide explains how to use the enhanced classifier with XGBoost and additional features.

## New Features Added

1. **XGBoost Classifier**: More powerful gradient boosting classifier as an alternative to logistic regression
2. **Enhanced Text Features**: 
   - Readability score (Flesch Reading Ease)
   - Capitalization percentage
3. **Comprehensive Metrics**: Per-class precision/recall for both Trump and Non-Trump classes
4. **Model Comparison Utility**: Systematic comparison of different configurations

## Installation

Install the new dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Single Model Training

Train with XGBoost and enhanced features (default):
```bash
python train_classifier.py --data test_subset.csv
```

Train with different configurations:
```bash
# XGBoost with just embeddings (no text features)
python train_classifier.py --data test_subset.csv --no-text-features

# Logistic regression with enhanced features
python train_classifier.py --data test_subset.csv --classifier logistic

# Logistic regression with just embeddings
python train_classifier.py --data test_subset.csv --classifier logistic --no-text-features
```

### 2. Model Comparison

Compare all configurations systematically:
```bash
python compare_models.py --data test_subset.csv
```

Run a single experiment:
```bash
# XGBoost with enhanced features
python compare_models.py --data test_subset.csv --single xgboost true

# Logistic regression with embeddings only
python compare_models.py --data test_subset.csv --single logistic false
```

Save comparison results:
```bash
python compare_models.py --data test_subset.csv --output my_comparison.json
```

## Understanding the Output

### Key Metrics Explained

- **Overall Metrics**:
  - `Accuracy`: Overall correct predictions
  - `ROC AUC`: Area under ROC curve (good for imbalanced data)
  - `Avg Precision`: Average precision score
  - `F1-Score`: Harmonic mean of precision and recall

- **Per-Class Metrics**:
  - `Non-Trump Precision`: Of predicted non-Trump, how many were correct
  - `Non-Trump Recall`: Of actual non-Trump, how many were caught
  - `Trump Precision`: Of predicted Trump, how many were correct
  - `Trump Recall`: Of actual Trump, how many were caught

### Feature Information

The enhanced vectorizer now creates vectors with:
- **Embedding dimensions**: 384 (from sentence-transformers model)
- **Text features**: 2 additional features when enabled
  - Readability score (0-100, higher = more readable)
  - Caps percentage (0-100, percentage of uppercase characters)

## Command Line Options

### train_classifier.py
- `--data`: Path to CSV file (default: test_subset.csv)
- `--classifier`: xgboost or logistic (default: xgboost)
- `--no-text-features`: Disable readability/caps features
- `--test-size`: Fraction for test set (default: 0.2)
- `--model-dir`: Directory to save model (default: models)
- `--save-plots`: Save confusion matrix plot

### compare_models.py
- `--data`: Path to CSV file (default: test_subset.csv)
- `--test-size`: Fraction for test set (default: 0.2)
- `--output`: Path to save JSON results
- `--single CLASSIFIER FEATURES`: Run single experiment

## Expected Results

The comparison should help you determine:

1. **Best overall classifier**: XGBoost vs Logistic Regression
2. **Feature impact**: Whether readability/caps features improve performance
3. **Class-specific performance**: Which model is better for Trump vs Non-Trump detection

## Next Steps for Iteration

1. **Feature Engineering**: Add more text features based on results
2. **Hyperparameter Tuning**: Optimize XGBoost parameters
3. **Cross-Validation**: Use k-fold CV for more robust evaluation
4. **Error Analysis**: Examine misclassified examples to identify patterns

## Troubleshooting

- **Memory Issues**: Reduce dataset size or use smaller batch processing
- **Slow Training**: Try `--no-text-features` to speed up experiments
- **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
