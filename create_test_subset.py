"""
Script to create a test subset of the TweetData.csv file.
Samples 5k Trump tweets and 5k non-Trump tweets for classification testing.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import os

def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV data and process it.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pandas.DataFrame: Processed dataframe with is_trump column
    """
    print(f"Loading data from {csv_path}...")
    
    # Load the data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tweets")
    
    # Create is_trump column based on username
    df['is_trump'] = df['user'] == '@realDonaldTrump'
    
    # Count Trump vs non-Trump tweets
    trump_count = df['is_trump'].sum()
    non_trump_count = len(df) - trump_count
    
    print(f"Trump tweets: {trump_count}")
    print(f"Non-Trump tweets: {non_trump_count}")
    
    return df

def create_balanced_subset(df: pd.DataFrame, n_samples: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a balanced subset with n_samples of Trump and n_samples of non-Trump tweets.
    
    Args:
        df: Input dataframe
        n_samples: Number of samples per class
        
    Returns:
        Tuple of (test_subset, remaining_data)
    """
    print(f"\nCreating balanced subset with {n_samples} samples per class...")
    
    # Separate Trump and non-Trump tweets
    trump_tweets = df[df['is_trump'] == True].copy()
    non_trump_tweets = df[df['is_trump'] == False].copy()
    
    # Check if we have enough samples
    if len(trump_tweets) < n_samples:
        raise ValueError(f"Not enough Trump tweets. Available: {len(trump_tweets)}, Requested: {n_samples}")
    
    if len(non_trump_tweets) < n_samples:
        raise ValueError(f"Not enough non-Trump tweets. Available: {len(non_trump_tweets)}, Requested: {n_samples}")
    
    # Sample randomly
    trump_sample = trump_tweets.sample(n=n_samples, random_state=42)
    non_trump_sample = non_trump_tweets.sample(n=n_samples, random_state=42)
    
    # Combine samples
    test_subset = pd.concat([trump_sample, non_trump_sample], ignore_index=True)
    
    # Shuffle the combined dataset
    test_subset = test_subset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create remaining data (everything except the sampled tweets)
    sampled_indices = set(trump_sample.index) | set(non_trump_sample.index)
    remaining_data = df[~df.index.isin(sampled_indices)].copy()
    
    print(f"Test subset created with {len(test_subset)} tweets")
    print(f"Remaining data has {len(remaining_data)} tweets")
    
    return test_subset, remaining_data

def save_datasets(test_subset: pd.DataFrame, remaining_data: pd.DataFrame, output_dir: str = "."):
    """
    Save the datasets to CSV files.
    
    Args:
        test_subset: Test dataset
        remaining_data: Remaining training data
        output_dir: Output directory
    """
    print(f"\nSaving datasets to {output_dir}...")
    
    # Save test subset
    test_path = os.path.join(output_dir, "test_subset.csv")
    test_subset.to_csv(test_path, index=False)
    print(f"Test subset saved to {test_path}")
    
    # Save remaining data
    remaining_path = os.path.join(output_dir, "remaining_data.csv")
    remaining_data.to_csv(remaining_path, index=False)
    print(f"Remaining data saved to {remaining_path}")
    
    # Print statistics for test subset
    trump_count_test = test_subset['is_trump'].sum()
    non_trump_count_test = len(test_subset) - trump_count_test
    
    print(f"\nTest subset statistics:")
    print(f"  Trump tweets: {trump_count_test}")
    print(f"  Non-Trump tweets: {non_trump_count_test}")
    print(f"  Total: {len(test_subset)}")
    
    # Print statistics for remaining data
    trump_count_remaining = remaining_data['is_trump'].sum()
    non_trump_count_remaining = len(remaining_data) - trump_count_remaining
    
    print(f"\nRemaining data statistics:")
    print(f"  Trump tweets: {trump_count_remaining}")
    print(f"  Non-Trump tweets: {non_trump_count_remaining}")
    print(f"  Total: {len(remaining_data)}")

def main():
    """Main function to create the test subset."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load and process data
    df = load_and_process_data("TweetData.csv")
    
    # Create balanced subset
    test_subset, remaining_data = create_balanced_subset(df, n_samples=5000)
    
    # Save datasets
    save_datasets(test_subset, remaining_data)
    
    print("\nDataset creation completed successfully!")

if __name__ == "__main__":
    main()

