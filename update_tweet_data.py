"""
Script to update TweetData.csv by adding the is_trump column.

This script adds a boolean column indicating whether each tweet
was written by @realDonaldTrump based on the user column.
"""

import pandas as pd
import os


def update_tweet_data(input_file: str = "TweetData.csv", output_file: str = None) -> None:
    """
    Update the tweet data CSV to include is_trump column.
    
    Args:
        input_file: Path to the original CSV file
        output_file: Path for the updated CSV (defaults to overwriting original)
    """
    if output_file is None:
        # Create backup first
        backup_file = input_file.replace('.csv', '_backup.csv')
        print(f"Creating backup at {backup_file}...")
        
    print(f"Loading data from {input_file}...")
    
    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} tweets")
    
    # Check current columns
    print(f"Current columns: {list(df.columns)}")
    
    # Add is_trump column
    print("Adding is_trump column...")
    df['is_trump'] = df['user'] == '@realDonaldTrump'
    
    # Count Trump vs non-Trump tweets
    trump_count = df['is_trump'].sum()
    non_trump_count = len(df) - trump_count
    
    print(f"Trump tweets: {trump_count}")
    print(f"Non-Trump tweets: {non_trump_count}")
    print(f"Percentage Trump: {trump_count/len(df)*100:.2f}%")
    
    # Create backup if overwriting original
    if output_file is None:
        backup_file = input_file.replace('.csv', '_backup.csv')
        if not os.path.exists(backup_file):
            print(f"Creating backup: {backup_file}")
            df_original = pd.read_csv(input_file)
            df_original.to_csv(backup_file, index=False)
        output_file = input_file
    
    # Save updated data
    print(f"Saving updated data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("Update completed successfully!")
    print(f"New columns: {list(df.columns)}")


if __name__ == "__main__":
    update_tweet_data()

