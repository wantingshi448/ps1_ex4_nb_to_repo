# -*- coding: utf-8 -*-
"""
Data Loading Module

Functions:
- Load training data
- Load test data
- Combine train and test data (for EDA)
"""

import pandas as pd
from pathlib import Path
from src.config import TRAIN_PATH, TEST_PATH


def load_train_data():
    """Load training dataset"""
    print(f"Loading training data: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Training data loaded! Shape: {train_df.shape}")
    return train_df


def load_test_data():
    """Load test dataset"""
    print(f"Loading test data: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    print(f"Test data loaded! Shape: {test_df.shape}")
    return test_df


def load_all_data():
    """Load both training and test data"""
    train_df = load_train_data()
    test_df = load_test_data()
    return train_df, test_df


def combine_train_test(train_df, test_df):
    """
    Combine training and test datasets with a 'set' column marker
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        
    Returns:
        Combined dataframe with 'set' column ('train' or 'test')
    """
    print("Combining train and test data...")
    
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    
    print(f"Data combined! Total shape: {all_df.shape}")
    print(f"  - Training: {(all_df['set'] == 'train').sum()} rows")
    print(f"  - Test: {(all_df['set'] == 'test').sum()} rows")
    
    return all_df


if __name__ == "__main__":
    print("="*50)
    print("Testing Data Loader Module")
    print("="*50)
    
    train_df, test_df = load_all_data()
    all_df = combine_train_test(train_df, test_df)
    
    print("\nFirst 3 rows of training data:")
    print(train_df.head(3))
