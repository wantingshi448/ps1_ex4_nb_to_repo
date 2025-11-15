# -*- coding: utf-8 -*-
"""
Data Inspection Module

Functions:
- Missing data analysis
- Most frequent data analysis
- Unique values analysis
- Basic data statistics
"""

import pandas as pd
import numpy as np


def analyze_missing_data(df, dataset_name="Dataset"):
    """Analyze missing values in dataset"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} - Missing Data Analysis")
    print(f"{'='*50}")

    total = df.isnull().sum()
    percent = (df.isnull().sum() / df.isnull().count() * 100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Types'] = types

    result = np.transpose(tt)

    cols_with_missing = total[total > 0].index.tolist()
    if cols_with_missing:
        print(f"\nFound {len(cols_with_missing)} columns with missing values:")
        for col in cols_with_missing:
            print(f"   - {col}: {total[col]} missing ({percent[col]:.2f}%)")
    else:
        print("\nNo missing values!")

    return result


def analyze_most_frequent(df, dataset_name="Dataset"):
    """Analyze most frequent values in each column"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} - Most Frequent Data Analysis")
    print(f"{'='*50}")

    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']

    items = []
    vals = []

    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]
            val = df[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception:
            items.append(0)
            vals.append(0)

    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)

    result = np.transpose(tt)

    print("\nFirst 5 columns most frequent data:")
    print(result.iloc[:, :5])

    return result


def analyze_unique_values(df, dataset_name="Dataset"):
    """Analyze unique value counts in each column"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} - Unique Values Analysis")
    print(f"{'='*50}")

    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']

    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)

    tt['Uniques'] = uniques
    result = np.transpose(tt)

    print("\nUnique value counts per column:")
    for col in df.columns:
        unique_count = df[col].nunique()
        total_count = df[col].count()
        print(f"   - {col}: {unique_count} unique / {total_count} total")

    return result


def quick_data_overview(df, dataset_name="Dataset"):
    """Quick overview of the dataset"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} - Quick Overview")
    print(f"{'='*60}")

    print(f"\nData shape: {df.shape}")
    print(f"   - {df.shape[0]} rows")
    print(f"   - {df.shape[1]} columns")

    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")

    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nData info:")
    df.info()

    print(f"\nNumeric columns statistics:")
    print(df.describe())


def full_data_inspection(df, dataset_name="Dataset"):
    """
    Complete data exploration analysis

    Calls all analysis functions for a comprehensive overview
    """
    print(f"\n{'#'*60}")
    print(f"# Starting Full Data Exploration: {dataset_name}")
    print(f"{'#'*60}")

    quick_data_overview(df, dataset_name)
    missing = analyze_missing_data(df, dataset_name)
    frequent = analyze_most_frequent(df, dataset_name)
    unique = analyze_unique_values(df, dataset_name)

    print(f"\n{'#'*60}")
    print(f"# {dataset_name} Data Exploration Complete!")
    print(f"{'#'*60}\n")

    return {
        'missing': missing,
        'frequent': frequent,
        'unique': unique
    }


if __name__ == "__main__":
    from src.data_loader import load_all_data

    print("Testing Data Inspection Module")
    print("=" * 60)

    train_df, test_df = load_all_data()
    train_analysis = full_data_inspection(train_df, "Training Set")
    test_analysis = full_data_inspection(test_df, "Test Set")
