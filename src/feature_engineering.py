# -*- coding: utf-8 -*-
"""
Feature Engineering Module

Functions:
- Create family size feature
- Create age/fare interval features
- Parse names to extract title
- Create composite features
- Encode categorical features
"""

import pandas as pd
import numpy as np
from src.config import (
    AGE_BINS, AGE_LABELS,
    FARE_BINS, FARE_LABELS,
    FAMILY_SIZE_SINGLE, FAMILY_SIZE_SMALL_MAX,
    TITLE_MAPPINGS, RARE_TITLES
)


def create_family_size(df):
    """Create family size feature: SibSp + Parch + 1"""
    df = df.copy()
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    print(f"Created 'Family Size' feature")
    print(f"  Range: {df['Family Size'].min()} - {df['Family Size'].max()}")
    return df


def create_age_interval(df):
    """
    Create age interval feature
    0: 0-16, 1: 17-32, 2: 33-48, 3: 49-64, 4: 65+
    """
    df = df.copy()
    df["Age Interval"] = 0.0

    df.loc[df['Age'] <= 16, 'Age Interval'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[df['Age'] > 64, 'Age Interval'] = 4

    print(f"Created 'Age Interval' feature")
    print(f"  Distribution:\n{df['Age Interval'].value_counts().sort_index()}")
    return df


def create_fare_interval(df):
    """
    Create fare interval feature
    0: 0-7.91, 1: 7.91-14.454, 2: 14.454-31, 3: 31+
    """
    df = df.copy()
    df['Fare Interval'] = 0.0

    df.loc[df['Fare'] <= 7.91, 'Fare Interval'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare Interval'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare Interval'] = 2
    df.loc[df['Fare'] > 31, 'Fare Interval'] = 3

    print(f"Created 'Fare Interval' feature")
    print(f"  Distribution:\n{df['Fare Interval'].value_counts().sort_index()}")
    return df


def parse_names(row):
    """
    Parse name string to extract:
    - Family Name
    - Title (Mr., Mrs., Miss., etc.)
    - Given Name
    - Maiden Name (if present)
    """
    try:
        text = row["Name"]

        # Extract family name (before comma)
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]

        # Extract title (before period)
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]

        # Check for maiden name (in parentheses)
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0].strip()
            maiden_name = split_text[1].rstrip(")").strip()
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text.strip()
            return pd.Series([family_name, title, given_name, None])

    except Exception:
        return pd.Series([None, None, None, None])


def extract_name_features(df):
    """Extract features from Name column"""
    df = df.copy()
    print("Parsing names...")

    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(
        lambda row: parse_names(row), axis=1
    )

    print(f"Extracted name features")
    print(f"  Found {df['Title'].nunique()} different titles:")
    print(f"  {df['Title'].value_counts().head()}")

    return df


def create_sex_pclass(df):
    """Create Sex + Pclass composite feature"""
    df = df.copy()
    df["Sex_Pclass"] = df.apply(
        lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]),
        axis=1
    )

    print(f"Created 'Sex_Pclass' feature")
    print(f"  Distribution:\n{df['Sex_Pclass'].value_counts()}")
    return df


def create_family_type(df):
    """
    Create family type classification
    Single: family size = 1
    Small: family size 2-4
    Large: family size >= 5
    """
    df = df.copy()

    df["Family Type"] = df["Family Size"]

    df.loc[df["Family Size"] == FAMILY_SIZE_SINGLE, "Family Type"] = "Single"
    df.loc[(df["Family Size"] > FAMILY_SIZE_SINGLE) &
           (df["Family Size"] < FAMILY_SIZE_SMALL_MAX + 1), "Family Type"] = "Small"
    df.loc[df["Family Size"] >= FAMILY_SIZE_SMALL_MAX + 1, "Family Type"] = "Large"

    print(f"Created 'Family Type' feature")
    print(f"  Distribution:\n{df['Family Type'].value_counts()}")
    return df


def unify_titles(df):
    """
    Unify and simplify titles
    - Merge similar titles (Mlle. -> Miss., Mme. -> Mrs.)
    - Group rare titles as 'Rare'
    """
    df = df.copy()

    df["Titles"] = df["Title"]
    df['Titles'] = df['Titles'].replace(TITLE_MAPPINGS)
    df['Titles'] = df['Titles'].replace(RARE_TITLES, 'Rare')

    print(f"Unified titles")
    print(f"  Distribution after unification:\n{df['Titles'].value_counts()}")
    return df


def encode_sex(df):
    """
    Encode Sex as numeric
    female -> 1, male -> 0
    """
    df = df.copy()
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)

    print(f"Encoded 'Sex' feature")
    print(f"  female -> 1, male -> 0")
    return df


def apply_all_features(df, include_name_features=True, encode_categorical=True):
    """
    Apply all feature engineering steps

    Args:
        df: Original dataset
        include_name_features: Whether to include name parsing features
        encode_categorical: Whether to encode categorical features

    Returns:
        DataFrame with all engineered features
    """
    print("\n" + "="*60)
    print("Starting Feature Engineering")
    print("="*60)

    df = df.copy()

    # Create basic features
    df = create_family_size(df)
    df = create_age_interval(df)
    df = create_fare_interval(df)

    # Create composite features
    df = create_sex_pclass(df)
    df = create_family_type(df)

    # Name features (optional)
    if include_name_features:
        df = extract_name_features(df)
        df = unify_titles(df)

    # Encode categorical features (optional)
    if encode_categorical:
        df = encode_sex(df)

    print("\n" + "="*60)
    print("Feature Engineering Complete!")
    print("="*60 + "\n")

    return df


if __name__ == "__main__":
    from src.data_loader import load_all_data

    print("Testing Feature Engineering Module")
    print("=" * 60)

    train_df, test_df = load_all_data()
    train_df_enhanced = apply_all_features(train_df)

    print("\nEnhanced dataset first 5 rows:")
    print(train_df_enhanced.head())

    print("\nNew columns created:")
    new_cols = set(train_df_enhanced.columns) - set(train_df.columns)
    for col in new_cols:
        print(f"  - {col}")
