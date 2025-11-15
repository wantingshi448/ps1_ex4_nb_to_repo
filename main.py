#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titanic Survival Prediction - Main Program

This is the main entry point of the project.
Run command: python main.py

Features:
1. Load Titanic dataset
2. Exploratory data analysis (optional)
3. Feature engineering
4. Train Random Forest model
5. Model evaluation
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all_data, combine_train_test
from src.data_inspector import full_data_inspection
from src.feature_engineering import apply_all_features
from src.model import train_and_evaluate
from src.config import BASELINE_FEATURES


def main():
    """
    Main function - Execute complete ML pipeline
    """
    print("\n")
    print("="*70)
    print(" "*20 + "TITANIC SURVIVAL PREDICTION")
    print("="*70)
    print()

    # ========================================
    # Step 1: Load Data
    # ========================================
    print("\n[Step 1/5] Loading Data")
    print("-"*70)

    train_df, test_df = load_all_data()

    # ========================================
    # Step 2: Data Exploration (Optional)
    # ========================================
    # Uncomment below to see detailed data analysis
    # print("\n[Step 2/5] Data Exploration")
    # print("-"*70)
    # full_data_inspection(train_df, "Training Set")
    # full_data_inspection(test_df, "Test Set")

    print("\n[Step 2/5] Data Exploration")
    print("-"*70)
    print("Skipping detailed exploration (edit main.py line 57-59 to enable)")

    # ========================================
    # Step 3: Feature Engineering
    # ========================================
    print("\n[Step 3/5] Feature Engineering")
    print("-"*70)

    # Apply feature engineering to training data
    train_df_enhanced = apply_all_features(
        train_df,
        include_name_features=True,
        encode_categorical=True
    )

    # Apply same feature engineering to test data
    test_df_enhanced = apply_all_features(
        test_df,
        include_name_features=True,
        encode_categorical=True
    )

    print(f"\nTraining set shape: {train_df_enhanced.shape}")
    print(f"Test set shape: {test_df_enhanced.shape}")

    # ========================================
    # Step 4: Model Training
    # ========================================
    print("\n[Step 4/5] Model Training and Evaluation")
    print("-"*70)

    # Train model using baseline features
    print(f"\nUsing features: {BASELINE_FEATURES}")

    clf, train_metrics, valid_metrics, feature_importance = train_and_evaluate(
        train_df_enhanced,
        BASELINE_FEATURES
    )

    # ========================================
    # Step 5: Summary
    # ========================================
    print("\n[Step 5/5] Results Summary")
    print("-"*70)

    print("\nFinal Model Performance:")
    print(f"  Training Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Validation Accuracy: {valid_metrics['accuracy']:.4f}")
    print(f"  Validation F1 Score: {valid_metrics['f1']:.4f}")

    print("\nFeature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")

    # ========================================
    # Complete
    # ========================================
    print("\n")
    print("="*70)
    print(" "*25 + "PROJECT COMPLETE!")
    print("="*70)
    print()

    # Next steps suggestions
    print("Next Steps:")
    print("  1. Try more features (Age Interval, Fare Interval, Family Size, etc.)")
    print("  2. Adjust model parameters (edit src/config.py)")
    print("  3. Try different algorithms")
    print("  4. Generate predictions for test set")
    print()


if __name__ == "__main__":
    """
    Program entry point

    This code runs when you execute: python main.py
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
