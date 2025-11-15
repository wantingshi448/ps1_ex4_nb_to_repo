# -*- coding: utf-8 -*-
"""
Model Training and Evaluation Module

Functions:
- Data splitting (train/validation)
- Model training
- Model prediction
- Model evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from src.config import (
    VALID_SIZE, RANDOM_STATE,
    RF_N_ESTIMATORS, RF_CRITERION, RF_N_JOBS, RF_RANDOM_STATE, RF_VERBOSE,
    BASELINE_FEATURES, TARGET_FEATURE
)


def split_data(df, features, target=TARGET_FEATURE, test_size=VALID_SIZE):
    """
    Split data into training and validation sets

    Args:
        df: Complete dataset
        features: List of feature column names
        target: Target column name
        test_size: Validation set proportion

    Returns:
        tuple: (train_X, valid_X, train_Y, valid_Y)
    """
    print("\n" + "="*60)
    print("Data Splitting")
    print("="*60)

    train, valid = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_X = train[features]
    train_Y = train[target].values
    valid_X = valid[features]
    valid_Y = valid[target].values

    print(f"Data split complete")
    print(f"  Training set: {len(train_X)} samples")
    print(f"  Validation set: {len(valid_X)} samples")
    print(f"  Features: {len(features)}")
    print(f"  Feature list: {features}")

    return train_X, valid_X, train_Y, valid_Y


def train_random_forest(train_X, train_Y,
                       n_estimators=RF_N_ESTIMATORS,
                       criterion=RF_CRITERION,
                       random_state=RF_RANDOM_STATE,
                       n_jobs=RF_N_JOBS,
                       verbose=RF_VERBOSE):
    """
    Train Random Forest classifier

    Args:
        train_X: Training features
        train_Y: Training labels
        n_estimators: Number of trees
        criterion: Split criterion ('gini' or 'entropy')
        random_state: Random seed
        n_jobs: Number of parallel jobs (-1 = use all CPUs)
        verbose: Whether to show training progress

    Returns:
        Trained RandomForestClassifier
    """
    print("\n" + "="*60)
    print("Model Training")
    print("="*60)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )

    print(f"Model parameters:")
    print(f"  - Number of trees: {n_estimators}")
    print(f"  - Split criterion: {criterion}")
    print(f"  - Random seed: {random_state}")

    print(f"\nTraining...")
    clf.fit(train_X, train_Y)
    print(f"Training complete!")

    return clf


def predict(clf, X):
    """Make predictions using trained classifier"""
    print(f"\nPredicting {len(X)} samples...")
    predictions = clf.predict(X)
    print(f"Prediction complete")
    return predictions


def evaluate_model(clf, X, Y, dataset_name="Dataset"):
    """
    Evaluate model performance

    Args:
        clf: Trained classifier
        X: Feature data
        Y: True labels
        dataset_name: Dataset name for display

    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*60)
    print(f"{dataset_name} - Model Evaluation")
    print("="*60)

    predictions = predict(clf, X)

    accuracy = metrics.accuracy_score(Y, predictions)

    print(f"\nClassification Report:")
    print(metrics.classification_report(
        Y, predictions,
        target_names=['Not Survived', 'Survived']
    ))

    print(f"Confusion Matrix:")
    cm = metrics.confusion_matrix(Y, predictions)
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Interpretation:")
    print(f"  True Negative (TN - Correct not survived): {tn}")
    print(f"  False Positive (FP - Wrong survived): {fp}")
    print(f"  False Negative (FN - Wrong not survived): {fn}")
    print(f"  True Positive (TP - Correct survived): {tp}")

    precision = metrics.precision_score(Y, predictions)
    recall = metrics.recall_score(Y, predictions)
    f1 = metrics.f1_score(Y, predictions)

    print(f"\nKey Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def get_feature_importance(clf, feature_names):
    """
    Get feature importance from Random Forest

    Args:
        clf: Trained Random Forest classifier
        feature_names: List of feature names

    Returns:
        DataFrame: Feature importance table (sorted by importance)
    """
    print("\n" + "="*60)
    print("Feature Importance Analysis")
    print("="*60)

    importances = clf.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance Ranking:")
    for idx, row in feature_importance_df.iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")

    return feature_importance_df


def train_and_evaluate(df, features, target=TARGET_FEATURE):
    """
    Complete training and evaluation pipeline

    This convenience function executes the full ML workflow:
    1. Data splitting
    2. Model training
    3. Model evaluation (training and validation sets)
    4. Feature importance analysis

    Args:
        df: Dataset
        features: List of feature column names
        target: Target column name

    Returns:
        tuple: (clf, train_metrics, valid_metrics, feature_importance)
    """
    print("\n" + "#"*60)
    print("# Starting Complete Training and Evaluation Pipeline")
    print("#"*60)

    # 1. Data splitting
    train_X, valid_X, train_Y, valid_Y = split_data(df, features, target)

    # 2. Model training
    clf = train_random_forest(train_X, train_Y)

    # 3. Evaluate on training set (check overfitting)
    train_metrics = evaluate_model(clf, train_X, train_Y, "Training Set")

    # 4. Evaluate on validation set
    valid_metrics = evaluate_model(clf, valid_X, valid_Y, "Validation Set")

    # 5. Feature importance
    feature_importance = get_feature_importance(clf, features)

    # 6. Overfitting check
    print("\n" + "="*60)
    print("Overfitting Check")
    print("="*60)
    train_acc = train_metrics['accuracy']
    valid_acc = valid_metrics['accuracy']
    diff = train_acc - valid_acc

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {valid_acc:.4f}")
    print(f"Difference: {diff:.4f}")

    if diff > 0.1:
        print("WARNING: Possible overfitting (difference > 0.1)")
    else:
        print("Model generalization looks good")

    print("\n" + "#"*60)
    print("# Training and Evaluation Complete!")
    print("#"*60 + "\n")

    return clf, train_metrics, valid_metrics, feature_importance


if __name__ == "__main__":
    from src.data_loader import load_train_data
    from src.feature_engineering import apply_all_features

    print("Testing Model Training Module")
    print("=" * 60)

    train_df = load_train_data()
    train_df = apply_all_features(train_df)

    clf, train_m, valid_m, importance = train_and_evaluate(
        train_df,
        BASELINE_FEATURES
    )
