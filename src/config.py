# -*- coding: utf-8 -*-
"""
Configuration File - All project parameters

This file contains:
- Data file paths
- Model parameters
- Feature engineering parameters
- Training parameters
"""

from pathlib import Path

# ==========================================
# Project Paths
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data files
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
GENDER_SUBMISSION_PATH = RAW_DATA_DIR / "gender_submission.csv"

# ==========================================
# Data Processing Parameters
# ==========================================
VALID_SIZE = 0.2
RANDOM_STATE = 42

# ==========================================
# Feature Engineering Parameters
# ==========================================
AGE_BINS = [0, 16, 32, 48, 64, 100]
AGE_LABELS = [0, 1, 2, 3, 4]

FARE_BINS = [0, 7.91, 14.454, 31, 1000]
FARE_LABELS = [0, 1, 2, 3]

FAMILY_SIZE_SINGLE = 1
FAMILY_SIZE_SMALL_MAX = 4

TITLE_MAPPINGS = {
    'Mlle.': 'Miss.',
    'Ms.': 'Miss.',
    'Mme.': 'Mrs.',
}

RARE_TITLES = [
    'Lady.', 'the Countess.', 'Capt.', 'Col.',
    'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.',
    'Jonkheer.', 'Dona.'
]

# ==========================================
# Model Parameters
# ==========================================
RF_N_ESTIMATORS = 100
RF_CRITERION = "gini"
RF_N_JOBS = -1
RF_RANDOM_STATE = 42
RF_VERBOSE = False

BASELINE_FEATURES = ["Sex", "Pclass"]
TARGET_FEATURE = "Survived"

# ==========================================
# Visualization Parameters
# ==========================================
COLOR_LIST = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
FIGURE_SIZE = (8, 4)
