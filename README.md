# Titanic Survival Prediction - Modular Pipeline

This project implements a modular machine learning pipeline for predicting survival on the Titanic using the famous Kaggle dataset.

**Note:** This repository contains work from both **PS1 Exercise 4** (modular pipeline development) and **PS2 Exercise 5** (focused visualization analysis).

## Project Structure
```
ps1_ex4_nb_to_repo/
├── data/
│   └── raw/              # Original Titanic dataset
├── figures/              # Visualization outputs (PS2 Ex5)
│   ├── 1_feature_importance.png
│   ├── 2_gender_survival.png
│   ├── 3_pclass_survival.png
│   ├── 4_gender_class_interaction.png
│   └── 5_age_analysis.png
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration and constants
│   ├── data_loader.py    # Data loading utilities
│   ├── feature_engineering.py  # Feature engineering
│   ├── model.py          # Model training and evaluation
│   └── utils.py          # Utility functions
├── create_visualizations.py  # Visualization generation script (PS2 Ex5)
├── main.py               # Main execution script
├── report.ipynb          # Focused visualization analysis report (PS2 Ex5)
├── requirements.txt      # Project dependencies
└── README.md
```

## Features

### PS1 Exercise 4: Modular Pipeline
- Modular code organization
- Random Forest classifier
- Feature importance analysis
- Comprehensive data preprocessing
- Model evaluation metrics

### PS2 Exercise 5: Focused Visualization Analysis
- 5 professional data visualizations
- Feature selection justification through visual analysis
- Comprehensive analysis report (≤500 words)
- Demonstrates why Sex and Pclass are the most critical features

## Installation
```bash
# Clone the repository
git clone https://github.com/wantingshi448/ps1_ex4_nb_to_repo.git

# Navigate to project directory
cd ps1_ex4_nb_to_repo

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the ML Pipeline (PS1 Ex4)
```bash
python main.py
```

### Generate Visualizations (PS2 Ex5)
```bash
python create_visualizations.py
```

This will create 5 visualization charts in the `figures/` directory.

### View the Analysis Report (PS2 Ex5)

Open `report.ipynb` in Jupyter Notebook to see the complete focused visualization analysis.
```bash
jupyter notebook report.ipynb
```

## Model Performance

- **Validation Accuracy:** 76.54%
- **Key Features:**
  - Sex (Gender): 73.44% importance
  - Pclass (Passenger Class): 26.56% importance

## Key Findings (PS2 Ex5)

The visualization analysis demonstrates:

1. **Gender is the dominant predictor** (73.44% importance)
   - Female survival rate: 74.2%
   - Male survival rate: 18.9%

2. **Passenger class shows clear socioeconomic gradient** (26.56% importance)
   - 1st class: 63.0% survival
   - 3rd class: 24.2% survival

3. **Interaction effects are significant**
   - 1st class females: 96.8% survival (highest)
   - 3rd class males: 13.5% survival (lowest)

4. **Age has secondary importance**
   - Children (0-12): 58.0% survival
   - Seniors (60+): 22.7% survival

## Dataset

The Titanic dataset from Kaggle includes:
- Training set: 891 passengers
- Features: Age, Sex, Pclass, Fare, Embarked, SibSp, Parch, etc.
- Target: Survived (0 = No, 1 = Yes)

## Visualizations

The `figures/` directory contains 5 key visualizations that support feature selection decisions:

1. **Feature Importance** - Quantitative ranking from Random Forest model
2. **Gender vs Survival** - Demonstrates the critical impact of gender
3. **Passenger Class vs Survival** - Shows socioeconomic stratification
4. **Gender × Class Interaction** - Reveals multiplicative effects
5. **Age Distribution Analysis** - Supporting evidence for age as tertiary feature

## Project Timeline

- **PS1 Exercise 4:** Initial modular pipeline development
- **PS2 Exercise 5:** Focused visualization analysis and feature selection justification

## Author

**Yvette (Wanting SHI)**  
Cambridge University  
Course: Fundamentals in Data Science

## Acknowledgments

- Kaggle for the Titanic dataset
- Cambridge University FDS course instructors