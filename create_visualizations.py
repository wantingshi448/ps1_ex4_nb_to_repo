"""
PS2 Exercise 5 - Visualization Generator
Creates all visualizations for the Titanic survival analysis report
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create figures directory
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

print("="*60)
print("PS2 Exercise 5 - Visualization Generator")
print("="*60)

# Load data
print("\n[1/6] Loading Titanic dataset...")
try:
    train_df = pd.read_csv('data/raw/train.csv')
    print(f"✓ Dataset loaded successfully")
    print(f"  - Total passengers: {len(train_df)}")
    print(f"  - Overall survival rate: {train_df['Survived'].mean():.1%}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# ============================================
# Visualization 1: Feature Importance
# ============================================
print("\n[2/6] Creating Feature Importance visualization...")
try:
    # Based on your Random Forest model results
    feature_importance = {
        'Sex (Gender)': 73.44,
        'Pclass (Ticket Class)': 26.56
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    bars = ax.barh(features, importance, color=['#e74c3c', '#3498db'])
    ax.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest Feature Importance\nTitanic Survival Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    
    # Add percentage labels
    for i, (feature, imp) in enumerate(zip(features, importance)):
        ax.text(imp + 2, i, f'{imp:.2f}%', 
                va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / '1_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 1_feature_importance.png")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================
# Visualization 2: Gender vs Survival
# ============================================
print("\n[3/6] Creating Gender vs Survival visualization...")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate survival rates
    gender_survival = train_df.groupby('Sex')['Survived'].agg(['sum', 'count', 'mean'])
    gender_survival['died'] = gender_survival['count'] - gender_survival['sum']
    
    # Create grouped bar chart
    x = np.arange(len(gender_survival))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gender_survival['sum'], width, 
                   label='Survived', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, gender_survival['died'], width, 
                   label='Died', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Gender', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Passengers', fontsize=12, fontweight='bold')
    ax.set_title('Survival by Gender: The Most Critical Feature', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Female', 'Male'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add counts and percentages
    for i, (idx, row) in enumerate(gender_survival.iterrows()):
        height1 = row['sum']
        ax.text(i - width/2, height1 + 5, 
                f"{int(height1)}\n({row['mean']:.1%})", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        height2 = row['died']
        ax.text(i + width/2, height2 + 5, 
                f"{int(height2)}\n({1-row['mean']:.1%})", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / '2_gender_survival.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 2_gender_survival.png")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================
# Visualization 3: Passenger Class vs Survival
# ============================================
print("\n[4/6] Creating Passenger Class vs Survival visualization...")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate survival rates by class
    class_survival = train_df.groupby('Pclass')['Survived'].agg(['sum', 'count', 'mean'])
    class_survival['died'] = class_survival['count'] - class_survival['sum']
    
    # Create grouped bar chart
    x = np.arange(len(class_survival))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, class_survival['sum'], width, 
                   label='Survived', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, class_survival['died'], width, 
                   label='Died', color='#e67e22', alpha=0.8)
    
    ax.set_xlabel('Passenger Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Passengers', fontsize=12, fontweight='bold')
    ax.set_title('Survival by Passenger Class: Socioeconomic Impact', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add counts and percentages
    for i, (idx, row) in enumerate(class_survival.iterrows()):
        height1 = row['sum']
        ax.text(i - width/2, height1 + 5, 
                f"{int(height1)}\n({row['mean']:.1%})", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        height2 = row['died']
        ax.text(i + width/2, height2 + 5, 
                f"{int(height2)}\n({1-row['mean']:.1%})", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / '3_pclass_survival.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 3_pclass_survival.png")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================
# Visualization 4: Combined Gender + Class
# ============================================
print("\n[5/6] Creating Combined Gender + Class visualization...")
try:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Survival rate by gender and class
    gender_class = train_df.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack()
    
    # Create grouped bar chart
    x = np.arange(len(gender_class))
    width = 0.25
    
    bars1 = ax.bar(x - width, gender_class[1], width, 
                   label='1st Class', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, gender_class[2], width, 
                   label='2nd Class', color='#e67e22', alpha=0.8)
    bars3 = ax.bar(x + width, gender_class[3], width, 
                   label='3rd Class', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Gender', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Rate', fontsize=12, fontweight='bold')
    ax.set_title('Interaction Effect: Gender × Passenger Class', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Female', 'Male'])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i in range(len(gender_class)):
        for j, col in enumerate([1, 2, 3]):
            height = gender_class.iloc[i, j-1]
            ax.text(i + (j-1)*width, height + 0.03, 
                    f'{height:.1%}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / '4_gender_class_interaction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 4_gender_class_interaction.png")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================
# Visualization 5: Age Analysis
# ============================================
print("\n[6/6] Creating Age Distribution visualization...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Age distribution by survival
    survived_ages = train_df[train_df['Survived'] == 1]['Age'].dropna()
    died_ages = train_df[train_df['Survived'] == 0]['Age'].dropna()
    
    axes[0].hist([survived_ages, died_ages], bins=30, 
                 label=['Survived', 'Died'],
                 color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Passengers', fontsize=12, fontweight='bold')
    axes[0].set_title('Age Distribution by Survival Status', 
                      fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Right: Survival rate by age group
    train_df['AgeGroup'] = pd.cut(train_df['Age'], 
                                   bins=[0, 12, 18, 35, 60, 100], 
                                   labels=['Child\n(0-12)', 'Teen\n(13-18)', 
                                          'Adult\n(19-35)', 'Middle\n(36-60)', 
                                          'Senior\n(60+)'])
    age_survival = train_df.groupby('AgeGroup', observed=True)['Survived'].mean()
    
    bars = axes[1].bar(range(len(age_survival)), age_survival.values, 
                       color='#9b59b6', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Age Group', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Survival Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('Survival Rate by Age Group', fontsize=13, fontweight='bold')
    axes[1].set_xticks(range(len(age_survival)))
    axes[1].set_xticklabels(age_survival.index)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(age_survival.values):
        axes[1].text(i, v + 0.03, f'{v:.1%}', 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / '5_age_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 5_age_analysis.png")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================
# Summary and Key Statistics
# ============================================
print("\n" + "="*60)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*60)

# Count generated figures
fig_files = sorted(figures_dir.glob('*.png'))
print(f"\n✓ Generated {len(fig_files)} visualizations in 'figures/' directory:\n")
for fig_file in fig_files:
    print(f"   {fig_file.name}")

# Display key statistics
print("\n" + "="*60)
print("KEY FINDINGS FOR YOUR REPORT:")
print("="*60)

female_survival = train_df[train_df['Sex'] == 'female']['Survived'].mean()
male_survival = train_df[train_df['Sex'] == 'male']['Survived'].mean()
class1_survival = train_df[train_df['Pclass'] == 1]['Survived'].mean()
class3_survival = train_df[train_df['Pclass'] == 3]['Survived'].mean()

print(f"\n1. GENDER (Most Important - 73.44% importance):")
print(f"   • Female survival rate: {female_survival:.1%}")
print(f"   • Male survival rate: {male_survival:.1%}")
print(f"   • Difference: {female_survival - male_survival:.1%}")

print(f"\n2. PASSENGER CLASS (Second Most Important - 26.56% importance):")
print(f"   • 1st class survival: {class1_survival:.1%}")
print(f"   • 3rd class survival: {class3_survival:.1%}")
print(f"   • Difference: {class1_survival - class3_survival:.1%}")

print(f"\n3. INTERACTION EFFECT:")
female_1st = train_df[(train_df['Sex'] == 'female') & (train_df['Pclass'] == 1)]['Survived'].mean()
male_3rd = train_df[(train_df['Sex'] == 'male') & (train_df['Pclass'] == 3)]['Survived'].mean()
print(f"   • 1st class females: {female_1st:.1%}")
print(f"   • 3rd class males: {male_3rd:.1%}")
print(f"   • Shows combined effect of both features")

print(f"\n4. AGE:")
child_survival = train_df[train_df['Age'] <= 12]['Survived'].mean()
print(f"   • Children (0-12): {child_survival:.1%}")
print(f"   • Adults (19-35): {train_df[(train_df['Age'] > 18) & (train_df['Age'] <= 35)]['Survived'].mean():.1%}")

print("\n" + "="*60)
print("Next step: Open report.ipynb in Jupyter and write your report!")
print("="*60)