# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:11:53 2025

@author: Mizbah Ahmed Sresto
"""

# ---------------------------
# Titanic Dataset Analysis
# ---------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set working directory and create plots folder
os.makedirs(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\plots', exist_ok=True)

# ======================
# 1. Load and Preprocess
# ======================
# Load dataset
titanic = pd.read_csv(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\train.csv')

# Clean column names
titanic.columns = titanic.columns.str.lower()

# Drop unnecessary columns
titanic.drop(['passengerid', 'name', 'ticket', 'cabin'], axis=1, inplace=True)

# Handle missing values
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])

# Convert categorical features
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Create age groups
bins = [0, 12, 19, 59, 100]
labels = ['Child', 'Teenager', 'Adult', 'Senior']
titanic['age_group'] = pd.cut(titanic['age'], bins=bins, labels=labels)

# ======================
# 2. Univariate Analysis
# ======================
# Age group distribution
plt.figure(figsize=(10,6))
sns.countplot(data=titanic, x='age_group', palette='husl')
plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.savefig(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\plots\age_group_distribution.png')
plt.close()

# Survival rates by age group
plt.figure(figsize=(10,6))
sns.barplot(data=titanic, x='age_group', y='survived', errorbar=None)
plt.title('Survival Rates by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.savefig(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\plots\survival_by_age_group.png')
plt.close()

# Passengers per class by age group
plt.figure(figsize=(10,6))
sns.countplot(data=titanic, x='pclass', hue='age_group', palette='husl')
plt.title('Passengers per Class by Age Group')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.savefig(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\plots\class_age_distribution.png')
plt.close()

# ======================
# 3. Bivariate Analysis
# ======================
# Fare distribution by class
plt.figure(figsize=(10,6))
sns.boxplot(data=titanic, x='pclass', y='fare')
plt.title('Fare Distribution by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.savefig(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\plots\fare_by_class.png')
plt.close()

# Age vs Fare scatter plot
plt.figure(figsize=(10,6))
sns.scatterplot(data=titanic, x='age', y='fare', hue='survived', palette='viridis')
plt.title('Age vs Fare by Survival Status')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.savefig(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\plots\age_fare_scatter.png')
plt.close()

# Age distribution by survival
plt.figure(figsize=(10,6))
sns.violinplot(data=titanic, x='survived', y='age')
plt.title('Age Distribution by Survival Status')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.savefig(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\plots\age_survival_violin.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(10,6))
corr_df = titanic[['age', 'fare', 'sex', 'pclass', 'sibsp', 'survived']].corr()
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\plots\correlation_heatmap.png')
plt.close()

# ======================
# 4. Multi-variable Analysis
# ======================
# Facet Grid for age distribution by class
g = sns.FacetGrid(titanic, col="pclass")
g.map(sns.histplot, "age", kde=False)
g.set_titles("Class {col_name}")
g.set_axis_labels("Age", "Count")
g.fig.suptitle('Age Distribution by Passenger Class', y=1.02)
plt.savefig(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\plots\age_class_facetgrid.png')
plt.close()

# Pairplot for numerical relationships
pairplot = sns.pairplot(titanic[['age', 'fare', 'survived']], 
                       hue='survived', 
                       palette='coolwarm',
                       plot_kws={'alpha':0.6})
pairplot.fig.suptitle('Pairwise Relationships', y=1.02)
plt.savefig(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\plots\numerical_pairplot.png')
plt.close()

# ======================
# 5. Generate Answers File
# ======================
answers = """
1. Age Group Survival Analysis:
   - Based on the bar plot, children show the highest survival rates
   - Survival rates decrease progressively through teenager -> adult -> senior groups

2. Fare Distribution Trends:
   - 1st class shows highest median fares with several outliers
   - 3rd class has the lowest fares but widest distribution
   - Fare amounts decrease significantly with lower class

3. Heatmap Insights:
   - Strongest positive correlation: Sex (female survival advantage)
   - Moderate correlation: Fare (higher fares linked to survival)
   - Negative correlation: Pclass (lower classes = lower survival)
   - Age shows weak negative correlation with survival

4. Age-Fare-Class Relationship:
   - 1st class passengers maintain high fares across all ages
   - 3rd class shows compressed fare distribution regardless of age
   - Older passengers in higher classes paid more than younger ones
"""

with open(r'D:\LSU_Spring 2025\BE 7901\Assignment 2\titanic\analysis_answers.txt', 'w') as f:
    f.write(answers)

print("Analysis complete! Check the output directory for results.")