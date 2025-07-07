# Patterns in Driving Accidents

This project explores patterns in vehicular collisions across California (2006–2021) using machine learning and data visualization. It focuses on understanding the relationship between accident types, environmental/contextual factors, and the severity of the collisions.

## Overview

The dataset includes a wide range of features such as time of crash, location type, vehicle type at fault, surface and weather conditions, and more. Multiple ML models are applied to analyze and predict **collision severity**.

## Key Components

###  Data Preprocessing
- Loaded California traffic collision data from 2006 to 2021.
- Cleaned missing values and encoded categorical variables using `LabelEncoder`.
- Subsetted columns of interest for analysis and model training.

###  Exploratory Data Analysis
- Stacked bar plots to visualize collision severity across:
  - Types of collisions (e.g., rear-end, pedestrian, sideswipe)
  - Motorcycle involvement
- Correlation matrix of encoded features to explore multicollinearity.

###  Machine Learning Models

#### Linear Regression
- Modeled relationships between `pedestrian_action`, `bicycle_collision`, and `motorcycle_collision` to predict collision severity.

#### Random Forest Classifier
- Applied to predict collision severity.
- Evaluated using accuracy, confusion matrix, and feature importance plots.
- Grid search commented for hyperparameter tuning.

#### K-Nearest Neighbors (KNN)
- Used both on full and reduced datasets (for visualization).
- Applied square-root heuristic for neighbor count selection.

#### AutoGluon
- Trained `TabularPredictor` to automate model selection and evaluation.

###  Visualizations
- Swarm plots showing distribution of severity across collision types and violation categories.
- Heatmaps showing majority severity outcomes for combinations of type of collision and violation category.
- Feature importance bar plots to interpret model predictions.

## Metrics Reported
- Accuracy, precision, recall, F1-score
- Weighted accuracy using custom weights for each class (e.g., fatal, severe injury, etc.)
- Full confusion matrix and classification report for final models

## Technologies Used
- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- AutoGluon
- Imbalanced-learn (SMOTE for oversampling)

## How to Run
1. Ensure all dependencies are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn autogluon shap
