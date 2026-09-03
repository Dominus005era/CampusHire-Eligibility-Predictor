"""
CampusHire-Eligibility-Predictor
Problem: Student Placement Eligibility Prediction (Binary Classification)
Objective: Predict whether a student is likely to meet placement eligibility criteria.
Algorithm: Logistic Regression
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def main():
    print("=" * 60)
    print("  CAMPUSHIRE: STUDENT PLACEMENT ELIGIBILITY PREDICTOR")
    print("  Binary Classification using Logistic Regression")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: LOAD AND INSPECT THE DATASET
    # ---------------------------------------------------------
    data_path = os.path.join(os.path.dirname(__file__), "data", "student_placement_eligibility.csv")
    print(f"\n[Step 1] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Quick overview of dataset
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 rows of data:")
    print(df.head())

    # Check for missing values and duplicates
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"\nMissing values check: {missing_count} missing values found.")
    print(f"Duplicate rows check: {duplicate_count} duplicate rows found.")

    # Target class distribution (target = 1: Eligible, target = 0: Not Eligible)
    print("\nTarget Class Distribution:")
    class_counts = df['target'].value_counts()
    print(f" - Class 0 (Not Eligible): {class_counts.get(0, 0)} students ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f" - Class 1 (Eligible):     {class_counts.get(1, 0)} students ({class_counts.get(1, 0)/len(df)*100:.1f}%)")

    # ---------------------------------------------------------
    # STEP 2: PREPARE FEATURES (X) AND TARGET (y) & TRAIN/TEST SPLIT
    # ---------------------------------------------------------
    # X contains our 6 input variables (predictors)
    X = df.drop(columns=['target'])
    # y is our binary outcome to predict (0 or 1)
    y = df['target']
    feature_names = X.columns.tolist()

    # We use an 80% train / 20% test split.
    # 'stratify=y' ensures both train and test have the same 50/50 balance.
    # 'random_state=42' ensures our results are reproducible every run.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[Step 2] Data split completed: {len(X_train)} training samples, {len(X_test)} test samples.")

    # ---------------------------------------------------------

if __name__ == '__main__':
    main()
