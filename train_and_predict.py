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
    # STEP 3: FEATURE SCALING (STANDARDIZATION)
    # ---------------------------------------------------------
    # Features have very different scales (e.g., CGPA is 4-10, attendance is 0-100%, internship is 1-60 months).
    # StandardScaler transforms each feature to mean=0 and variance=1.
    # IMPORTANT: We fit the scaler ONLY on X_train to avoid data leakage!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[Step 3] Feature scaling applied (StandardScaler fitted on training data only).")

    # ---------------------------------------------------------
    # STEP 4: ESTABLISH A BASELINE MODEL
    # ---------------------------------------------------------
    # A simple baseline (predicting the most frequent class) provides a benchmark
    # to confirm that Logistic Regression actually learns meaningful patterns.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)
    print(f"\n[Step 4] Baseline Model (Dummy Most-Frequent) Accuracy: {baseline_acc:.4f} (50%)")

    # ---------------------------------------------------------
    # STEP 5: TRAIN THE LOGISTIC REGRESSION MODEL
    # ---------------------------------------------------------
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    print("[Step 5] Logistic Regression model trained successfully.")

    # ---------------------------------------------------------
    # STEP 6: MODEL EVALUATION ON UNSEEN TEST DATA
    # ---------------------------------------------------------
    # Predict binary class (0 or 1)
    y_pred = model.predict(X_test_scaled)
    # Predict probability of meeting eligibility (target = 1)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate key metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("  EVALUATION METRICS SUMMARY (TEST SET)")
    print("=" * 60)
    print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f} (When model predicts eligible, it is correct {precision * 100:.1f}% of the time)")
    print(f"Recall    : {recall:.4f} (Model successfully identified {recall * 100:.1f}% of all eligible students)")
    print(f"F1-Score  : {f1:.4f} (Harmonic mean of precision and recall)")
    print(f"ROC-AUC   : {roc_auc:.4f} (Discriminative ability between eligible vs non-eligible)")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Eligible (0)", "Eligible (1)"]))

    print("Confusion Matrix:")
    print(f"                       Predicted Not Eligible (0)   Predicted Eligible (1)")
    print(f"Actual Not Eligible (0)        {cm[0, 0]:<28} {cm[0, 1]}")
    print(f"Actual Eligible (1)            {cm[1, 0]:<28} {cm[1, 1]}")

    # ---------------------------------------------------------
    # STEP 7: SAVE VISUAL EVALUATION PLOTS
    # ---------------------------------------------------------
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Not Eligible (0)", "Eligible (1)"],
                yticklabels=["Not Eligible (0)", "Eligible (1)"])
    plt.title("Confusion Matrix - CampusHire Eligibility Model")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"\n[Step 7] Saved Confusion Matrix plot: {cm_plot_path}")

    # 2. Feature Coefficients Plot
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Odds_Ratio": np.exp(coefs)
    }).sort_values(by="Coefficient", ascending=True)

    plt.figure(figsize=(8, 5))
    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df["Coefficient"]]
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title("Feature Coefficients (Direction of Impact on Placement Eligibility)")
    plt.xlabel("Logistic Regression Coefficient (Log-Odds)")
    plt.ylabel("Features")
    plt.tight_layout()
    feat_plot_path = os.path.join(output_dir, "feature_coefficients.png")
    plt.savefig(feat_plot_path, dpi=200)
    plt.close()
    print(f"[Step 7] Saved Feature Coefficients plot: {feat_plot_path}")

    # ---------------------------------------------------------
    # STEP 8: INTERPRET COEFFICIENTS IN PLAIN LANGUAGE
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  COEFFICIENT & FEATURE IMPACT INTERPRETATION")
    print("=" * 60)
    print(coef_df.to_string(index=False))
    print("\nPractical Understanding:")
    for _, row in coef_df.iterrows():
        feat = row['Feature']
        c = row['Coefficient']
        odds = row['Odds_Ratio']
        if c > 0:
            print(f" • [INCREASES ELIGIBILITY CHANCE] {feat}: positive coefficient ({c:.3f}). Higher values boost eligibility probability (Odds Ratio: {odds:.2f}x).")
        else:
            print(f" • [DECREASES ELIGIBILITY CHANCE] {feat}: negative coefficient ({c:.3f}). Higher values reduce eligibility probability (Odds Ratio: {odds:.2f}x).")

    # ---------------------------------------------------------
    # STEP 9: INTERACTIVE / SAMPLE PREDICTION FUNCTION
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTION FOR A NEW STUDENT")
    print("=" * 60)

    def predict_placement_eligibility(cgpa, attendance_pct, coding_score, projects_completed, internship_months, backlogs):
        input_data = pd.DataFrame([{
            'cgpa': cgpa,
            'attendance_pct': attendance_pct,
            'coding_score': coding_score,
            'projects_completed': projects_completed,
            'internship_months': internship_months,
            'backlogs': backlogs
        }])
        # Scale the input using our fitted scaler
        input_scaled = scaler.transform(input_data)
        prob_eligible = model.predict_proba(input_scaled)[0, 1]
        prediction = 1 if prob_eligible >= 0.5 else 0
        status = "ELIGIBLE FOR PLACEMENT" if prediction == 1 else "NOT ELIGIBLE"
        return status, prob_eligible

    # Test Sample Student 1: Good attendance, multiple projects, high backlogs (in this synthetic dataset)
    sample_student1 = {"cgpa": 7.8, "attendance_pct": 85.0, "coding_score": 3,
                       "projects_completed": 5, "internship_months": 12, "backlogs": 4}
    status1, prob1 = predict_placement_eligibility(**sample_student1)
    print(f"Student A Profile: {sample_student1}")
    print(f"Prediction: {status1} (Eligibility Probability: {prob1*100:.1f}%)\n")

    # Test Sample Student 2: Low attendance, few projects
    sample_student2 = {"cgpa": 6.2, "attendance_pct": 35.0, "coding_score": 6,
                       "projects_completed": 1, "internship_months": 24, "backlogs": 1}
    status2, prob2 = predict_placement_eligibility(**sample_student2)
    print(f"Student B Profile: {sample_student2}")
    print(f"Prediction: {status2} (Eligibility Probability: {prob2*100:.1f}%)")

    print("\n" + "=" * 60)
    print("  CAMPUSHIRE EXECUTION COMPLETE! ALL ARTIFACTS SAVED.")
    print("=" * 60)

if __name__ == "__main__":
    main()
