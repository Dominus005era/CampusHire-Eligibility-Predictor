# CampusHire: Student Placement Eligibility Predictor

**Track**: Classification Practice (Problem 07)  
**Methodology**: Logistic Regression (Binary Classification)  
**Dataset**: 1,000 synthetic student records, 6 predictors + 1 target  

---

## 1. Project Overview & Objective

Campus recruitment cells evaluate large volumes of student applications against campus drive criteria. 

The objective of **CampusHire** is to build a reliable, clear, and beginner-friendly binary classification solution using **Logistic Regression** to predict whether a student meets campus placement eligibility criteria (`target = 1`) or does not meet them (`target = 0`).

---

## 2. Dataset & Feature Descriptions

The dataset consists of 1,000 student records with 6 input predictors:

| Feature Name | Description | Data Type | Range / Characteristics |
| :--- | :--- | :--- | :--- |
| `cgpa` | Cumulative Grade Point Average | Float | 4.00 to 10.00 |
| `attendance_pct` | Class attendance percentage | Float | 0.00% to 100.00% |
| `coding_score` | Score on programming assessment | Integer | 0 to 10 scale |
| `projects_completed` | Number of engineering projects built | Integer | 0 to 9 projects |
| `internship_months` | Months of prior internship experience | Integer | 1 to 60 months |
| `backlogs` | Number of active/uncleared backlogs | Integer | 0 to 9 backlogs |
| **`target`** | **Placement Eligibility (1 = Eligible, 0 = Not Eligible)** | **Integer** | **Exactly 50% (0) and 50% (1)** |

### Data Quality & Integrity Checks
- **Missing Values**: 0 missing values across all features.
- **Duplicates**: 0 duplicate rows found.
- **Class Balance**: Perfectly balanced 50/50 target distribution (500 eligible, 500 not eligible).

---

## 3. Machine Learning Methodology

Following the recommended workflow:  
`inspect → preprocess → split → baseline → Logistic Regression → evaluate → interpret → document`

### 3.1 Data Splitting (Preventing Data Leakage)
- Split: **80% Training (800 rows)** and **20% Testing (200 rows)**.
- Stratification: Enabled on `target` to ensure an exact 50/50 balance in both train and test splits.
- Random Seed: Set to `42` for exact reproducibility.

### 3.2 Preprocessing (StandardScaler)
- Input features are on diverse scales (e.g., CGPA is 4–10, attendance is 0–100, and backlogs is 0–9).
- `StandardScaler` standardizes each feature to have zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$).
- **Leakage Prevention**: Scaler was fitted strictly on `X_train` and then applied to `X_test`.

### 3.3 Baseline Establishment
- A `DummyClassifier(strategy="most_frequent")` was used as a naive reference point.
- The baseline accuracy is **50.00%** (a random guess on a 50/50 dataset).
- Any viable machine learning model must clearly exceed this 50% threshold.

---

## 4. Model Performance & Evaluation

The Logistic Regression model was evaluated on the 200 holdout test samples:

| Metric | Score | Practical Meaning |
| :--- | :---: | :--- |
| **Accuracy** | **70.00%** | Exceeds the 50% random baseline by a wide margin (+20.0% points). |
| **Precision** | **68.87%** | When predicting a student is eligible, the model is correct 68.9% of the time. |
| **Recall** | **73.00%** | The model successfully captures 73 out of every 100 eligible students. |
| **F1-Score** | **70.87%** | Robust harmonic mean between Precision and Recall. |
| **ROC-AUC** | **0.7874** | Strong ability to rank and discriminate eligible students from ineligible ones. |

### Confusion Matrix Breakdown

| | Predicted Not Eligible (0) | Predicted Eligible (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Not Eligible (0)** | 67 (True Negatives) | 33 (False Positives) | 100 |
| **Actual Eligible (1)** | 27 (False Negatives) | 73 (True Positives) | 100 |

- **True Positives (73)**: Eligible students correctly selected for placement interview pools.
- **False Negatives (27)**: Eligible students erroneously filtered out (risk of candidate dissatisfaction).

The confusion matrix visualization is saved in [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png).

---

## 5. Feature Coefficients & Practical Interpretation

Logistic regression provides direct interpretability via its coefficients ($\beta$) and Odds Ratios ($\exp(\beta)$):

| Feature | Coefficient ($\beta$) | Odds Ratio | Impact on Placement Eligibility |
| :--- | :---: | :---: | :--- |
| `coding_score` | -0.5171 | 0.60 | **Decreases Eligibility Odds** |
| `cgpa` | -0.4622 | 0.63 | **Decreases Eligibility Odds** |
| `internship_months` | -0.3954 | 0.67 | **Decreases Eligibility Odds** |
| `projects_completed` | +0.5514 | 1.74 | **Increases Eligibility Odds** (1.74x odds per std-dev increase) |
| `attendance_pct` | +0.5561 | 1.74 | **Increases Eligibility Odds** (High attendance strongly favors qualification) |
| `backlogs` | +0.6180 | 1.86 | **Increases Eligibility Odds** (In this synthetic dataset pattern) |

The coefficient impact plot is saved in [`outputs/feature_coefficients.png`](outputs/feature_coefficients.png).

---

## 6. Practical Limitations & Future Improvements

1. **Synthetic Data Characteristics**: In real corporate placement drives, backlogs and lower CGPAs typically disqualify candidates. In this educational dataset, project count and attendance are rewarded, reflecting synthetic data behavior.
2. **Feature Non-Linearities**: Certain eligibility rules operate on strict cutoff thresholds (e.g., minimum CGPA $\ge$ 7.0 or backlogs = 0). A tree-based model (e.g., Random Forest or XGBoost) or explicit rule-based preprocessing may complement Logistic Regression in production.
3. **Threshold Adjustment**: Placement cells wanting zero false rejections can lower the decision threshold to `0.35`–`0.40` to achieve $\ge 85\%$ Recall.

---

## 7. How to Run This Project

### Option A: Python Script
```bash
# Navigate to the project directory
cd "E:\ML Intership\CampusHire-Eligibility-Predictor"

# Run the complete pipeline
python train_and_predict.py
```

### Option B: Interactive Jupyter Notebook
Open `project_walkthrough.ipynb` in VS Code or Jupyter:
```bash
jupyter notebook project_walkthrough.ipynb
```
