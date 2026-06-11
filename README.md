# Heart Disease Prediction — Data Mining Framework

A complete, end-to-end data mining pipeline built in **Java** using the **Weka** library, developed as a university project for the Data Mining course at **Vietnam National University — International University (IU-HCM)**.

The framework predicts heart disease from patient records using classification models, and discovers frequent patterns through association rule mining.

---

## Project Overview

| Detail | Value |
|---|---|
| Language | Java |
| Library | Weka |
| Dataset | `heart_disease.csv` — 10,000 instances, 21 attributes |
| Target Variable | `Heart Disease Status` (Yes / No) |
| Class Distribution | ~80% No, ~20% Yes (imbalanced) |

---

## Pipeline Architecture

```
CSV Input
   │
   ▼
Data Preprocessing
   ├── Missing value imputation (median/mode)
   ├── Duplicate removal
   └── Outlier detection (IQR method)
   │
   ▼
Feature Selection
   └── Information Gain + Ranker → top 11 attributes retained
   │
   ▼
Train / Test Split (70 / 30)
   └── Resample filter applied to training set only (bias toward minority class)
   │
   ▼
Classification
   ├── Naive Bayes
   └── Random Forest (100 trees)
   │
   ▼
Model Evaluation
   └── 10-fold cross-validation
       (Accuracy, F1, Precision, Recall, FP Rate, ROC)
   │
   ▼
Association Rule Mining
   └── Apriori on discretized attributes
       (Support ≥ 0.05, Confidence ≥ 0.8, Top 10 rules)
   │
   ▼
Model Persistence
   └── SerializationHelper saves .model files for reuse
```

---

## Methods

### Classification

**Naive Bayes**
- Probabilistic baseline based on Bayes' Theorem
- Assumes feature independence
- Fast, efficient on high-dimensional data
- Better at detecting the minority (disease-positive) class

**Random Forest**
- Ensemble of 100 decision trees
- Combines bagging and random feature selection
- Handles nonlinear relationships and noisy data
- Higher overall accuracy but weaker minority-class recall

### Association Rule Mining (Apriori)

Numeric attributes were discretized into 5 bins before mining. Rules were filtered by confidence ≥ 0.8 and support ≥ 0.05. Key finding: healthier lifestyle indicators (low stress, non-smoking, normal cholesterol) are consistently associated with the absence of heart disease.

---

## Results

### Hold-out Evaluation (70/30 Split)

| Metric | Naive Bayes | Random Forest |
|---|---|---|
| Accuracy | 52.67% | **62.17%** |
| F1-score (No) | 0.643 | **0.749** |
| Recall (No) | 0.533 | **0.704** |
| F1-score (Yes) | **0.298** | 0.237 |
| Recall (Yes) | **0.502** | 0.293 |
| Execution Time | 373 ms | 1,346 ms |

### 10-Fold Cross-Validation

| Metric | Naive Bayes | Random Forest |
|---|---|---|
| Accuracy | 50.76% | **61.69%** |
| F1-score (Yes) | **0.289** | 0.239 |
| Recall (Yes) | **0.501** | 0.302 |
| F1-score (No) | 0.623 | **0.744** |
| Execution Time | 1,321 ms | 10,405 ms |

**Key takeaway:** Random Forest wins on overall accuracy; Naive Bayes is better at catching true heart disease cases (higher minority-class recall). The choice depends on whether the priority is general correctness or clinical sensitivity.

---

## Class Imbalance Handling

Initial attempts with **SMOTE** (Weka API) caused data leakage and instability with mixed attribute types. The final approach uses Weka's **Resample** filter with bias toward the minority class, applied only to the training split. This approach:

- Works cleanly with mixed numeric/nominal data
- Avoids leakage by being applied post-split
- Allows fine-tuning of the target class distribution (50-50 or 60-40)

---

## Feature Selection

Information Gain combined with the Ranker method was used to score all 21 attributes. The **top 11** were retained for modeling:

- Age, Gender, Blood Pressure, Cholesterol Level, Exercise Habits
- Smoking, Family Heart Disease, Diabetes, BMI, High Blood Pressure, Stress Level

---

## Known Limitations & Future Work

- ROC values are low (~0.50), reflecting the inherent difficulty of separating classes in this dataset due to overlapping feature distributions
- No hyperparameter tuning was performed on Random Forest (default settings only)
- Potential improvements:
  - Cost-sensitive learning
  - Advanced ensemble methods (XGBoost, LightGBM via PMML bridge)
  - Grid search over `numTrees`, `maxDepth`, and class weights
  - Alternative evaluation metrics (MCC, PR-AUC) better suited for imbalanced data

---

## Setup & JVM Requirements

This project was developed with **Weka as an external dependency**. Due to Java 9+ module restrictions, the following JVM arguments are required at runtime:

```
--add-opens java.base/java.lang=ALL-UNNAMED
--add-opens java.base/java.io=ALL-UNNAMED
--add-opens java.base/java.util=ALL-UNNAMED
```

---

## Project Structure

```
├── src/
│   ├── CSVtoARFF.java          # CSV → ARFF conversion
│   ├── DataPreprocessing.java  # Cleaning, imputation, outlier detection
│   ├── Classification.java     # Train/test split, resampling, model training
│   ├── ModelEvaluation.java    # 10-fold cross-validation
│   └── AprioriMining.java      # Discretization + association rules
├── models/
│   ├── NAIVEBAYES.model
│   └── RANDOMFOREST.model
├── data/
│   ├── heart_disease.csv
│   └── heart_disease.arff
└── README.md
```

---

## Author

**Nguyễn Hoàng Bảo** — ITDSIU23004  
Vietnam National University HCMC — International University  
Course: Data Mining | Instructor: Dr. Nguyen Thi Thanh Sang
