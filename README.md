# 🫀 Heart Disease Prediction — Data Mining Pipeline

A complete, end-to-end data mining pipeline for heart disease prediction, implemented in **Python** as a reimplementation and extension of an original **Java/Weka** university project.

> **Course:** Data Mining  
> **Institution:** Vietnam National University HCMC — International University (IU-HCM)  
> **Author:** Nguyễn Hoàng Bảo (ITDSIU23004)  
> **Instructor:** Dr. Nguyen Thi Thanh Sang

---

## 📋 Project Overview

This project applies a full data mining workflow — cleaning, feature selection, classification, and association rule mining — to predict heart disease from clinical patient records.

| Detail | Value |
|---|---|
| Language | Python 3 |
| Key Libraries | scikit-learn, imbalanced-learn, mlxtend, pandas, numpy |
| Dataset | `heart_disease.csv` — 10,000 instances, 21 attributes |
| Target Variable | `Heart Disease Status` (Yes / No) |
| Class Distribution | 80% No Disease / 20% Disease (imbalanced) |

---

## 🗂️ Project Structure

```
Heart-Disease/
├── code/
│   ├── main.py               # Orchestrates the full pipeline
│   ├── processing.py         # Data cleaning & feature selection (DataPipeline)
│   ├── classification.py     # Model training & evaluation (ClassificationEngine)
│   └── association.py        # Apriori rule mining (AssociationMiner)
├── data/
│   ├── heart_disease.csv     # Raw dataset (10,000 instances, 21 attributes)
│   └── After_Cleaning.csv    # Cleaned & feature-selected output
└── README.md
```

---

## ⚙️ Pipeline Architecture

```
Raw CSV Input (10,000 × 21)
         │
         ▼
  Data Preprocessing
  ├── Missing value imputation  (median for numeric, mode for categorical)
  ├── Duplicate removal
  └── Outlier detection & capping  (IQR method, threshold = 1%)
         │
         ▼
  Feature Selection
  └── Mutual Information (Information Gain) + Ranker → top 11 attributes
         │
         ▼
  Train / Test Split  (70% / 30%, stratified)
  └── RandomOverSampler applied to training set only  (avoids data leakage)
         │
         ▼
  Classification
  ├── Naive Bayes  (GaussianNB)
  └── Random Forest  (100 trees)
         │
         ▼
  Model Evaluation
  ├── Holdout test set  (30%)
  └── 10-Fold Stratified Cross-Validation
         │
         ▼
  Association Rule Mining
  └── Apriori on discretized features
      (support ≥ 0.05, confidence ≥ 0.80, top 10 rules by lift)
```

---

## 🔬 Methods

### Data Preprocessing (`processing.py`)

- **Missing value imputation** — numeric columns filled with the column median; categorical columns filled with the mode.
- **Duplicate removal** — exact duplicate rows are dropped before any modeling.
- **Outlier handling** — IQR-based detection per numeric column. Columns where outliers exceed 1% of the dataset have outlier values replaced with the column median (rather than dropped, to preserve dataset size).

### Feature Selection (`processing.py`)

Mutual Information scores are computed for all 21 attributes against the target label. The top **11 features** are retained:

`Age`, `Gender`, `Blood Pressure`, `Cholesterol Level`, `Exercise Habits`, `Smoking`, `Family Heart Disease`, `Diabetes`, `BMI`, `High Blood Pressure`, `Stress Level`

### Classification (`classification.py`)

Two classifiers are trained and evaluated:

**Naive Bayes (GaussianNB)**
- Probabilistic baseline using Bayes' Theorem with feature independence assumption
- Fast training; performs better on the minority (disease-positive) class
- Categorical features are one-hot encoded before training

**Random Forest (100 trees)**
- Ensemble of decision trees combining bagging and random feature subsampling
- Higher overall accuracy; handles nonlinear relationships and feature interactions
- Weaker minority-class recall compared to Naive Bayes

**Class imbalance handling** — `RandomOverSampler` from imbalanced-learn is used to balance the training set. Critically, it is applied *inside* the cross-validation loop (via `imblearn.Pipeline`) to prevent data leakage into validation folds. This replicates the behaviour of Weka's biased `Resample` filter from the original Java implementation.

### Association Rule Mining (`association.py`)

All numeric features are discretized into 5 ordinal bins (`Low`, `Low-Mid`, `Medium`, `Mid-High`, `High`). The encoded boolean transaction matrix is passed to `mlxtend`'s Apriori implementation. Rules are filtered by confidence ≥ 0.80 and support ≥ 0.05, then ranked by lift. Rules whose consequent references `Heart Disease Status` are surfaced as the key findings.

**Key finding:** lifestyle indicators such as low stress, non-smoking, and normal cholesterol are consistently associated with the *absence* of heart disease.

---

## 📊 Results

### Holdout Evaluation (70/30 Split)

| Metric | Naive Bayes | Random Forest |
|---|---|---|
| Accuracy | 52.67% | **62.17%** |
| F1-score (No Disease) | 0.643 | **0.749** |
| Recall (No Disease) | 0.533 | **0.704** |
| F1-score (Heart Disease) | **0.298** | 0.237 |
| Recall (Heart Disease) | **0.502** | 0.293 |
| Training Time | ~373 ms | ~1,346 ms |

### 10-Fold Cross-Validation

| Metric | Naive Bayes | Random Forest |
|---|---|---|
| Accuracy | 50.76% | **61.69%** |
| Macro F1-score | **0.456** | 0.492 |
| Recall (Heart Disease) | **0.501** | 0.302 |
| CV Time (total) | ~1.3 s | ~10.4 s |

**Interpretation:** Random Forest wins on overall accuracy. Naive Bayes achieves higher recall on the minority (disease-positive) class, making it preferable in a clinical sensitivity context where missing a true positive has a higher cost.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn imbalanced-learn mlxtend
```

### Run the full pipeline

```bash
# From the project root
python code/main.py
```

This will:
1. Load and analyse the raw dataset
2. Run the full cleaning and feature selection pipeline
3. Save the cleaned dataset to `data/After_Cleaning.csv`
4. Train and evaluate both classifiers (holdout + 10-fold CV)
5. Run Apriori association rule mining and print the top 10 rules

---

## ⚠️ Known Limitations & Future Work

- ROC values are low (~0.50), reflecting overlapping feature distributions between the two classes in this dataset
- No hyperparameter tuning was performed (default `n_estimators=100`, no `max_depth`)
- Potential improvements:
  - Cost-sensitive learning to penalise false negatives more heavily
  - Advanced ensemble methods (XGBoost, LightGBM)
  - Grid search over `n_estimators`, `max_depth`, and class weights
  - Alternative metrics better suited for imbalanced data: Matthews Correlation Coefficient (MCC), PR-AUC

---

## 🔗 Background: Java/Weka Implementation

This Python codebase is a faithful reimplementation of an original pipeline built in Java using the Weka library. The Java version used:

- `ReplaceMissingValues` → replicated as pandas median/mode imputation
- `InformationGain + Ranker` → replicated as `mutual_info_classif`
- `Resample` (biased) → replicated as `RandomOverSampler`
- `NaiveBayes` / `RandomForest` → replicated as `GaussianNB` / `RandomForestClassifier`
- `Apriori` → replicated via `mlxtend.frequent_patterns`

The Python implementation adds pipeline-level leakage prevention for cross-validation that was not present in the original Weka approach.

---

## 📄 License

This project was developed for academic purposes at IU-HCM. Dataset and results are for educational use only.
