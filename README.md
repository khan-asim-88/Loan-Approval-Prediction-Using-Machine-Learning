# Loan Approval Prediction Using Machine Learning

MSc Data Science with Advanced Research — Final Project (University of Hertfordshire, 2025)

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Dataset Description](#2-dataset-description)
- [3. Research Objectives](#3-research-objectives)
- [4. Methodology](#4-methodology)
- [5. Results](#5-results)
- [6. Key Findings](#6-key-findings)
- [7. How to Run](#7-how-to-run)

## 1. Introduction

In the banking sector, assessing the creditworthiness of loan applicants is crucial to minimise financial risk and ensure profitability. Traditional loan approval processes often rely on manual evaluation, which can be time-consuming and inconsistent. This project develops a machine learning model to predict loan approval status, using ensemble learning and feature selection techniques to improve on individual classifiers.

## 2. Dataset Description

The dataset is Kaggle's [Loan Approval Classification Data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data). It contains **45,000 instances** with 13 features and the target variable `loan_status`, and has no missing values.

| Feature | Description |
|---|---|
| `person_age` | Applicant's age |
| `person_gender` | Applicant's gender |
| `person_education` | Applicant's highest education level |
| `person_income` | Applicant's annual income |
| `person_emp_exp` | Applicant's employment experience (years) |
| `person_home_ownership` | Home ownership status (RENT, OWN, MORTGAGE, OTHER) |
| `loan_amnt` | Requested loan amount |
| `loan_intent` | Purpose of the loan |
| `loan_int_rate` | Loan interest rate (%) |
| `loan_percent_income` | Loan amount as a proportion of annual income |
| `cb_person_cred_hist_length` | Credit history length (years) |
| `credit_score` | Applicant's credit score |
| `previous_loan_defaults_on_file` | Whether the applicant has a previous loan default on file (Yes/No) |

**Target:** `loan_status` — 1 = approved, 0 = rejected. The classes are imbalanced: roughly 78% of applications are rejected and 22% approved.

## 3. Research Objectives

1. **To assess the impact of ensemble learning techniques on the accuracy of loan approval predictions.**
2. **To evaluate the effectiveness of feature selection methods in enhancing the performance of loan approval prediction models.**
3. **To compare the predictive accuracy of combined ensemble learning and feature selection approaches against individual classifiers in loan approval modelling.**

## 4. Methodology

1. **Exploratory data analysis:** approval rates by category, distributions, box plots, correlation analysis and pair plots.
2. **Preprocessing:**
   - Log transformation of skewed features (`person_income`, `loan_amnt`)
   - Capping of `credit_score` outliers
   - Robust scaling of `loan_percent_income`
   - One-hot encoding (`person_home_ownership`, `loan_intent`) and label encoding (`previous_loan_defaults_on_file`, `age_group`)
3. **Feature selection:**
   - Variance Inflation Factor (VIF) to detect multicollinearity among numerical features
   - Cramér's V to measure the association of categorical features with `loan_status`
   - `person_gender` and `person_education` showed no association with approval (Cramér's V ≈ 0) and were removed, along with `person_age` (replaced by `age_group`)
4. **Modelling:** 80/20 train-test split, balanced class weights to handle class imbalance, and hyperparameter tuning with `GridSearchCV` (optimising ROC-AUC).
5. **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, ROC and precision-recall curves.
6. **Explainability:** feature importance and SHAP analysis.

## 5. Results

Performance on the test set (9,000 applications):

| Model | Type | Accuracy | ROC-AUC |
|---|---|---|---|
| Ridge Classifier | Individual | 82.2% | – |
| Lasso (L1 Logistic Regression) | Individual | 85.4% | 0.954 |
| Logistic Regression | Individual | 86.5% | 0.959 |
| Decision Tree (tuned) | Individual | 88.5% | 0.961 |
| AdaBoost (tuned) | Ensemble | 91.7% | 0.966 |
| Random Forest (tuned) | Ensemble | 93.2% | 0.977 |
| XGBoost (tuned) | Ensemble | 93.6% | 0.979 |
| **Gradient Boosting (tuned)** | **Ensemble** | **93.8%** | **0.980** |

**Best model — Gradient Boosting** (`learning_rate=0.2`, `max_depth=5`, `n_estimators=200`):

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Rejected (0) | 0.95 | 0.97 | 0.96 |
| Approved (1) | 0.90 | 0.81 | 0.85 |

Confusion matrix: 6,815 correct rejections, 1,630 correct approvals, 185 false approvals and 370 missed approvals.

## 6. Key Findings

- **Ensemble methods outperformed individual classifiers.** Every ensemble model beat the best individual model, with Gradient Boosting improving accuracy by about 7 percentage points over Logistic Regression.
- **Accuracy alone is misleading.** The Ridge Classifier achieved 97% recall on approvals but only 56% precision, meaning it would approve many risky applications. Precision, recall and ROC-AUC were needed to choose a model suitable for lending.
- **The strongest predictors** were previous loan defaults, loan interest rate, loan-to-income ratio and income, consistent with real-world credit risk logic.
- **Gender and education had no measurable relationship with approval**, so removing them simplified the model without losing performance.

## 7. How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/khan-asim-88/Loan-Approval-Prediction-Using-Machine-Learning.git
   cd Loan-Approval-Prediction-Using-Machine-Learning
   ```
2. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap statsmodels tabulate jupyter
   ```
3. Download `loan_data.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data) and place it in a `data/` folder:
   ```
   data/loan_data.csv
   ```
4. Open and run the notebook:
   ```bash
   jupyter notebook Loan_Approval_Prediction.ipynb
   ```

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.
