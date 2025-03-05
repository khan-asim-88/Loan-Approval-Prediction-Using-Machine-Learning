# Loan Approval Prediction Using Machine Learning

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Dataset Description](#2-dataset-description)
- [3. Research Objectives](#3-research-objectives)

## 1. Introduction

In the banking sector, assessing the creditworthiness of loan applicants is crucial to minimize financial risks and ensure profitability. Traditional loan approval processes often involve manual evaluations, which can be time-consuming and prone to inconsistencies. To enhance decision-making efficiency and accuracy, machine learning techniques have been increasingly adopted to predict loan approval outcomes based on applicant data. This report presents a study on developing a machine learning model to predict loan approval status by leveraging ensemble learning and feature selection techniques.

## 2. Dataset Description

The dataset utilized in this study is sourced from Kaggle's "Loan Approval Classification Data" and comprises 13 features along with the target variable `loan_status`. The features encompass demographic information, financial status, and loan details of applicants, as detailed below:

- `person_age`: Applicant's age
- `person_gender`: Applicant's gender
- `person_education`: Applicant's education level
- `person_income`: Applicant's annual income
- `person_emp_exp`: Applicant's professional experience (years)
- `person_home_ownership`: Applicant's home ownership status
- `loan_amnt`: Requested loan amount
- `loan_intent`: Purpose of the loan
- `loan_int_rate`: Loan interest rate (%)
- `loan_percent_income`: Ratio of loan amount to annual income
- `cb_person_cred_hist_length`: Credit history length (years)
- `credit_score`: Applicant's credit score
- `previous_loan_defaults_on_file`: Number of previous loan defaults

The target variable, `loan_status`, indicates whether a loan application was approved or not. The dataset contains 10,000 instances, with no missing values reported.

## 3. Research Objectives

The primary objectives of this study are:

1. **To assess the impact of ensemble learning techniques on the accuracy of loan approval predictions.**
2. **To evaluate the effectiveness of feature selection methods in enhancing the performance of loan approval prediction models.**
3. **To compare the predictive accuracy of combined ensemble learning and feature selection approaches against individual classifiers in loan approval modeling.**
