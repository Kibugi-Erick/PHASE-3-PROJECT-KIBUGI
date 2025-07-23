# PHASE 3 PROJECT

# PREDICTING CUSTOMER CHURN FOR SyriaTel TELECOMMUNICATIONS

## Business Problem
#### Stakeholder:
The Marketing and Customer Retention Team at SyriaTel, a leading telecommunications provider in the region.


# 📉 SyriaTel Customer Churn Prediction

This project uses a dataset from SyriaTel, a telecommunications company, to predict customer churn using classification models. The goal is to help the business reduce customer loss by identifying users likely to churn before it happens.

---

## 🧠 Problem Statement

SyriaTel is facing customer churn, which is expensive to manage reactively. The business needs a predictive system that uses customer data to:
- Identify high-risk customers
- Inform proactive retention strategies
- Optimize resource allocation for marketing and customer support

---
## The libraries in use will be 
- pandas
- numpy
- matplotlib.pyplot
- seaborn
- sklearn.preprocessing.LabelEncoder
- sklearn.model_selection.train_test_split
- sklearn.preprocessing.StandardScaler
- sklearn.linear_model.LogisticRegression
- sklearn.tree.DecisionTreeClassifier
- sklearn.model_selection.GridSearchCV
- sklearn.metrics.confusion_matrix
- sklearn.metrics.classification_report
- sklearn.metrics.roc_auc_score
- sklearn.metrics.roc_curve
- sklearn.metrics.accuracy_score
- precision_score

---
## 📦 Dataset Overview

- **Rows**: 3,333 customers  
- **Target**: `churn` (binary: `True`/`False`)  
- **Features**:  
  - Demographics (`state`, `area code`, `account length`)
  - Service plans (`international plan`, `voice mail plan`)
  - Usage (`day/eve/night/intl minutes`, `calls`, `charges`)
  - Customer service interactions (`customer service calls`)

---

## 🧹 Preprocessing

- Dropped `phone number` (irrelevant)
- Categorical encoding:
  - Binary features (`yes`/`no`) → 0/1
  - One-hot encoding for `state` and `area code`
- Feature scaling for numerical features using `StandardScaler`
- Train/test split: 80% training, 20% testing

---

## 🔍 Exploratory Data Analysis (EDA)

- Churn rate in dataset: ~14%
- Customers who churn:
  - Often have **no voice mail plan**
  - Tend to have **more customer service calls**
  - May use **international plans**
- Class imbalance observed (majority class: non-churners)

---

## 🤖 Modeling

### 1. Logistic Regression (Baseline)
- **Accuracy**: 86%
- **Recall (churn)**: 27%
- **ROC AUC**: 0.80
- **Issue**: Poor performance on minority class

### 2. Decision Tree (Untuned)
- **Accuracy**: 91%
- **Recall (churn)**: 66%
- **Precision (churn)**: 68%
- **ROC AUC**: 0.80

### 3. Decision Tree (Tuned)
- **Best Params**: `max_depth=10`, `min_samples_leaf=4`
- **Accuracy**: 93%
- **Recall (churn)**: 67%
- **Precision (churn)**: 84%
- **F1-score (churn)**: 75%
- **ROC AUC**: 0.82

---

## ✅ Final Model Summary

The **tuned decision tree** provides the best balance of accuracy, interpretability, and churn detection performance. With a 93% accuracy and strong recall/precision, it is well-suited for operational deployment by SyriaTel’s marketing and retention teams.

---

## 📈 Business Recommendations

- **Target high-risk users** flagged by the model with retention offers
- **Improve customer service**, especially for users who call frequently
- **Incentivize loyalty** with discounts or perks for high-usage customers
- Retrain the model quarterly to adapt to evolving customer behavior

---