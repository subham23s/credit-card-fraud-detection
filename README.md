# 💳 Credit Card Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Project Overview

Credit card fraud is a major challenge in the financial industry. Fraudulent transactions represent only a very small percentage of total transactions, making fraud detection a highly **imbalanced classification problem**.

In this project, I built a Machine Learning pipeline that detects fraudulent transactions using:

✅ Logistic Regression

✅ Random Forest Classifier

✅ SMOTE Oversampling

✅ ROC-AUC Evaluation

✅ Feature Importance Analysis

The goal is not just achieving high accuracy, but effectively identifying fraudulent transactions while minimizing missed fraud cases.

---

## 🎯 Project Objectives

* Detect fraudulent credit card transactions
* Handle severe class imbalance
* Compare multiple ML algorithms
* Evaluate models using appropriate metrics
* Visualize fraud detection performance
* Understand real-world financial fraud detection challenges

---

## 📂 Dataset Information

📌 Dataset Source: Credit Card Fraud Detection Dataset (Kaggle)

### Dataset Features

| Feature Type | Description                                 |
| ------------ | ------------------------------------------- |
| V1-V28       | PCA-transformed anonymized features         |
| Time         | Seconds elapsed between transactions        |
| Amount       | Transaction amount                          |
| Class        | Target variable (0 = Legitimate, 1 = Fraud) |

---

## ⚠️ The Real Challenge

Fraud detection datasets are highly imbalanced.

Example:

* Legitimate Transactions → ~99.83%
* Fraudulent Transactions → ~0.17%

A model predicting **everything as legitimate** would still achieve extremely high accuracy while catching **zero fraud cases**.

That's why metrics such as:

* Recall
* ROC-AUC
* Precision
* Confusion Matrix

are more important than Accuracy.

---

## 🛠️ Technologies Used

| Category            | Tools               |
| ------------------- | ------------------- |
| Programming         | Python              |
| Data Analysis       | Pandas, NumPy       |
| Visualization       | Matplotlib, Seaborn |
| Machine Learning    | Scikit-Learn        |
| Imbalanced Learning | SMOTE               |
| Dataset Management  | KaggleHub           |

---

## 🔄 Project Workflow

### 📥 1. Dataset Loading

* Download dataset using KaggleHub
* Load into Pandas DataFrame
* Inspect shape and missing values

---

### 📊 2. Exploratory Data Analysis

* Class distribution analysis
* Fraud vs Legitimate transaction comparison
* Transaction amount visualization

Generated:

✅ EDA Plot

---

### 🧹 3. Data Preprocessing

* Scale Amount feature
* Scale Time feature
* Remove original columns
* Split dataset into train and test sets

---

### ⚖️ 4. Handle Class Imbalance

Applied:

**SMOTE (Synthetic Minority Oversampling Technique)**

Benefits:

✔ Generates synthetic fraud samples

✔ Balances training data

✔ Improves fraud detection capability

---

### 🤖 5. Model Training

#### Model 1: Logistic Regression

* Baseline classification model
* Fast and interpretable

#### Model 2: Random Forest

* Ensemble learning algorithm
* Handles complex relationships
* Better fraud detection performance

---

### 📈 6. Model Evaluation

Evaluation Metrics:

* Accuracy Score
* ROC-AUC Score
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

---

### 📉 7. ROC Curve Analysis

ROC Curve helps visualize:

* True Positive Rate
* False Positive Rate
* Model discrimination capability

Higher AUC = Better model performance.

---

### 🔍 8. Feature Importance Analysis

Random Forest Feature Importance was used to identify:

✅ Most influential fraud indicators

✅ Features contributing most to predictions

---

## 📁 Output Files

After running the project, the following visualizations are automatically generated:

📊 `eda_plots.png`

📊 `confusion_matrix.png`

📊 `roc_curve.png`

📊 `feature_importance.png`

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/subham23s/credit-card-fraud-detection

cd credit-card-fraud-detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python creditcard_fraud.py
```

---

## 🧠 What I Learned

Through this project, I learned:

✔ Handling imbalanced datasets

✔ Applying SMOTE effectively

✔ Data preprocessing techniques

✔ Feature scaling

✔ Classification model comparison

✔ ROC-AUC interpretation

✔ Confusion Matrix analysis

✔ Fraud detection evaluation strategies

✔ End-to-End Machine Learning workflow

---

## 🌟 Key Takeaway

> High Accuracy does NOT always mean a good model.

For highly imbalanced datasets like fraud detection, metrics such as Recall and ROC-AUC provide a much clearer picture of real-world performance.

---

## 👨‍💻 Author

### Subham Mishra

🎓 B.Tech CSE (AI/ML)

🔗 GitHub: [https://github.com/subham23s]

🔗 LinkedIn: [https://www.linkedin.com/in/subhammishra23/]

---

⭐ If you found this project useful, consider giving it a star on GitHub!
