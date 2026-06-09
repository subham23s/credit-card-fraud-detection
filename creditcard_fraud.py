# ===== IMPORTS =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,
                              classification_report,
                              confusion_matrix,
                              roc_auc_score,
                              roc_curve,
                              precision_recall_curve)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported!")


# =============================================================
# STEP 1 — LOAD DATA USING KAGGLEHUB
# =============================================================

print("\n📥 Downloading dataset...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print(f"Dataset path: {path}")

df = pd.read_csv(os.path.join(path, "creditcard.csv"))

print("\n📊 Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())


# =============================================================
# STEP 2 — UNDERSTAND THE IMBALANCE PROBLEM
# =============================================================
# This is the CORE problem of today's project
# Only 0.17% transactions are fraud!
# A dumb model that predicts "all legitimate" gets 99.83% accuracy
# but catches ZERO fraud — completely useless!

fraud_count = df['Class'].value_counts()
print("\n🔍 Class Distribution:")
print(f"   Legitimate (0): {fraud_count[0]:,} ({fraud_count[0]/len(df)*100:.2f}%)")
print(f"   Fraud (1)     : {fraud_count[1]:,} ({fraud_count[1]/len(df)*100:.2f}%)")
print(f"\n   Imbalance ratio: {fraud_count[0]//fraud_count[1]}:1")

# Visualize the imbalance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Class distribution
axes[0].bar(['Legitimate', 'Fraud'],
            [fraud_count[0], fraud_count[1]],
            color=['steelblue', 'coral'])
axes[0].set_title('Class Distribution — Severely Imbalanced!')
axes[0].set_ylabel('Count')
for i, v in enumerate([fraud_count[0], fraud_count[1]]):
    axes[0].text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')

# Plot 2: Transaction amount by class
df[df['Class']==1]['Amount'].hist(ax=axes[1], bins=50,
                                   alpha=0.7, color='coral',
                                   label='Fraud')
df[df['Class']==0]['Amount'].hist(ax=axes[1], bins=50,
                                   alpha=0.5, color='steelblue',
                                   label='Legitimate')
axes[1].set_title('Transaction Amount Distribution')
axes[1].set_xlabel('Amount')
axes[1].legend()
axes[1].set_xlim(0, 500)

plt.tight_layout()
plt.savefig('eda_plots.png')
plt.show()
print("✅ EDA plot saved!")

# Show why accuracy is misleading
dumb_accuracy = fraud_count[0] / len(df) * 100
print(f"\n⚠️  A model predicting ALL as legitimate gets: {dumb_accuracy:.2f}% accuracy")
print("   But it catches ZERO fraud! This is why we need better metrics.")


# =============================================================
# STEP 3 — PREPARE DATA
# =============================================================

# Scale Amount and Time (other features V1-V28 are already scaled)
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled']   = scaler.fit_transform(df[['Time']])

# Drop original Amount and Time
df = df.drop(['Amount', 'Time'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

# Split BEFORE applying SMOTE (never apply SMOTE to test data!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing samples  : {len(X_test):,}")
print(f"\nTraining fraud cases  : {y_train.sum():,}")
print(f"Training legit cases  : {(y_train==0).sum():,}")


# =============================================================
# STEP 4 — SMOTE (Synthetic Minority Oversampling Technique)
# =============================================================
# SMOTE creates SYNTHETIC fraud samples to balance the dataset
# It doesn't just copy existing fraud samples
# It creates NEW fake-but-realistic fraud samples
# by interpolating between existing fraud samples
#
# IMPORTANT: Apply SMOTE only on TRAINING data, never on test!

print("\n⚖️ Applying SMOTE to balance training data...")
print(f"Before SMOTE — Fraud: {y_train.sum():,} | Legit: {(y_train==0).sum():,}")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"After SMOTE  — Fraud: {y_train_smote.sum():,} | Legit: {(y_train_smote==0).sum():,}")
print(f"Training samples after SMOTE: {len(X_train_smote):,}")
print("✅ Dataset balanced!")


# =============================================================
# STEP 5 — TRAIN MODEL 1: LOGISTIC REGRESSION
# =============================================================

print("\n📐 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_smote, y_train_smote)
lr_pred     = lr.predict(X_test)
lr_pred_prob= lr.predict_proba(X_test)[:, 1]
lr_auc      = roc_auc_score(y_test, lr_pred_prob)
lr_acc      = accuracy_score(y_test, lr_pred)

print(f"Logistic Regression:")
print(f"  Accuracy : {lr_acc*100:.2f}%")
print(f"  ROC-AUC  : {lr_auc:.4f}")


# =============================================================
# STEP 6 — TRAIN MODEL 2: RANDOM FOREST
# =============================================================

print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42,
                             n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)
rf_pred     = rf.predict(X_test)
rf_pred_prob= rf.predict_proba(X_test)[:, 1]
rf_auc      = roc_auc_score(y_test, rf_pred_prob)
rf_acc      = accuracy_score(y_test, rf_pred)

print(f"Random Forest:")
print(f"  Accuracy : {rf_acc*100:.2f}%")
print(f"  ROC-AUC  : {rf_auc:.4f}")


# =============================================================
# STEP 7 — DETAILED EVALUATION (BEST MODEL)
# =============================================================
# For fraud detection, RECALL matters most!
# Recall = out of all actual frauds, how many did we catch?
# Missing a fraud (False Negative) is worse than a false alarm

print("\n📋 Detailed Report — Random Forest:")
print(classification_report(y_test, rf_pred,
      target_names=['Legitimate', 'Fraud']))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('Confusion Matrix — Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Extract key numbers
tn, fp, fn, tp = cm.ravel()
print(f"\n🔍 Fraud Detection Breakdown:")
print(f"   True Positives  (caught fraud)    : {tp}")
print(f"   False Negatives (missed fraud)    : {fn}")
print(f"   False Positives (false alarm)     : {fp}")
print(f"   True Negatives  (correct legit)   : {tn}")
print(f"\n   Fraud caught: {tp}/{tp+fn} = {tp/(tp+fn)*100:.1f}%")


# =============================================================
# STEP 8 — ROC CURVE
# =============================================================
# ROC curve shows tradeoff between catching fraud vs false alarms
# AUC = 1.0 is perfect, AUC = 0.5 is random guessing

plt.figure(figsize=(8, 5))

# Logistic Regression ROC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_pred_prob)
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC={lr_auc:.3f})',
         color='steelblue', linewidth=2)

# Random Forest ROC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_prob)
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC={rf_auc:.3f})',
         color='coral', linewidth=2)

# Random baseline
plt.plot([0, 1], [0, 1], 'k--', label='Random baseline (AUC=0.5)')

plt.title('ROC Curve — Fraud Detection')
plt.xlabel('False Positive Rate (False Alarms)')
plt.ylabel('True Positive Rate (Fraud Caught)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()
print("✅ ROC curve saved!")


# =============================================================
# STEP 9 — FEATURE IMPORTANCE
# =============================================================

importance = pd.DataFrame({
    'Feature'   : X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance, x='Importance', y='Feature',
            palette='viridis')
plt.title('Top 10 Most Important Features for Fraud Detection')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("✅ Feature importance saved!")

print("\n🔑 Top 5 fraud indicators:")
print(importance.head())


# =============================================================
# STEP 10 — MODEL COMPARISON SUMMARY
# =============================================================

print("\n📊 Final Model Comparison:")
print(f"{'Model':<25} {'Accuracy':<15} {'ROC-AUC'}")
print("-" * 50)
print(f"{'Logistic Regression':<25} {lr_acc*100:.2f}%"
      f"         {lr_auc:.4f}")
print(f"{'Random Forest':<25} {rf_acc*100:.2f}%"
      f"         {rf_auc:.4f}")

print(f"\n✅ Best model: Random Forest (AUC: {rf_auc:.4f})")
print("\n💡 Key insight: Don't trust accuracy for imbalanced data!")
print(f"   A dumb model gets {dumb_accuracy:.2f}% accuracy but AUC of 0.5")
print(f"   Our Random Forest gets AUC of {rf_auc:.4f} — much better!")

print("\n🎉 Project Complete!")