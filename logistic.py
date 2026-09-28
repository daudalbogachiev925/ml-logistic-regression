"""Логистическая регрессия с полным пайплайном."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt

# === 1. Данные ===
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)

# === 2. Препроцессинг ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# === 3. Модель ===
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, y_train)

# === 4. Метрики ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

# === 5. Кросс-валидация ===
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# === 6. ROC-кривая ===
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC-кривая")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

# === 7. Матрица ошибок ===
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
