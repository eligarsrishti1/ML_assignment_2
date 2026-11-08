"""
train_models.py
---------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset (UCI), evaluates them on a held-out test set and saves:

    model/<name>.joblib   -> one full sklearn Pipeline per classifier
    model/metrics.csv     -> comparison table (used in README)
    model/feature_names.json
    test_data.csv         -> held-out test split (features + target)
    full_dataset.csv      -> the complete dataset (for reference)

Run:  python train_models.py
"""
import json
import os
import warnings

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TARGET = "target"
MODEL_DIR = "model"

# --------------------------------------------------------------------------
# 1. Load dataset (UCI WDBC, bundled with scikit-learn -> works offline)
# --------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()           # 30 features + 'target' (0 = malignant, 1 = benign)
df.to_csv("full_dataset.csv", index=False)
print(f"Dataset shape: {df.shape}  (instances, features+target)")
print(df[TARGET].value_counts().rename({0: "malignant", 1: "benign"}), "\n")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# --------------------------------------------------------------------------
# 2. Train / test split (80/20, stratified)
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

# Save test data (features + target) for the Streamlit app
test_df = X_test.copy()
test_df[TARGET] = y_test.values
test_df.to_csv("test_data.csv", index=False)

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
    json.dump({"features": list(X.columns), "target": TARGET}, f, indent=2)

# --------------------------------------------------------------------------
# 3. Define the 5 models (scaling is part of the pipeline so the saved
#    artifact is self-contained: no separate scaler file is needed)
# --------------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": Pipeline([
        ("clf", GaussianNB())]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
}

# --------------------------------------------------------------------------
# 4. Train, evaluate, save
# --------------------------------------------------------------------------
rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    rows.append({
        "ML Model Name": name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "AUC":       roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "MCC":       matthews_corrcoef(y_test, y_pred),
    })

    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
    print(f"Saved {fname}")

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
print("\n=== Evaluation metrics on test set ===")
print(metrics.to_string(index=False))
try:  # needs the optional 'tabulate' package; only used to print a README-ready table
    print("\nMarkdown table for README:\n")
    print(metrics.to_markdown(index=False))
except ImportError:
    pass
