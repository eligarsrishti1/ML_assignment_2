"""
app.py  -  Streamlit front-end for the ML Assignment 2 classifiers.

Loads PRE-TRAINED models from ./model (no re-training), lets the user upload
a test CSV (features + target column), choose a model, and shows metrics,
confusion matrix and classification report.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

st.set_page_config(page_title="ML Assignment 2 - Classifier Dashboard",
                   page_icon="🩺", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
CLASS_NAMES = ["Malignant (0)", "Benign (1)"]


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_data
def load_schema():
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        return json.load(f)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


schema = load_schema()
FEATURES, TARGET = schema["features"], schema["target"]

# ------------------------------------------------------------------ header
st.title("🩺 Breast Cancer Classification – Model Comparison")
st.caption("M.Tech (AIML) · Machine Learning · Assignment 2 · "
           "Dataset: Breast Cancer Wisconsin (Diagnostic) – UCI")

# ----------------------------------------------------------------- sidebar
st.sidebar.header("⚙️ Controls")
uploaded = st.sidebar.file_uploader(
    "Upload TEST data (CSV)", type=["csv"],
    help=f"Must contain the {len(FEATURES)} feature columns and the "
         f"target column '{TARGET}'.")

default_path = os.path.join(BASE_DIR, "test_data.csv")
if os.path.exists(default_path):
    with open(default_path, "rb") as f:
        st.sidebar.download_button("⬇️ Download sample test_data.csv", f,
                                   file_name="test_data.csv", mime="text/csv")

model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))
st.sidebar.info("Models are pre-trained and loaded from the `model/` folder. "
                "The uploaded data is used only for prediction/evaluation.")

if uploaded is None:
    st.info("👈 Upload a test CSV from the sidebar to begin "
            "(you can download a sample `test_data.csv` there).")
    st.stop()

# -------------------------------------------------------------- validation
try:
    data = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error("Uploaded CSV is missing required columns: " + ", ".join(missing))
    st.stop()

data = data[FEATURES + [TARGET]].dropna()
X, y = data[FEATURES], data[TARGET].astype(int)

if y.nunique() < 2:
    st.error("The target column needs both classes (0 and 1) to compute AUC.")
    st.stop()

with st.expander(f"📄 Preview of uploaded data ({data.shape[0]} rows × "
                 f"{data.shape[1]} columns)"):
    st.dataframe(data.head(20), use_container_width=True)

# ------------------------------------------------------ selected model view
model = load_model(model_name)
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]
m = compute_metrics(y, y_pred, y_prob)

st.subheader(f"Results – {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("#### ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
    ax2.plot(fpr, tpr, label=f"AUC = {m['AUC']:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

st.markdown("#### Classification Report")
report = classification_report(y, y_pred, target_names=CLASS_NAMES,
                               output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

# --------------------------------------------- all-model comparison on data
st.markdown("---")
st.subheader("📊 Comparison of all 5 models on the uploaded data")
rows = []
for name in MODEL_FILES:
    mdl = load_model(name)
    rows.append({"ML Model Name": name,
                 **compute_metrics(y, mdl.predict(X),
                                   mdl.predict_proba(X)[:, 1])})
comp = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(comp.style.highlight_max(axis=0, color="#c6efce"),
             use_container_width=True)

best = comp["MCC"].idxmax()
st.success(f"🏆 Best model on this data (by MCC): **{best}** "
           f"(MCC = {comp.loc[best, 'MCC']:.4f})")

fig3, ax3 = plt.subplots(figsize=(9, 3.8))
comp.plot(kind="bar", ax=ax3, width=0.8)
ax3.set_ylim(0.5, 1.02)
ax3.set_ylabel("Score")
ax3.set_xlabel("")
plt.xticks(rotation=20, ha="right")
ax3.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
st.pyplot(fig3)
