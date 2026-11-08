# Breast Cancer Classification – ML Assignment 2 (M.Tech AIML, BITS WILP)

**Live Streamlit App:** `<PASTE YOUR STREAMLIT APP LINK HERE>`
**GitHub Repository:** `<PASTE YOUR GITHUB REPO LINK HERE>`

## 1. Problem Statement

Early and accurate diagnosis of breast cancer is critical for treatment outcomes. The goal of this project is to build, evaluate and compare **five machine-learning classification models** that predict whether a breast tumour is **malignant** or **benign** from measurements computed on a digitised image of a fine-needle aspirate (FNA) of a breast mass. The models are deployed in an interactive Streamlit web application that evaluates each saved model on user-uploaded test data.

## 2. Dataset Description  

| Property | Details |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic) – WDBC |
| Source | UCI Machine Learning Repository (also bundled with scikit-learn as `load_breast_cancer`) |
| Problem type | Binary classification |
| Instances | 569 (minimum required: 500) |
| Features | 30 numeric features (minimum required: 12) |
| Target | `target` – 0 = Malignant (212 samples), 1 = Benign (357 samples) |
| Missing values | None |

The 30 features are the **mean**, **standard error** and **worst (largest)** value of 10 real-valued characteristics of each cell nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

**Train/Test split:** 80% / 20%, stratified, `random_state=42` → 455 training and 114 test instances. The test split is saved as `test_data.csv`.

**Pre-processing:** `StandardScaler` is applied (inside a scikit-learn `Pipeline`) for Logistic Regression, KNN and SVM, which are scale-sensitive. Decision Tree and Gaussian Naive Bayes use raw features. Because scaling is part of each saved pipeline, no separate preprocessing file is needed.

### GitHub Repository Link  
`https://github.com/eligarsrishti1/ML_assignment_2.git`

### Repository structure
```
project-folder/
│-- app.py                  # Streamlit application
│-- train_models.py         # Training + evaluation + model saving script
│-- requirements.txt
│-- README.md
│-- test_data.csv           # Held-out test data (features + target)
│-- full_dataset.csv        # Complete dataset for reference
│-- model/
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn.joblib
    │-- naive_bayes.joblib
    │-- support_vector_machine.joblib
    │-- feature_names.json
    │-- metrics.csv
```

## 3. Models Used 

All five models were trained on the same training split and evaluated on the same 114-row test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

*Precision, Recall and F1 are computed for the positive class (Benign = 1). Hyper-parameters: Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=5`), KNN (`k=5`), Gaussian Naive Bayes (defaults), SVM (RBF kernel, `probability=True`).*

### Observations on model performance 

| ML Model Name | Observation about model performance |
|:--|:--|
| Logistic Regression | Joint-best model (Accuracy 98.25%, AUC 0.9954, MCC 0.9623). The classes are almost linearly separable after standardisation, so a simple linear model generalises very well while staying fast and interpretable. |
| Decision Tree | Weakest model (Accuracy 92.11%, AUC 0.9163, MCC 0.8341). A single depth-5 tree has high variance and its hard splits give poorly ranked probabilities, hence the lowest AUC. Its recall (0.9167) means it misses more benign cases than the other models. |
| KNN | Good and balanced performance (Accuracy 95.61%, AUC 0.9788, MCC 0.9054). It benefits strongly from feature scaling, but is a little less accurate than LR/SVM and is sensitive to the choice of `k` and to noise in a 30-dimensional space. |
| Naive Bayes | Decent accuracy (93.86%) but a very high AUC (0.9878), meaning it ranks cases well but its probability threshold is not optimal. The independence assumption is violated because many features (radius, perimeter, area) are strongly correlated, which limits accuracy and MCC (0.8676). |
| Support Vector Machine | Ties with Logistic Regression on every threshold-based metric (Accuracy 98.25%, F1 0.9861, MCC 0.9623) with AUC 0.9950. The RBF kernel captures non-linearity well on standardised data; it is slower and less interpretable than Logistic Regression. |
| **Overall, Winner for your dataset?** | **Logistic Regression** – it ties with SVM on Accuracy/F1/MCC, has the (marginally) highest AUC, and is simpler, faster and more interpretable. SVM is a very close runner-up. With only 114 test samples, a one-sample difference moves accuracy by ~0.9%, so the top-two ranking should be read with caution. |

## 4. Streamlit App Features

- **CSV upload** of test data (must contain the 30 feature columns and the `target` column); a sample `test_data.csv` can be downloaded from the sidebar.
- **Model selection dropdown** for the 5 classifiers.
- **Evaluation metrics** displayed: Accuracy, AUC, Precision, Recall, F1, MCC.
- **Confusion matrix**, **ROC curve** and **classification report**.
- **Comparison table** and bar chart of all 5 models on the uploaded data.
- Models are **loaded from `model/`** – nothing is re-trained in the app.

## 5. How to Run Locally

```bash
git clone https://github.com/eligarsrishti1/ML_assignment_2.git
cd <your-repo-folder>
pip install -r requirements.txt
python train_models.py        # optional: regenerates models, test_data.csv, metrics
streamlit run app.py
```
