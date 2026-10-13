# 🔄 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.60-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

An end-to-end machine learning project that predicts whether a telecom customer will leave (churn). It covers exploratory analysis, feature engineering, model selection, and an interactive Streamlit app that scores a single customer.

---

## 🎯 Business Problem

A telecom company loses revenue every time a customer cancels. The goal is to:

- **Predict** which customers are likely to churn
- **Identify** factors associated with churn
- **Prioritize** high-risk customers for retention outreach

---

## 📊 Dataset

- **Source:** IBM Telco Customer Churn sample dataset, available on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Raw rows:** 7,043 customers, 21 columns
- **After cleaning:** 7,032 customers. 11 rows with a blank `TotalCharges` (all with tenure 0) are removed.
- **Churn rate:** 26.6% (1,869 churners) in the cleaned data

---

## 💡 Key Findings from EDA

These are descriptive statistics from the cleaned data (7,032 rows). They describe associations, not causes.

| Finding | Detail |
|---------|--------|
| Month-to-month churn | 42.7% |
| One-year contract churn | 11.3% |
| Two-year contract churn | 2.8% |
| Churner median tenure | 10 months |
| Stayer median tenure | 38 months |
| Churner average monthly charges | $74.44 |
| Stayer average monthly charges | $61.31 |

**Strongest correlations with churn** (Pearson correlation with the `Churn` column, not model feature importance):

1. `tenure`: −0.35
2. `InternetService_Fiber optic`: +0.31
3. `Contract_Two year`: −0.30
4. `PaymentMethod_Electronic check`: +0.30
5. `TechSupport_No internet service`: −0.23

---

## ⚙️ Feature Engineering

The model uses 36 features: the cleaned columns, one-hot encoded, plus five engineered features:

| Feature | Definition |
|---------|-----------|
| `tenure_group` | New (0–12 months), Mid (13–48), Loyal (49–72) |
| `total_services` | Number of the 8 add-on services set to "Yes" |
| `is_new_customer` | 1 if tenure ≤ 12 months |
| `avg_monthly_from_total` | `TotalCharges / tenure` (tenure 0 is treated as 1) |
| `high_value_customer` | 1 if `MonthlyCharges` is above the training-set median ($70.60) |

The app computes these features with the functions in `src/features.py`. The notebook computes the same features inline, so any change to the feature logic must be made in both places.

---

## 🤖 Modeling

- **Split:** 80/20 stratified train/test (5,625 / 1,407 rows)
- **Imbalance:** SMOTE is applied to the training data only, giving 8,260 training rows
- **Models compared:** Logistic Regression (LR) and Random Forest (RF)
- **Model selection:** 5-fold stratified cross-validation on the training data, with scaling and SMOTE inside each fold. The test set was not used for selection.

| Model | 5-fold CV ROC-AUC (train) |
|-------|---------------------------|
| Logistic Regression | 0.8463 |
| Random Forest | 0.8455 |

The margin is very small (0.0008). LR was selected because it had the higher CV score.

---

## 📈 Test-Set Performance (1,407 customers)

Each row below is one model, evaluated once on the test set.

| Metric | Logistic Regression (selected) | Random Forest |
|--------|-------------------------------|---------------|
| Accuracy | 73.35% | 76.69% |
| Precision | 49.91% | 54.83% |
| Recall | 77.81% | 69.79% |
| F1 score | 0.6082 | 0.6141 |
| ROC-AUC | 0.8310 | 0.8344 |

The selected model catches more churners (higher recall) at the cost of more false alarms. Accuracy alone is misleading here because 73% of customers do not churn.

---

## 🖥️ Streamlit App

`app/app.py` takes one customer's details and returns:

- **Churn probability** and **retention probability**
- **Risk level:** LOW (≤ 40%), MEDIUM (40–70%), HIGH (> 70%)
- **Recommended actions** that match the risk level
- A probability bar chart and a customer summary

The model's own binary decision uses a 50% threshold. The MEDIUM message shows that decision, so a MEDIUM badge can appear next to a "Stay" decision.

---

## 🧪 Tests

```bash
pytest
```

- `test_tenure_group`: checks the tenure boundaries at 0, 12, 13, 48, 49, and 72
- `test_single_row_encoding_parity`: checks that one-customer inference (the app path) produces the same feature vector and scaled values as a batch built with the same functions. It is skipped if `data/raw/telco_churn.csv` is not present.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv          ← download from Kaggle (not tracked)
│   └── processed/                   ← generated (not tracked)
│       ├── telco_churn_clean.csv
│       ├── telco_churn_encoded.csv
│       └── telco_churn_engineered.csv
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb    ← EDA, feature engineering, modeling
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                ← load, clean, encode, save
│   ├── features.py                  ← feature engineering and inference encoding (used by the app)
│   └── evaluate.py                  ← metrics and charts
│
├── app/
│   └── app.py                       ← Streamlit web app
│
├── models/
│   ├── best_model.pkl               ← selected model
│   ├── scaler.pkl                   ← fitted StandardScaler
│   ├── feature_columns.pkl          ← the 36 model input columns, in order
│   └── model_info.json              ← model name, metrics, high-value threshold
│
├── outputs/                         ← charts saved by the notebook
│
├── tests/
│   └── test_features.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 How to Run Locally

**Requirements:** Python 3.11 or newer (pandas 3.0.3 requires it).

### 1. Clone the repository
```bash
git clone https://github.com/Venkatavishnuvardhanthota/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the IBM Telco CSV from the Kaggle link above and save it as:

`data/raw/telco_churn.csv`

### 4. Run the notebook
Run the notebook from inside the `notebooks/` folder. Its paths (`../data`, `../models`, `../outputs`) are relative to that folder.

```bash
cd notebooks
jupyter notebook 01_eda_and_modeling.ipynb
# Run All cells
```

Running the notebook regenerates `data/processed/`, `outputs/`, and `models/`.

### 5. Run the Streamlit app
From the project root:

```bash
streamlit run app/app.py
```

Then open http://localhost:8501

---

## 🌐 Live Demo

🚀 **[Try the app](https://customer-churn-prediction-0001.streamlit.app/)**

---

## ⚠️ Limitations

- **Single public dataset.** The IBM sample is not a live customer database, and its provenance has not been independently verified.
- **Prediction timing.** The model assumes predictions are made for active customers (tenure ≥ 1), when `tenure` and cumulative `TotalCharges` are known. Confirm this fits the intended use.
- **Random split, not time-based.** The test set does not measure performance on future customers.
- **Probabilities are not calibrated.** Training on SMOTE-balanced data pushes probabilities above the true 26.6% base rate. Use the probabilities for ranking, not as exact likelihoods.
- **Decision threshold.** The binary prediction uses 0.50. It was not tuned to a retention budget or customer lifetime value.
- **No subgroup analysis.** Performance across customer segments (for example, senior citizens or contract types) has not been checked.
- **Redundant columns.** `InternetService_No` and the six `*_No internet service` columns carry the same information. They are kept for consistency. Tree and regularized models handle this, but correlation charts show them as separate signals.
- **Correlation is not causation.** The EDA correlations describe association only.

---

## 📄 License

This project is open source under the MIT License. See `LICENSE`.

---

**Thota Venkata Vishnu Vardhan**

📧 Email: [venkatavishnuvardhanthota@gmail.com](mailto:venkatavishnuvardhanthota@gmail.com)

💻 GitHub: https://github.com/Venkatavishnuvardhanthota

🔗 LinkedIn: https://www.linkedin.com/in/venkata-vishnu-vardhan-thota/