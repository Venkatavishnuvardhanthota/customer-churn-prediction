# 🔄 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

An end-to-end machine learning project that predicts
whether a telecom customer will leave (churn) the company.
Built with Python, trained using Random Forest,
and deployed as an interactive Streamlit web application.

---

## 🎯 Business Problem

A telecom company loses revenue every time a customer
cancels their subscription. The goal is to:

- **Predict** which customers are likely to churn
- **Identify** key factors that drive churn
- **Enable** proactive retention before customer leaves
- **Prioritize** high-risk customers for retention calls

> Acquiring a new customer costs 5x more than
> retaining an existing one!

---

## 💡 Key Findings from EDA

| Finding | Detail |
|---------|--------|
| Month-to-month churn | 42.7% churn rate |
| Two-year contract churn | Only 2.9% churn rate |
| New customer median tenure | 10 months |
| Loyal customer median tenure | 38 months |
| Churner avg monthly charges | $74.44 |
| Stayer avg monthly charges | $61.27 |

**Most Important Features for Prediction:**
1. `tenure` — longer = safer (correlation: -0.35)
2. `Contract_Two year` — two year = very loyal
3. `MonthlyCharges` — higher = more risk
4. `total_services` — more services = less churn
5. `is_new_customer` — first year = critical!

---

## 🤖 Model Performance

| Metric | Score |
|--------|-------|
| Model | Random Forest |
| AUC Score | 0.8343 |
| Accuracy | ~80% |
| Training samples | 8,260 (after SMOTE) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib/Seaborn | Static visualizations |
| Scikit-learn | ML algorithms and evaluation |
| Imbalanced-learn | SMOTE for class balancing |
| Streamlit | Web app deployment |
| Plotly | Interactive charts in app |

---

## 📁 Project Structure

\`\`\`
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv        ← download from Kaggle
│   └── processed/
│       ├── telco_churn_cleaned.csv
│       └── telco_churn_engineered.csv
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb  ← complete analysis
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py              ← data pipeline
│   └── evaluate.py
│
├── app/
│   └── app.py                     ← Streamlit web app
│
├── models/
│   ├── best_model.pkl             ← trained RF model
│   ├── scaler.pkl                 ← feature scaler
│   ├── feature_columns.pkl        ← column names
│   └── model_info.json            ← model metadata
│
├── outputs/
│   ├── 01_churn_distribution.png
│   ├── 02_churn_by_contract.png
│   ├── 03_tenure_vs_churn.png
│   ├── 04_monthly_charges_churn.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_feature_validation.png
│   └── 07_confusion_matrix.png
│
├── requirements.txt
└── README.md
\`\`\`

---

## 🚀 How to Run Locally

### 1. Clone the repository
\`\`\`bash
git clone https://github.com/Venkatavishnuvardhanthota/customer-churn-prediction.git
cd customer-churn-prediction
\`\`\`

### 2. Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Download the dataset
Download from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Save as: `data/raw/telco_churn.csv`

### 4. Run the notebook
\`\`\`bash
# Open VS Code
code .
# Open notebooks/01_eda_and_modeling.ipynb
# Run all cells
\`\`\`

### 5. Run the Streamlit app
\`\`\`bash
streamlit run app/app.py
\`\`\`

Open browser at: http://localhost:8501

---

## 📊 Analysis Workflow

\`\`\`
Raw Data (7043 customers, 21 features)
         ↓
Quick EDA (class imbalance, missing values)
         ↓
Data Cleaning (fix types, encode target)
         ↓
Deep EDA (3 hypotheses confirmed!)
         ↓
Feature Engineering (5 new features)
         ↓
Train/Test Split (80/20, stratified)
         ↓
SMOTE (balance training data)
         ↓
Model Training (LR + Random Forest)
         ↓
Evaluation (RF wins with AUC 0.8343)
         ↓
Streamlit App (anyone can predict!)
\`\`\`

---

## 🌐 Live Demo

[Coming soon - Streamlit Cloud deployment]

---

## 👤 Author

**Venkata Vishnu Vardhan Thota**
- GitHub: [@Venkatavishnuvardhanthota](https://github.com/Venkatavishnuvardhanthota)

---

## 📄 License

This project is open source under the MIT License.