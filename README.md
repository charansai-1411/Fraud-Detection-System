# 🔍 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green?style=flat-square)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9769-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> An end-to-end machine learning system for detecting fraudulent credit card transactions — with XGBoost, SMOTE-based class balancing, SHAP explainability, and an interactive Streamlit dashboard for real-time prediction.

---

## 📌 Problem Statement

Credit card fraud causes billions in losses annually. Traditional rule-based systems miss novel fraud patterns and generate excessive false alarms. This project builds a **data-driven fraud detection pipeline** that:

- Handles extreme class imbalance (fraud rate: **0.17%**)
- Achieves high recall to minimize missed fraud cases
- Provides **per-transaction explainability** for regulatory auditability
- Allows business-level threshold tuning (precision vs recall tradeoff)

---

## 🎯 Results

| Metric | Default Threshold (0.5) | Tuned Threshold (0.89) |
|--------|------------------------|------------------------|
| ROC-AUC | **0.9769** | 0.9769 |
| Precision (Fraud) | 0.61 | **0.87** |
| Recall (Fraud) | 0.87 | 0.83 |
| F1 Score | 0.71 | **0.85** |
| False Positives | 55 | **12** |
| Avg Precision Score | 0.8619 | 0.8619 |

> Threshold tuning reduced false positives by **78%** (55 → 12) while retaining 83% fraud recall — critical for minimizing customer friction in a production banking system.

---

## 🏗️ System Architecture

```
Raw Transaction Data (284,807 rows)
         │
         ▼
   EDA & Preprocessing
   ├── Amount / Time scaling (StandardScaler)
   └── Train-Test Split (80/20, stratified)
         │
         ▼
   SMOTE Oversampling (Training only)
   └── Balanced: 227,451 fraud vs 227,451 legit
         │
         ▼
   XGBoost Classifier
   ├── n_estimators=200, max_depth=6, lr=0.1
   └── eval_metric=AUC
         │
         ▼
   Threshold Tuning (0.5 → 0.89)
         │
         ▼
   SHAP Explainability
   ├── Global: Feature importance + Beeswarm
   └── Local: Waterfall per transaction
         │
         ▼
   Streamlit Dashboard
   └── CSV upload → predictions → SHAP explanations
```

---

## 📊 Key Visualizations

### SHAP Beeswarm — Global Feature Importance
> V14 has 2x the predictive impact of any other feature. Low V14 values are the strongest fraud signal.

![SHAP Beeswarm](assets/shap_beeswarm.png)

### SHAP Waterfall — Single Transaction Explanation
> V14 = -6.17 pushed this transaction +6.15 toward fraud — the dominant driver.

![SHAP Waterfall](assets/shap_waterfall.png)

### Precision-Recall Curve
> AP = 0.8619 on a 0.17% fraud rate dataset — strong performance under extreme imbalance.

![PR Curve](assets/pr_curve.png)

### Confusion Matrix (Tuned Threshold)
> 81 fraud caught, only 12 false alarms out of 56,962 transactions.

![Confusion Matrix](assets/confusion_matrix.png)

---

## 🧠 Key Technical Decisions

**Why SMOTE over class_weight?**
SMOTE generates synthetic minority samples by interpolating between existing fraud cases, giving the model richer fraud patterns to learn from. Applied strictly on training data to prevent data leakage.

**Why XGBoost over Random Forest?**
Sequential boosting corrects residual errors iteratively — better suited for imbalanced tabular data with PCA-transformed features.

**Why threshold 0.89?**
In banking, a false positive (blocking a legitimate customer) damages trust and costs ops time. At 0.89, precision jumps from 61% → 87% while sacrificing only 4 fraud catches. This is a **business decision**, not just a model decision.

**Why SHAP?**
Regulatory compliance in financial services (Basel III, SR 11-7) increasingly requires model explainability. SHAP provides auditable, per-prediction justifications.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install xgboost shap imbalanced-learn scikit-learn streamlit pandas numpy matplotlib seaborn
```

### Run the Notebook
```bash
# Open in Google Colab or Jupyter
fraud_detection.ipynb
```

### Run the Streamlit Dashboard
```bash
git clone https://github.com/yourusername/fraud-detection-system
cd fraud-detection-system
streamlit run app.py
```

Upload `sample_transactions.csv` (included) to test the dashboard immediately.

---

## 📁 Project Structure

```
fraud-detection-system/
├── fraud_detection.ipynb     # Full ML pipeline notebook
├── app.py                    # Streamlit dashboard
├── fraud_model.pkl           # Trained XGBoost model
├── scaler.pkl                # Fitted StandardScaler
├── sample_transactions.csv   # Sample test data for demo
├── assets/
│   ├── shap_beeswarm.png
│   ├── shap_waterfall.png
│   ├── pr_curve.png
│   └── confusion_matrix.png
└── README.md
```

---

## 📦 Dataset

**Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions over 2 days (September 2013, European cardholders)
- 492 fraud cases (0.172% of all transactions)
- Features V1–V28: PCA-transformed behavioral features (confidential)
- Features: Time, Amount (original, scaled in preprocessing)

---

## 🔬 Model Pipeline

```python
# Core pipeline summary
1. Load & EDA          → Visualize class imbalance, amount distributions
2. Preprocessing       → Scale Amount/Time, train-test split (stratified)
3. SMOTE               → Balance training set (227k → 227k each class)
4. XGBoost Training    → 200 estimators, AUC eval metric
5. Evaluation          → ROC-AUC, PR curve, confusion matrix
6. Threshold Tuning    → Maximize F1, optimize for business context
7. SHAP                → Global beeswarm + local waterfall explanations
8. Streamlit App       → Interactive CSV upload + prediction dashboard
```

---

## 💡 Business Impact

If deployed on a system processing 1M transactions/day:

- At **0.89 threshold**: ~1,720 fraud cases flagged daily with only ~210 false alarms
- Ops team investigates **1 false alarm per 8 real frauds** — manageable workload
- Each missed fraud averages $150 loss → catching 83% saves **~$214K/day** per million transactions

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| XGBoost | Gradient boosted classifier |
| imbalanced-learn | SMOTE oversampling |
| SHAP | Model explainability |
| Scikit-learn | Preprocessing, evaluation |
| Streamlit | Interactive dashboard |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualizations |

---

## 👤 Author

**Y. Charan Sai**
BE — Artificial Intelligence & Data Science
Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built as part of UBS recruitment preparation — demonstrating production-grade ML engineering with financial domain context.*
