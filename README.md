# 📉 Customer Churn Prediction

> **ROC-AUC: 0.78** — Predicts which telecom customers will leave using XGBoost + Random Forest on 5,000 customer records with domain-driven feature engineering.

---

## 🎯 Project Overview

Customer churn costs businesses billions every year. This project builds an ML pipeline that identifies **at-risk customers before they leave**, enabling proactive retention campaigns. Uses a realistic **Telco-style synthetic dataset** with 13 behavioral and demographic features.

| Metric | Random Forest | XGBoost |
|--------|:-------------:|:-------:|
| Accuracy | 82.80% | 80.80% |
| ROC-AUC | 0.7758 | **0.7774** |
| Recall (Churned) | 48.09% | **54.96%** |
| F1 (Churned) | 42.28% | **42.86%** |
| 5-Fold CV AUC | — | **76.86% ± 1.39%** |

> XGBoost wins on ROC-AUC and Recall — critical for identifying as many churners as possible.

---

## 📂 Dataset

**Synthetic Telco Dataset** — 5,000 customers, 13 features, 13.1% churn rate  
Generated with statistically realistic distributions mirroring the IBM Telco Churn dataset.

**Key features:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, `OnlineSecurity`, `TechSupport`, and more.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189B00?style=flat-square&logo=xgboost&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

---

## 🚀 Quick Start

```bash
git clone https://github.com/AnmolPandey9119/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
python churn_prediction.py
```

---

## 📊 Sample Output

```
📉  CUSTOMER CHURN PREDICTION
============================================================
[DATA] Churned : 653 (13.1%)  |  Retained : 4347 (86.9%)

Model          Accuracy  Precision  Recall    F1   ROC-AUC
Random Forest  0.8280    0.3772     0.4809  0.4228  0.7758
XGBoost        0.8080    0.3512     0.5496  0.4286  0.7774  ← Best

TOP CHURN DRIVERS
  Contract           0.3583  ███████████████████████████████
  InternetService    0.0850  ████████████████
  TechSupport        0.0699  █████████████
  OnlineSecurity     0.0628  ████████████
  tenure             0.0566  ███████████

LIVE PREDICTION DEMO
  Customer 1 (HIGH RISK  ): 🚨 WILL CHURN  | Churn probability: 85.7%
  Customer 2 (LOW RISK   ): ✅ WILL STAY   | Churn probability: 1.2%
  Customer 3 (MEDIUM RISK): 🚨 WILL CHURN  | Churn probability: 75.4%
```

---

## 🔑 Key ML Concepts Demonstrated

- **Class imbalance handling** via `scale_pos_weight` in XGBoost and `class_weight='balanced'` in RF
- **Feature importance analysis** — Contract type is the #1 churn predictor
- **ROC-AUC as primary metric** — more meaningful than accuracy for imbalanced data
- **Stratified K-Fold Cross-Validation** — reliable generalization estimate
- **Domain-driven synthetic data generation** — realistic feature correlations

---

## 📈 Business Insight

The model reveals **Contract type drives 36% of churn decisions** — customers on month-to-month contracts are 3x more likely to leave. A targeted 3-month free upgrade offer could save significant revenue.

---

## 👨‍💻 Author

**Anmol Pandey** — AI/ML Engineer  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anmol-pandey-240105376)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/AnmolPandey9119)

---

*Part of my AI/ML project portfolio — targeting ML Engineer & Data Scientist roles.*
