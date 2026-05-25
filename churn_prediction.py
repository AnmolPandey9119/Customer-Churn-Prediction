"""
📉 Customer Churn Prediction
==============================
Predicts whether a telecom customer will churn using
Random Forest + XGBoost ensemble on Telco-style features.
Author: Anmol Pandey (AnmolPandey9119)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier


# ─── 1. GENERATE SYNTHETIC TELCO DATASET ────────────────────────────
def generate_telco_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic Telco-style churn dataset.
    Features mirror IBM Telco Customer Churn dataset distributions.
    """
    rng = np.random.default_rng(seed)

    tenure          = rng.integers(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n).round(2)
    total_charges   = (monthly_charges * tenure + rng.normal(0, 50, n)).clip(0).round(2)

    contract    = rng.choice(["Month-to-month", "One year", "Two year"], n,
                             p=[0.55, 0.24, 0.21])
    internet    = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment     = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Binary features
    senior         = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner        = rng.choice([0, 1], n, p=[0.52, 0.48])
    dependents     = rng.choice([0, 1], n, p=[0.70, 0.30])
    phone_service  = rng.choice([0, 1], n, p=[0.10, 0.90])
    paperless      = rng.choice([0, 1], n, p=[0.40, 0.60])
    online_security= rng.choice([0, 1], n, p=[0.50, 0.50])
    tech_support   = rng.choice([0, 1], n, p=[0.50, 0.50])

    # Churn probability driven by domain knowledge
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.15 * (internet == "Fiber optic")
        + 0.10 * (monthly_charges > 80) / 40 * (monthly_charges - 80)
        - 0.15 * (tenure > 24)
        - 0.10 * (online_security == 1)
        - 0.08 * (tech_support == 1)
        + 0.08 * (payment == "Electronic check")
        + 0.05 * (senior == 1)
        - 0.05 * (partner == 1)
    ).clip(0.02, 0.80)

    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "PaperlessBilling": paperless,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "Churn": churn,
    })


# ─── 2. LOAD & INSPECT ──────────────────────────────────────────────
print("=" * 60)
print("📉  CUSTOMER CHURN PREDICTION")
print("=" * 60)

df = generate_telco_dataset(5000)

print(f"\n[DATA] Shape            : {df.shape}")
print(f"[DATA] Churned customers : {df['Churn'].sum()} ({df['Churn'].mean()*100:.1f}%)")
print(f"[DATA] Retained          : {(~df['Churn'].astype(bool)).sum()} ({(1-df['Churn'].mean())*100:.1f}%)")

print("\n[DATA] Sample statistics:")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2).to_string())


# ─── 3. PREPROCESSING ───────────────────────────────────────────────
# Encode categorical columns
cat_cols = ["Contract", "InternetService", "PaymentMethod"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

feature_cols = [c for c in df.columns if c != "Churn"]
X = df[feature_cols]
y = df["Churn"]

# Scale numeric features for models that benefit from it
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[SPLIT] Train : {len(X_train)}  |  Test : {len(X_test)}")


# ─── 4. MODELS ──────────────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost      ": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=42, verbosity=0
    ),
}

print("\n" + "─" * 65)
print(f"{'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
      f"{'F1':>8} {'ROC-AUC':>9}")
print("─" * 65)

best_model, best_auc, best_name = None, 0, ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    y_proba  = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    print(f"{name:<20} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {auc:>9.4f}")

    if auc > best_auc:
        best_auc, best_model, best_name = auc, model, name.strip()

print("─" * 65)
print(f"\n🏆  Best Model: {best_name}  (ROC-AUC = {best_auc:.4f})\n")


# ─── 5. DETAILED EVALUATION ─────────────────────────────────────────
y_pred_best = best_model.predict(X_test)
print("─" * 60)
print(f"DETAILED REPORT — {best_name}")
print("─" * 60)
print(classification_report(y_test, y_pred_best,
                             target_names=["Retained", "Churned"]))

cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:")
print(f"  True Negatives  (Retained → Retained): {tn}")
print(f"  False Positives (Retained → Churned) : {fp}")
print(f"  False Negatives (Churned  → Retained): {fn}  ← missed churners (costly!)")
print(f"  True Positives  (Churned  → Churned) : {tp}")


# ─── 6. FEATURE IMPORTANCE ──────────────────────────────────────────
print("\n" + "─" * 60)
print("TOP 10 CHURN DRIVERS (Feature Importance)")
print("─" * 60)

importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
for feat, imp in feat_imp.head(10).items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<20} {imp:.4f}  {bar}")


# ─── 7. CROSS-VALIDATION ────────────────────────────────────────────
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvs  = cross_val_score(best_model, X_scaled, y, cv=cv, scoring="roc_auc")
print(f"\n5-Fold CV ROC-AUC : {[round(s,4) for s in cvs]}")
print(f"Mean CV ROC-AUC   : {cvs.mean():.4f} ± {cvs.std():.4f}")


# ─── 8. LIVE PREDICTION DEMO ────────────────────────────────────────
print("\n" + "─" * 60)
print("LIVE PREDICTION DEMO")
print("─" * 60)

sample_customers = pd.DataFrame([
    # tenure, MonthlyCharges, TotalCharges, Senior, Partner, Dependents,
    # Phone, Paperless, OnlineSec, TechSupport, Contract(0=M2M,1=1yr,2=2yr),
    # Internet(0=DSL,1=Fiber,2=No), Payment(0=Bank,1=CC,2=Echeck,3=Mail)
    [2, 105, 210, 1, 0, 0, 1, 1, 0, 0, 0, 1, 2],   # High risk
    [60, 35, 2100, 0, 1, 1, 1, 0, 1, 1, 2, 0, 1],   # Low risk
    [12, 75, 900, 0, 0, 0, 1, 1, 0, 0, 0, 1, 2],    # Medium risk
], columns=feature_cols)

sample_scaled = pd.DataFrame(scaler.transform(sample_customers), columns=feature_cols)
preds  = best_model.predict(sample_scaled)
probas = best_model.predict_proba(sample_scaled)[:, 1]

labels = ["HIGH RISK", "LOW RISK", "MEDIUM RISK"]
for i, (pred, prob) in enumerate(zip(preds, probas)):
    status = "🚨 WILL CHURN" if pred == 1 else "✅ WILL STAY "
    print(f"  Customer {i+1} ({labels[i]:<11}): {status}  |  Churn probability: {prob:.1%}")

print("\n" + "=" * 60)
print("✅  Churn prediction model complete!")
print("=" * 60)
