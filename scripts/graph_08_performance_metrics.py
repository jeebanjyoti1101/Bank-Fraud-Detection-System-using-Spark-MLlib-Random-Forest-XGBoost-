"""
Graph 8: Model Performance Metrics Comparison
Shows bar charts comparing different performance metrics across models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

print("📈 Loading models and data...")

models = {}
for name, path in [('rf', 'models/rf_model.pkl'), ('xgb', 'models/xgboost_model.pkl'), 
                   ('lgb', 'models/lightgbm_model.pkl'), ('cat', 'models/catboost_model.pkl')]:
    try:
        models[name] = joblib.load(path)
        print(f"✅ Loaded {name.upper()}")
    except:
        pass

scaler = joblib.load('models/scaler.pkl')
with open('models/advanced_metadata.json', 'r') as f:
    feature_cols = json.load(f).get("feature_names", [])

df = pd.read_csv('data/Fraud.csv')
df.columns = df.columns.str.strip()
for col in ['oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig', 'amount', 'type']:
    if col not in df.columns:
        df[col] = 0
for col in ['amount', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

def engineer_features(tx):
    a, ob, nb, t = tx["amount"], tx["oldbalanceOrg"], tx["newbalanceOrig"], tx["type"]
    features = {"amount": a, "oldbalanceOrg": ob, "newbalanceOrig": nb, "amount_log": np.log1p(a),
                "amount_sqrt": np.sqrt(a), "balance_change": ob - nb, "amount_to_balance_ratio": a / ob if ob > 0 else 0}
    if t == "CASH_IN":
        features["balance_error"] = abs((ob + a) - nb)
        if nb < ob:
            features["balance_error"] += (ob - nb) * 100
        elif nb > ob and (nb - ob) > a + 0.01:
            features["balance_error"] += ((nb - ob) - a) * 100
    else:
        features["balance_error"] = abs(ob - a - nb)
        if nb > ob:
            features["balance_error"] += (nb - ob) * 100
    features["balance_error_ratio"] = features["balance_error"] / a if a > 0 else 0
    features["has_balance_error"] = int(features["balance_error"] > 0.01)
    features["large_balance_error"] = int(features["balance_error"] > 1000)
    features["zero_balance_before"] = int(ob == 0)
    features["zero_balance_after"] = int(nb == 0)
    features["complete_drain"] = int(nb == 0 and ob > 0)
    features["partial_drain"] = int(features["balance_change"] > 0 and nb > 0)
    drain_pct = features["balance_change"] / ob if ob > 0 else 0
    features["high_drain_ratio"] = int(drain_pct > 0.9)
    features["medium_drain_ratio"] = int(0.5 < drain_pct <= 0.9)
    features["low_drain_ratio"] = int(0 < drain_pct <= 0.5)
    features["near_complete_drain"] = int(drain_pct > 0.95)
    features["exact_balance_match"] = int(nb == (ob + a if t == "CASH_IN" else ob - a))
    features["almost_exact_match"] = int(abs(nb - (ob + a if t == "CASH_IN" else ob - a)) < 1)
    features["suspicious_zero_transaction"] = int(a == 0 and ob > 0)
    features["balance_mismatch"] = int(features["balance_error"] > a * 0.01)
    features["amount_quintile"] = pd.qcut([a], q=5, labels=False, duplicates='drop')[0] if a > 0 else 0
    features["amount_decile"] = pd.qcut([a], q=10, labels=False, duplicates='drop')[0] if a > 0 else 0
    features["round_amount"] = int(a % 1000 == 0 and a > 0)
    features["round_large_amount"] = int(a % 10000 == 0 and a > 10000)
    features["round_medium_amount"] = int(a % 5000 == 0 and a > 5000)
    features["odd_amount"] = int(a % 1 != 0)
    features["amount_outlier_99"] = int(a > 10000000)
    features["amount_outlier_95"] = int(a > 500000)
    features["amount_outlier_90"] = int(a > 250000)
    features["small_amount"] = int(a < 100)
    features["transfer_large"] = int(t == "TRANSFER" and a > 200000)
    features["transfer_medium"] = int(t == "TRANSFER" and 50000 < a <= 200000)
    features["cashout_large"] = int(t == "CASH_OUT" and a > 200000)
    features["cashout_medium"] = int(t == "CASH_OUT" and 50000 < a <= 200000)
    features["payment_large"] = int(t == "PAYMENT" and a > 100000)
    features["transfer_or_cashout"] = int(t in ["TRANSFER", "CASH_OUT"])
    features["high_risk_type"] = int(t in ["TRANSFER", "CASH_OUT"])
    features["low_risk_type"] = int(t in ["PAYMENT", "CASH_IN"])
    features["type_risk_score"] = {"TRANSFER": 0.8, "CASH_OUT": 0.8, "PAYMENT": 0.2, "CASH_IN": 0.2, "DEBIT": 0.4}.get(t, 0.3)
    features["risky_transaction"] = int(t in ["TRANSFER", "CASH_OUT"] and a > 100000)
    features["balance_zscore"] = (ob - 50000) / 100000
    features["amount_zscore"] = (a - 50000) / 100000
    features["balance_zscore_outlier"] = int(abs(features["balance_zscore"]) > 2)
    features["amount_zscore_outlier"] = int(abs(features["amount_zscore"]) > 2)
    features["balance_iqr_outlier"] = int(ob > 500000 or ob < 1000)
    features["amount_iqr_outlier"] = int(a > 500000 or a < 100)
    features["extreme_outlier"] = int(features["balance_zscore_outlier"] or features["amount_zscore_outlier"])
    features["balance_percentile"] = min(ob / 1000000, 1.0)
    features["amount_percentile"] = min(a / 1000000, 1.0)
    features["percentile_diff"] = abs(features["balance_percentile"] - features["amount_percentile"])
    features["new_to_old_balance_ratio"] = nb / ob if ob > 0 else 0
    features["amount_balance_product"] = a * ob
    features["amount_balance_product_log"] = np.log1p(features["amount_balance_product"])
    features["balance_change_pct"] = (ob - nb) / ob if ob > 0 else 0
    features["extreme_change"] = int(abs(features["balance_change_pct"]) > 0.95)
    features["type_encoded"] = {"TRANSFER": 1, "CASH_OUT": 2, "PAYMENT": 0, "CASH_IN": 3, "DEBIT": 4}.get(t, 0)
    features["dataset_source_encoded"] = 0
    df_f = pd.DataFrame([features])
    if feature_cols:
        for col in feature_cols:
            if col not in df_f.columns:
                df_f[col] = 0
        df_f = df_f[feature_cols]
    return df_f

print("🔧 Preparing test data...")
df_sample = df.sample(n=min(5000, len(df)), random_state=42)
X_test = pd.concat([engineer_features({"amount": r["amount"], "oldbalanceOrg": r["oldbalanceOrg"],
                                       "newbalanceOrig": r["newbalanceOrig"], "oldbalanceDest": r.get("oldbalanceDest", 0),
                                       "newbalanceDest": r.get("newbalanceDest", 0), "type": r["type"]})
                    for _, r in df_sample.iterrows()], ignore_index=True)
y_test = df_sample['isFraud'].values
X_scaled = scaler.transform(X_test)
print(f"✅ Test data prepared")

print("\n📈 Calculating metrics...")
metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
model_names = []

for name, model in models.items():
    try:
        y_pred = model.predict(X_scaled)
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['Precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['Recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['F1-Score'].append(f1_score(y_test, y_pred, zero_division=0))
        model_names.append(name.upper())
        print(f"  ✅ {name.upper()} metrics calculated")
    except Exception as e:
        print(f"  ⚠️ Error with {name.upper()}: {e}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for idx, (metric_name, values) in enumerate(metrics.items()):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(range(len(model_names)), values, color=colors)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
print("\n✅ Displaying performance metrics...")
plt.show()
