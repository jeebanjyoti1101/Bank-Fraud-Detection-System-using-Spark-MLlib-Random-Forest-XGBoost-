"""
Graph 2: Combined ROC Curve
Shows all models on one plot for comparison
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
import json

print("📈 Loading models and data...")

# Load models
models = {}
model_files = {
    'rf': 'models/rf_model.pkl',
    'xgb': 'models/xgboost_model.pkl',
    'lgb': 'models/lightgbm_model.pkl',
    'cat': 'models/catboost_model.pkl'
}

for name, path in model_files.items():
    try:
        models[name] = joblib.load(path)
        print(f"✅ Loaded {name.upper()}")
    except Exception as e:
        print(f"⚠️ Could not load {name.upper()}: {e}")

scaler = joblib.load('models/scaler.pkl')
with open('models/advanced_metadata.json', 'r') as f:
    metadata = json.load(f)
    feature_cols = metadata.get("feature_names", [])

# Load and prepare data
print("\n📊 Loading dataset...")
df = pd.read_csv('data/Fraud.csv')
df.columns = df.columns.str.strip()

for col in ['oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig', 'amount', 'type']:
    if col not in df.columns:
        df[col] = 0

for col in ['amount', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Feature engineering (same as graph_01)
def engineer_features(tx):
    a, ob, nb, obd, nbd = [tx[k] for k in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]]
    t = tx["type"]
    
    features = {
        "amount": a, "oldbalanceOrg": ob, "newbalanceOrig": nb,
        "amount_log": np.log1p(a), "amount_sqrt": np.sqrt(a),
        "balance_change": ob - nb, "amount_to_balance_ratio": a / ob if ob > 0 else 0,
    }
    
    if t == "CASH_IN":
        expected_new_increase = ob + a
        error_if_increase = abs(expected_new_increase - nb)
        features["balance_error"] = error_if_increase
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
    
    if t == "CASH_IN":
        features["exact_balance_match"] = int(nb == ob + a)
        features["almost_exact_match"] = int(abs(nb - (ob + a)) < 1)
    else:
        features["exact_balance_match"] = int(nb == ob - a)
        features["almost_exact_match"] = int(abs(nb - (ob - a)) < 1)
    
    features.update({
        "suspicious_zero_transaction": int(a == 0 and ob > 0),
        "balance_mismatch": int(features["balance_error"] > a * 0.01),
        "amount_quintile": pd.qcut([a], q=5, labels=False, duplicates='drop')[0] if a > 0 else 0,
        "amount_decile": pd.qcut([a], q=10, labels=False, duplicates='drop')[0] if a > 0 else 0,
        "round_amount": int(a % 1000 == 0 and a > 0),
        "round_large_amount": int(a % 10000 == 0 and a > 10000),
        "round_medium_amount": int(a % 5000 == 0 and a > 5000),
        "odd_amount": int(a % 1 != 0),
        "amount_outlier_99": int(a > 10000000),
        "amount_outlier_95": int(a > 500000),
        "amount_outlier_90": int(a > 250000),
        "small_amount": int(a < 100),
        "transfer_large": int(t == "TRANSFER" and a > 200000),
        "transfer_medium": int(t == "TRANSFER" and 50000 < a <= 200000),
        "cashout_large": int(t == "CASH_OUT" and a > 200000),
        "cashout_medium": int(t == "CASH_OUT" and 50000 < a <= 200000),
        "payment_large": int(t == "PAYMENT" and a > 100000),
        "transfer_or_cashout": int(t in ["TRANSFER", "CASH_OUT"]),
        "high_risk_type": int(t in ["TRANSFER", "CASH_OUT"]),
        "low_risk_type": int(t in ["PAYMENT", "CASH_IN"]),
        "type_risk_score": {"TRANSFER": 0.8, "CASH_OUT": 0.8, "PAYMENT": 0.2, "CASH_IN": 0.2, "DEBIT": 0.4}.get(t, 0.3),
        "risky_transaction": int(t in ["TRANSFER", "CASH_OUT"] and a > 100000),
        "balance_zscore": (ob - 50000) / 100000,
        "amount_zscore": (a - 50000) / 100000,
    })
    
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
    
    df_features = pd.DataFrame([features])
    if feature_cols:
        for col in feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0
        df_features = df_features[feature_cols]
    
    return df_features

# Prepare test data
print("🔧 Preparing test data...")
sample_size = min(5000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

all_features_list = []
for _, row in df_sample.iterrows():
    tx_data = {
        "amount": row["amount"],
        "oldbalanceOrg": row["oldbalanceOrg"],
        "newbalanceOrig": row["newbalanceOrig"],
        "oldbalanceDest": row.get("oldbalanceDest", 0),
        "newbalanceDest": row.get("newbalanceDest", 0),
        "type": row["type"]
    }
    all_features_list.append(engineer_features(tx_data))

X_test = pd.concat(all_features_list, ignore_index=True)
y_test = df_sample['isFraud'].values
print(f"✅ Test data prepared with {X_test.shape[1]} features.")

# Generate combined ROC curve
print("\n📈 Generating combined ROC curve...")

plt.figure(figsize=(12, 8))
X_scaled = scaler.transform(X_test)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
for idx, (name, model) in enumerate(models.items()):
    try:
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=3, label=f'{name.upper()} (AUC = {roc_auc:.3f})',
                color=colors[idx % len(colors)])
        print(f"  ✅ {name.upper()}: AUC = {roc_auc:.4f}")
    except Exception as e:
        print(f"  ⚠️ Error with {name.upper()}: {e}")

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('Model Performance Comparison - ROC Curves', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)

print("\n✅ Displaying combined ROC curve...")
plt.show()
