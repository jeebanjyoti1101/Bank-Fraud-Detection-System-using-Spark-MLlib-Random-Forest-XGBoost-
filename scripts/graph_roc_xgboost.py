"""
ROC Curve for XGBoost Model Only
Displays a clean, focused ROC curve with AUC score
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
import json
import xgboost as xgb

print("=" * 70)
print("📈 ROC Curve - XGBoost Model")
print("=" * 70)

# Load XGBoost model
print("\n🤖 Loading XGBoost model...")
try:
    xgb_model = joblib.load('models/xgboost_model.pkl')
    print("   ✅ XGBoost model loaded")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    exit(1)

# Load scaler and metadata
scaler = joblib.load('models/scaler.pkl')
with open('models/advanced_metadata.json', 'r') as f:
    metadata = json.load(f)
    feature_cols = metadata.get("feature_names", [])

print("   ✅ Scaler and metadata loaded")

# Load and prepare data
print("\n📊 Loading dataset...")
df = pd.read_csv('data/Fraud.csv')
df.columns = df.columns.str.strip()

# Ensure required columns exist
for col in ['oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig', 'amount', 'type']:
    if col not in df.columns:
        df[col] = 0

for col in ['amount', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"   ✅ Loaded {len(df):,} transactions")

# Feature engineering function
def engineer_features(tx):
    a, ob, nb, t = tx["amount"], tx["oldbalanceOrg"], tx["newbalanceOrig"], tx["type"]
    
    features = {
        "amount": a, "oldbalanceOrg": ob, "newbalanceOrig": nb,
        "amount_log": np.log1p(a), "amount_sqrt": np.sqrt(a),
        "balance_change": ob - nb, "amount_to_balance_ratio": a / ob if ob > 0 else 0,
    }
    
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

# Prepare test data
print("\n🔧 Preparing test data (5,000 samples)...")
df_sample = df.sample(n=min(5000, len(df)), random_state=42)

X_test = pd.concat([engineer_features({
    "amount": r["amount"],
    "oldbalanceOrg": r["oldbalanceOrg"],
    "newbalanceOrig": r["newbalanceOrig"],
    "oldbalanceDest": r.get("oldbalanceDest", 0),
    "newbalanceDest": r.get("newbalanceDest", 0),
    "type": r["type"]
}) for _, r in df_sample.iterrows()], ignore_index=True)

y_test = df_sample['isFraud'].values
X_scaled = scaler.transform(X_test)

print(f"   ✅ Test data prepared with {X_test.shape[1]} features")

# Generate ROC curve
print("\n📈 Generating ROC curve for XGBoost...")

plt.figure(figsize=(10, 8))

try:
    # Get predictions
    y_pred_proba = xgb_model.predict_proba(X_scaled)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, linewidth=4, label=f'XGBoost (AUC = {roc_auc:.4f})', 
             color='#2E86AB', marker='^', markevery=50, markersize=6)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5000)')
    
    # Styling
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curve - XGBoost Model', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add text box with metrics
    textstr = f'AUC Score: {roc_auc:.4f}\nSamples: {len(y_test):,}\nFraud Cases: {sum(y_test):,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.98, 0.02, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    
    print(f"   ✅ XGBoost AUC: {roc_auc:.4f}")
    print(f"   ✅ Test samples: {len(y_test):,}")
    print(f"   ✅ Fraud cases: {sum(y_test):,} ({sum(y_test)/len(y_test)*100:.2f}%)")
    
except Exception as e:
    print(f"   ❌ Error generating ROC curve: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ ROC CURVE GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\n👁️  Displaying graph... (close window to exit)")
plt.show()

print("\n🎉 Done! Thank you for using the fraud detection visualizer.")
print("=" * 70)
