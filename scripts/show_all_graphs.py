"""
Master Script: Show All Graphs Sequentially
Close each graph window to automatically show the next one
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🎨 INTERACTIVE FRAUD DETECTION GRAPHS")
print("=" * 70)
print("\n📌 Instructions:")
print("   - Each graph will display in a window")
print("   - Close the window to see the next graph")
print("   - Press Ctrl+C in terminal to stop anytime")
print("\n" + "=" * 70 + "\n")

# Load models and metadata
print("📦 Loading models and data...")
models = {}
for name, path in [('rf', 'models/rf_model.pkl'), ('xgb', 'models/xgboost_model.pkl'), 
                   ('lgb', 'models/lightgbm_model.pkl'), ('cat', 'models/catboost_model.pkl')]:
    try:
        models[name] = joblib.load(path)
        print(f"   ✅ {name.upper()}")
    except Exception as e:
        print(f"   ⚠️ {name.upper()} failed: {e}")

scaler = joblib.load('models/scaler.pkl')
with open('models/advanced_metadata.json', 'r') as f:
    metadata = json.load(f)
    feature_cols = metadata.get("feature_names", [])

# Load data
df = pd.read_csv('data/Fraud.csv')
df.columns = df.columns.str.strip()
for col in ['oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig', 'amount', 'type']:
    if col not in df.columns:
        df[col] = 0
for col in ['amount', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"   ✅ Loaded {len(df):,} transactions\n")

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

# Prepare test data once
print("🔧 Preparing test data (5,000 samples)...")
df_sample = df.sample(n=min(5000, len(df)), random_state=42)
X_test = pd.concat([engineer_features({"amount": r["amount"], "oldbalanceOrg": r["oldbalanceOrg"],
                                       "newbalanceOrig": r["newbalanceOrig"], "oldbalanceDest": r.get("oldbalanceDest", 0),
                                       "newbalanceDest": r.get("newbalanceDest", 0), "type": r["type"]})
                    for _, r in df_sample.iterrows()], ignore_index=True)
y_test = df_sample['isFraud'].values
X_scaled = scaler.transform(X_test)
print(f"   ✅ Test data ready with {X_test.shape[1]} features\n")

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# ==============================================================================
# GRAPH 1: ROC Curves - Individual Models
# ==============================================================================
print("📊 [1/8] ROC Curves - Individual Models")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, (name, model) in enumerate(models.items()):
    if idx >= 4:
        break
    ax = axes[idx]
    try:
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=3, label=f'ROC (AUC = {roc_auc:.3f})', color='#4ECDC4')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'ROC Curve - {name.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        print(f"   ✅ {name.upper()}: AUC = {roc_auc:.4f}")
    except Exception as e:
        ax.text(0.5, 0.5, f'Error\n{name.upper()}', ha='center', va='center')

plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.25)
fig.suptitle('Graph 1/8: ROC Curves - Individual Models', fontsize=16, fontweight='bold', y=0.985)
print("   👁️  Displaying... (close window to continue)\n")
plt.show()

# ==============================================================================
# GRAPH 2: Combined ROC Curve
# ==============================================================================
print("📊 [2/8] Combined ROC Curve")
plt.figure(figsize=(12, 8))

for idx, (name, model) in enumerate(models.items()):
    try:
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=3, label=f'{name.upper()} (AUC = {roc_auc:.3f})',
                color=colors[idx % len(colors)])
        print(f"   ✅ {name.upper()}: AUC = {roc_auc:.4f}")
    except:
        pass

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.95)
plt.gcf().suptitle('Graph 2/8: Model Performance Comparison - ROC Curves', fontsize=16, fontweight='bold', y=0.985)
print("   👁️  Displaying... (close window to continue)\n")
plt.show()

# ==============================================================================
# GRAPH 3: Precision-Recall Curve
# ==============================================================================
print("📊 [3/8] Precision-Recall Curves")
plt.figure(figsize=(12, 8))

for idx, (name, model) in enumerate(models.items()):
    try:
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, linewidth=3, label=f'{name.upper()} (AP = {avg_precision:.3f})',
                color=colors[idx % len(colors)])
        print(f"   ✅ {name.upper()}: AP = {avg_precision:.4f}")
    except:
        pass

plt.xlabel('Recall', fontsize=14, fontweight='bold')
plt.ylabel('Precision', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)
plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.95)
plt.gcf().suptitle('Graph 3/8: Precision-Recall Curve', fontsize=16, fontweight='bold', y=0.985)
print("   👁️  Displaying... (close window to continue)\n")
plt.show()

# ==============================================================================
# GRAPH 4: Confusion Matrix Heatmap
# ==============================================================================
print("📊 [4/8] Confusion Matrix Heatmap")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, (name, model) in enumerate(models.items()):
    if idx >= 4:
        break
    try:
        y_pred = model.predict(X_scaled)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
        axes[idx].set_title(f'{name.upper()}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual', fontsize=12, fontweight='bold')
        print(f"   ✅ {name.upper()}")
    except:
        axes[idx].text(0.5, 0.5, f'Error', ha='center', va='center')

plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.25)
fig.suptitle('Graph 4/8: Confusion Matrices', fontsize=16, fontweight='bold', y=0.985)
print("   👁️  Displaying... (close window to continue)\n")
plt.show()

# ==============================================================================
# GRAPH 5: Feature Importance
# ==============================================================================
print("📊 [5/8] Feature Importance")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, (name, model) in enumerate(models.items()):
    if idx >= 4:
        break
    ax = axes[idx]
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_score'):
            importance_dict = model.get_score(importance_type='weight')
            importances = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_cols))]
        else:
            importances = np.zeros(len(feature_cols))
        
        indices = np.argsort(importances)[-15:]
        top_features = [feature_cols[i] if i < len(feature_cols) else f'Feature {i}' for i in indices]
        top_importances = [importances[i] for i in indices]
        
        bar_colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax.barh(range(len(top_features)), top_importances, color=bar_colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'{name.upper()}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        print(f"   ✅ {name.upper()}")
    except Exception as e:
        ax.text(0.5, 0.5, f'Error', ha='center', va='center')

plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95, hspace=0.40, wspace=0.25)
fig.suptitle('Graph 5/8: Top 15 Feature Importance', fontsize=16, fontweight='bold', y=0.985)
print("   👁️  Displaying... (close window to continue)\n")
plt.show()

# ==============================================================================
# GRAPH 6: Transaction Type Distribution
# ==============================================================================
print("📊 [6/8] Transaction Type Distribution")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

type_counts = df['type'].value_counts()
plot_colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
bars = ax1.bar(range(len(type_counts)), type_counts.values, color=plot_colors)
ax1.set_xticks(range(len(type_counts)))
ax1.set_xticklabels(type_counts.index, rotation=45, ha='right', fontsize=12)
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Transaction Type Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for i, v in enumerate(type_counts.values):
    ax1.text(i, v + max(type_counts.values) * 0.01, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

if 'isFraud' in df.columns:
    fraud_by_type = df.groupby('type')['isFraud'].agg(['sum', 'count'])
    fraud_by_type['fraud_rate'] = (fraud_by_type['sum'] / fraud_by_type['count'] * 100)
    
    bars = ax2.bar(range(len(fraud_by_type)), fraud_by_type['fraud_rate'].values, color=plot_colors)
    ax2.set_xticks(range(len(fraud_by_type)))
    ax2.set_xticklabels(fraud_by_type.index, rotation=45, ha='right', fontsize=12)
    ax2.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Fraud Rate by Transaction Type', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(fraud_by_type['fraud_rate'].values):
        ax2.text(i, v + max(fraud_by_type['fraud_rate'].values) * 0.01, f'{v:.2f}%', 
                ha='center', fontsize=10, fontweight='bold')
    print(f"   ✅ Distribution and fraud rates calculated")

plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, wspace=0.25)
fig.suptitle('Graph 6/8: Transaction Type Distribution', fontsize=16, fontweight='bold', y=0.985)
print("   👁️  Displaying... (close window to continue)\n")
plt.show()

# ==============================================================================
# GRAPH 7: Amount Distribution
# ==============================================================================
print("📊 [7/8] Amount Distribution")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

amounts = df[df['amount'] > 0]['amount']
axes[0, 0].hist(np.log10(amounts + 1), bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Log10(Amount + 1)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

if 'isFraud' in df.columns:
    fraud_amounts = df[df['isFraud'] == 1]['amount']
    normal_amounts = df[df['isFraud'] == 0]['amount'].sample(min(10000, len(df[df['isFraud'] == 0])))
    axes[0, 1].hist(np.log10(normal_amounts + 1), bins=30, alpha=0.5, label='Normal', color='green', edgecolor='black')
    axes[0, 1].hist(np.log10(fraud_amounts + 1), bins=30, alpha=0.5, label='Fraud', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Log10(Amount + 1)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Fraud vs Normal', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)

types = df['type'].unique()[:5]
data_by_type = [df[df['type'] == t]['amount'].values for t in types]
bp = axes[1, 0].boxplot(data_by_type, labels=types, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(types)))):
    patch.set_facecolor(color)
axes[1, 0].set_xlabel('Transaction Type', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Amount', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Amount by Type', fontsize=14, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=0)
axes[1, 0].set_xticklabels(types, rotation=0, ha='center', fontsize=11)
axes[1, 0].grid(axis='y', alpha=0.3)

sorted_amounts = np.sort(amounts.sample(min(10000, len(amounts))))
cdf = np.arange(1, len(sorted_amounts) + 1) / len(sorted_amounts)
axes[1, 1].plot(sorted_amounts, cdf, linewidth=2, color='#FF6B6B')
axes[1, 1].set_xlabel('Amount', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('CDF', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.25)
fig.suptitle('Graph 7/8: Amount Distribution Analysis', fontsize=16, fontweight='bold', y=0.985)
print(f"   ✅ Amount distributions calculated")
print("   👁️  Displaying... (close window to continue)\n")
plt.show()

# ==============================================================================
# GRAPH 8: Performance Metrics Comparison
# ==============================================================================
print("📊 [8/8] Performance Metrics Comparison")
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
        print(f"   ✅ {name.upper()} metrics calculated")
    except:
        pass

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, (metric_name, values) in enumerate(metrics.items()):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(range(len(model_names)), values, color=colors)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.25)
fig.suptitle('Graph 8/8: Performance Metrics Comparison', fontsize=16, fontweight='bold', y=0.985)
print("   👁️  Displaying... (close window to finish)\n")
plt.show()

# ==============================================================================
# COMPLETE
# ==============================================================================
print("\n" + "=" * 70)
print("✅ ALL GRAPHS DISPLAYED SUCCESSFULLY!")
print("=" * 70)
print("\n🎉 Thank you for viewing all the fraud detection visualizations!")
print("\n💡 Tip: Run this script again anytime with:")
print("   python scripts/show_all_graphs.py")
print("\n" + "=" * 70)
