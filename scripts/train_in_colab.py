"""
Google Colab - Fraud Detection Model Training
Upload this notebook to Google Colab and run it cell by cell
"""

# ========== CELL 1: Install Required Packages ==========
!pip install imbalanced-learn xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn

# ========== CELL 2: Upload Data File ==========
from google.colab import files
import io
import pandas as pd

print("📤 Please upload your Fraud.csv file:")
uploaded = files.upload()

# Load the uploaded file
df = pd.read_csv(io.BytesIO(uploaded['Fraud.csv']))
print(f"✅ Loaded {len(df):,} transactions")
print(f"   Fraud cases: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")

# ========== CELL 3: Data Preprocessing & Feature Engineering ==========
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

print("\n🔧 Feature Engineering...")

df.columns = df.columns.str.strip()

# Ensure required columns exist
for col in ['oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig', 'amount', 'type']:
    if col not in df.columns:
        df[col] = 0

for col in ['amount', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Feature engineering
df['amount_log'] = np.log1p(df['amount'])
df['amount_sqrt'] = np.sqrt(df['amount'])
df['balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
df['balance_error'] = np.abs(df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig'])
df['balance_error_ratio'] = df['balance_error'] / (df['amount'] + 1)

# Type encoding
df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
df['type_PAYMENT'] = (df['type'] == 'PAYMENT').astype(int)
df['type_CASH_IN'] = (df['type'] == 'CASH_IN').astype(int)
df['type_DEBIT'] = (df['type'] == 'DEBIT').astype(int)

# Risk features
df['high_risk_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
df['large_transaction'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
df['complete_drain'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
df['zero_balance_before'] = (df['oldbalanceOrg'] == 0).astype(int)
df['amount_percentile'] = df['amount'].rank(pct=True)
df['balance_percentile'] = df['oldbalanceOrg'].rank(pct=True)
df['risk_score'] = (df['high_risk_type'] * df['large_transaction'] * 
                    (df['balance_error_ratio'] > 0.01).astype(int))
df['amount_balance_product_log'] = np.log1p(df['amount'] * df['oldbalanceOrg'])

feature_cols = [
    'amount', 'amount_log', 'amount_sqrt',
    'oldbalanceOrg', 'newbalanceOrig', 'balance_change',
    'amount_to_balance_ratio', 'balance_error', 'balance_error_ratio',
    'type_TRANSFER', 'type_CASH_OUT', 'type_PAYMENT', 'type_CASH_IN', 'type_DEBIT',
    'high_risk_type', 'large_transaction', 'complete_drain', 'zero_balance_before',
    'amount_percentile', 'balance_percentile', 'risk_score',
    'amount_balance_product_log'
]

X = df[feature_cols].copy()
y = df['isFraud'].copy()

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"✅ Created {X.shape[1]} features")

# ========== CELL 4: Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ========== CELL 5: Handle Class Imbalance (SMOTE) ==========
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

print("\n⚖️  Balancing classes with SMOTE...")
smote = SMOTE(sampling_strategy=0.5, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
X_train_balanced, y_train_balanced = under.fit_resample(X_train_balanced, y_train_balanced)

print(f"✅ Balanced: {y_train_balanced.sum():,} frauds, {len(y_train_balanced)-y_train_balanced.sum():,} non-frauds")

# ========== CELL 6: Scale Features ==========
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled")

# ========== CELL 7: Train Models ==========
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib

print("\n🤖 Training models...")

# Train individual models
print("   Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train_balanced)

print("   Training XGBoost...")
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train_balanced)

print("   Training LightGBM...")
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
lgb_model.fit(X_train_scaled, y_train_balanced)

print("   Training CatBoost...")
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(iterations=200, depth=10, random_state=42, verbose=0)
cat_model.fit(X_train_scaled, y_train_balanced)

print("   Training Stacking Ensemble...")
base_models = [
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('lgb', lgb_model)
]
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3,
    n_jobs=-1
)
stacking_model.fit(X_train_scaled, y_train_balanced)

print("✅ All models trained!")

# ========== CELL 8: Evaluate Models ==========
print("\n📊 Model Evaluation:")
print("=" * 80)

models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'CatBoost': cat_model,
    'Stacking Ensemble': stacking_model
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Model': name,
        'Accuracy': f"{acc*100:.2f}%",
        'F1-Score': f"{f1:.4f}",
        'AUC': f"{auc:.4f}"
    })
    print(f"{name:20s} | Acc: {acc*100:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}")

print("=" * 80)

# ========== CELL 9: Visualize Results ==========
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (name, model) in enumerate(models.items()):
    row = idx // 3
    col = idx % 3
    
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
    axes[row, col].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[row, col].set_xlabel('Predicted')
    axes[row, col].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ========== CELL 10: Download Trained Models ==========
print("\n💾 Saving models...")

joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(lgb_model, 'lightgbm_model.pkl')
joblib.dump(cat_model, 'catboost_model.pkl')
joblib.dump(stacking_model, 'stacking_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Models saved!")

# Download files
from google.colab import files
print("\n📥 Downloading models...")
files.download('rf_model.pkl')
files.download('xgboost_model.pkl')
files.download('lightgbm_model.pkl')
files.download('catboost_model.pkl')
files.download('stacking_model.pkl')
files.download('scaler.pkl')

print("\n🎉 Training complete! Models downloaded to your computer.")
