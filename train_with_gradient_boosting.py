"""
Train Gradient Boosting Classifier and integrate into existing ensemble
Adds sklearn's GradientBoostingClassifier to complement XGB/LGB/CatBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 TRAINING GRADIENT BOOSTING CLASSIFIER")
print("="*80)

# ============================================================================
# 1. LOAD EXISTING MODELS AND METADATA
# ============================================================================
print("\n📂 Step 1: Loading existing metadata...")
models_dir = "models"
meta_path = os.path.join(models_dir, "advanced_metadata.json")

if not os.path.exists(meta_path):
    raise FileNotFoundError(f"❌ Metadata not found at {meta_path}. Train base models first!")

with open(meta_path, 'r') as f:
    metadata = json.load(f)

feature_names = metadata.get("feature_names", [])
print(f"   ✅ Loaded metadata with {len(feature_names)} features")

# Load scaler
scaler_path = os.path.join(models_dir, "scaler.pkl")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(f"   ✅ Loaded existing scaler")
else:
    scaler = None
    print(f"   ⚠️ No scaler found - will create new one")

# ============================================================================
# 2. LOAD AND PREPARE DATA
# ============================================================================
print("\n📂 Step 2: Loading dataset...")
df = pd.read_csv('data/Fraud.csv')
print(f"   Total transactions: {len(df):,}")
print(f"   Fraud cases: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")

# ============================================================================
# 3. FEATURE ENGINEERING (Reuse same logic as app.py)
# ============================================================================
print("\n🔧 Step 3: Engineering features...")

from sklearn.preprocessing import LabelEncoder

# Encode categoricals
le_city = LabelEncoder()
le_card = LabelEncoder()
le_exp = LabelEncoder()
le_gender = LabelEncoder()

df['city_encoded'] = le_city.fit_transform(df['City'])
df['card_encoded'] = le_card.fit_transform(df['Card Type'])
df['exp_encoded'] = le_exp.fit_transform(df['Exp Type'])
df['gender_encoded'] = le_gender.fit_transform(df['Gender'])

# Basic features
df['amount_log'] = np.log1p(df['amount'])
df['amount_sqrt'] = np.sqrt(df['amount'])
df['balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['amount_to_balance_ratio'] = np.where(df['oldbalanceOrg'] > 0, 
                                          df['amount'] / df['oldbalanceOrg'], 0)

# Balance errors
df['balance_error'] = np.abs(df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig'])
df['balance_error_ratio'] = np.where(df['oldbalanceOrg'] > 0,
                                      df['balance_error'] / df['oldbalanceOrg'], 0)
df['has_balance_error'] = (df['balance_error'] > 0.01).astype(int)
df['large_balance_error'] = (df['balance_error'] > 100).astype(int)

# Zero balance flags
df['zero_balance_before'] = (df['oldbalanceOrg'] == 0).astype(int)
df['zero_balance_after'] = (df['newbalanceOrig'] == 0).astype(int)

# Drain patterns
drain_ratio = np.where(df['oldbalanceOrg'] > 0, df['amount'] / df['oldbalanceOrg'], 0)
df['complete_drain'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
df['partial_drain'] = ((drain_ratio > 0.5) & (drain_ratio < 0.95)).astype(int)
df['high_drain_ratio'] = (drain_ratio >= 0.95).astype(int)
df['medium_drain_ratio'] = ((drain_ratio >= 0.5) & (drain_ratio < 0.95)).astype(int)
df['low_drain_ratio'] = ((drain_ratio > 0) & (drain_ratio < 0.5)).astype(int)
df['near_complete_drain'] = ((drain_ratio >= 0.80) & (drain_ratio < 0.95)).astype(int)

# Balance matching
df['exact_balance_match'] = (df['newbalanceOrig'] == df['oldbalanceOrg'] - df['amount']).astype(int)
df['almost_exact_match'] = (np.abs(df['newbalanceOrig'] - (df['oldbalanceOrg'] - df['amount'])) < 1).astype(int)
df['balance_mismatch'] = (df['balance_error'] > df['amount'] * 0.01).astype(int)
df['suspicious_zero_transaction'] = ((df['oldbalanceOrg'] == 0) & (df['amount'] > 0)).astype(int)

# Amount categorization
amount_quintile_thresholds = [0, 13000, 74000, 163000, 350000, float('inf')]
df['amount_quintile'] = pd.cut(df['amount'], bins=amount_quintile_thresholds, 
                                labels=[0,1,2,3,4], include_lowest=True).astype(int)

amount_decile_thresholds = [0, 5000, 13000, 30000, 74000, 120000, 163000, 250000, 350000, 500000, float('inf')]
df['amount_decile'] = pd.cut(df['amount'], bins=amount_decile_thresholds, 
                              labels=[0,1,2,3,4,5,6,7,8,9], include_lowest=True).astype(int)

# Amount flags
df['round_amount'] = (np.mod(df['amount'], 1000) == 0).astype(int)
df['round_large_amount'] = ((df['amount'] >= 100000) & (np.mod(df['amount'], 1000) == 0)).astype(int)
df['round_medium_amount'] = ((df['amount'] >= 10000) & (df['amount'] < 100000) & 
                              (np.mod(df['amount'], 1000) == 0)).astype(int)
df['odd_amount'] = (np.mod(df['amount'], 1000) != 0).astype(int)
df['small_amount'] = (df['amount'] < 1000).astype(int)

# Outliers
amount_mean = df['amount'].mean()
amount_std = df['amount'].std()
df['amount_zscore'] = (df['amount'] - amount_mean) / (amount_std + 1e-10)
df['amount_outlier_99'] = (np.abs(df['amount_zscore']) > 2.58).astype(int)
df['amount_outlier_95'] = (np.abs(df['amount_zscore']) > 1.96).astype(int)
df['amount_outlier_90'] = (np.abs(df['amount_zscore']) > 1.64).astype(int)

balance_mean = df['oldbalanceOrg'].mean()
balance_std = df['oldbalanceOrg'].std()
df['balance_zscore'] = (df['oldbalanceOrg'] - balance_mean) / (balance_std + 1e-10)
df['balance_zscore_outlier'] = (np.abs(df['balance_zscore']) > 2.58).astype(int)
df['amount_zscore_outlier'] = (np.abs(df['amount_zscore']) > 2.58).astype(int)

# IQR outliers
amount_q1, amount_q3 = df['amount'].quantile(0.25), df['amount'].quantile(0.75)
amount_iqr = amount_q3 - amount_q1
df['amount_iqr_outlier'] = ((df['amount'] < amount_q1 - 1.5*amount_iqr) | 
                             (df['amount'] > amount_q3 + 1.5*amount_iqr)).astype(int)

balance_q1, balance_q3 = df['oldbalanceOrg'].quantile(0.25), df['oldbalanceOrg'].quantile(0.75)
balance_iqr = balance_q3 - balance_q1
df['balance_iqr_outlier'] = ((df['oldbalanceOrg'] < balance_q1 - 1.5*balance_iqr) | 
                              (df['oldbalanceOrg'] > balance_q3 + 1.5*balance_iqr)).astype(int)

df['extreme_outlier'] = ((df['amount_outlier_99'] == 1) | (df['balance_zscore_outlier'] == 1)).astype(int)

# Percentiles
balance_percentiles = [0, 1, 13000, 50000, 150000, 500000, 1000000, 5000000, float('inf')]
df['balance_percentile'] = pd.cut(df['oldbalanceOrg'], bins=balance_percentiles, 
                                   labels=[0, 10, 25, 50, 75, 90, 95, 99], include_lowest=True).astype(float)

amount_percentiles = [0, 5000, 13000, 30000, 74000, 150000, 350000, 500000, float('inf')]
df['amount_percentile'] = pd.cut(df['amount'], bins=amount_percentiles, 
                                  labels=[10, 25, 40, 50, 75, 85, 95, 99], include_lowest=True).astype(float)

df['percentile_diff'] = df['balance_percentile'] - df['amount_percentile']

# Transaction type features
df['transfer_large'] = ((df['type'] == 'TRANSFER') & (df['amount'] > 200000)).astype(int)
df['transfer_medium'] = ((df['type'] == 'TRANSFER') & (df['amount'] >= 50000) & 
                         (df['amount'] <= 200000)).astype(int)
df['cashout_large'] = ((df['type'] == 'CASH_OUT') & (df['amount'] > 200000)).astype(int)
df['cashout_medium'] = ((df['type'] == 'CASH_OUT') & (df['amount'] >= 50000) & 
                        (df['amount'] <= 200000)).astype(int)
df['payment_large'] = ((df['type'] == 'PAYMENT') & (df['amount'] > 100000)).astype(int)
df['transfer_or_cashout'] = (df['type'].isin(['TRANSFER', 'CASH_OUT'])).astype(int)
df['high_risk_type'] = (df['type'].isin(['TRANSFER', 'CASH_OUT'])).astype(int)
df['low_risk_type'] = (df['type'].isin(['PAYMENT', 'CASH_IN'])).astype(int)

type_risk_map = {"TRANSFER": 0.8, "CASH_OUT": 0.8, "PAYMENT": 0.2, "CASH_IN": 0.2, "DEBIT": 0.4}
df['type_risk_score'] = df['type'].map(type_risk_map)
df['risky_transaction'] = ((df['high_risk_type'] == 1) & (df['amount'] > 100000)).astype(int)

# Ratio features
df['new_to_old_balance_ratio'] = np.where(df['oldbalanceOrg'] > 0, 
                                           df['newbalanceOrig'] / df['oldbalanceOrg'], 0)
df['amount_balance_product'] = df['amount'] * df['oldbalanceOrg']
df['amount_balance_product_log'] = np.log1p(df['amount_balance_product'])
df['balance_change_pct'] = np.where(df['oldbalanceOrg'] > 0,
                                     (df['oldbalanceOrg'] - df['newbalanceOrig']) / df['oldbalanceOrg'], 0)
df['extreme_change'] = (np.abs(df['balance_change_pct']) > 0.95).astype(int)

# Type encoding
type_encode_map = {"TRANSFER": 1, "CASH_OUT": 2, "PAYMENT": 0, "CASH_IN": 3, "DEBIT": 4}
df['type_encoded'] = df['type'].map(type_encode_map)
df['dataset_source_encoded'] = 0

print(f"   ✅ Created {len([c for c in df.columns if c not in ['City', 'Card Type', 'Exp Type', 'Gender', 'type', 'isFraud']])} features")

# ============================================================================
# 4. PREPARE TRAINING DATA
# ============================================================================
print("\n📊 Step 4: Preparing training data...")

# Select features matching metadata
if feature_names:
    # Ensure all required features exist
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"   ⚠️ Missing features: {missing_features[:5]}...")
        # Add missing features as zeros
        for feat in missing_features:
            df[feat] = 0
    
    X = df[feature_names]
else:
    # Use all numeric features
    X = df.select_dtypes(include=[np.number]).drop(columns=['isFraud'], errors='ignore')

y = df['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Train fraud rate: {y_train.mean()*100:.2f}%")

# Scale features
if scaler is None:
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✅ Created new scaler")
else:
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✅ Used existing scaler")

# ============================================================================
# 5. TRAIN GRADIENT BOOSTING CLASSIFIER
# ============================================================================
print("\n🌳 Step 5: Training Gradient Boosting Classifier...")
print("   Configuration:")
print("   - n_estimators: 200 (boosting rounds)")
print("   - max_depth: 5 (shallow trees prevent overfitting)")
print("   - learning_rate: 0.1 (moderate for stability)")
print("   - subsample: 0.8 (80% data per tree)")
print("   - min_samples_split: 100 (robust splits)")
print()

# Calculate scale for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"   Class imbalance ratio: {scale_pos_weight:.2f}:1 (legit:fraud)")

gb_model = GradientBoostingClassifier(
    n_estimators=200,        # 200 boosting rounds (good balance of speed/accuracy)
    max_depth=5,             # Shallow trees (prevents overfitting)
    learning_rate=0.1,       # Moderate learning rate
    subsample=0.8,           # 80% row sampling per tree
    min_samples_split=100,   # Require 100 samples to split
    min_samples_leaf=50,     # Require 50 samples per leaf
    max_features='sqrt',     # Use sqrt(n_features) for each split
    random_state=42,
    verbose=1                # Show progress
)

print("   Training started...")
gb_model.fit(X_train_scaled, y_train)
print("   ✅ Training complete!")

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================
print("\n📊 Step 6: Evaluating model...")

# Predictions
gb_pred = gb_model.predict(X_test_scaled)
gb_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
gb_auc = roc_auc_score(y_test, gb_proba)

print(f"\n   ✅ Gradient Boosting AUC: {gb_auc:.4f} ({gb_auc*100:.2f}%)")
print("\n   Classification Report:")
print(classification_report(y_test, gb_pred, target_names=['Legitimate', 'Fraud'], digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, gb_pred)
print("\n   Confusion Matrix:")
print(f"   TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
print(f"   FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")

# ============================================================================
# 7. SAVE MODEL AND UPDATE METADATA
# ============================================================================
print("\n💾 Step 7: Saving model and updating metadata...")

# Save model
gb_model_path = os.path.join(models_dir, "gb_model.pkl")
joblib.dump(gb_model, gb_model_path)
print(f"   ✅ Saved: {gb_model_path}")

# Update metadata
metadata['ensemble_weights']['gb'] = 0.20  # Initial weight (can be tuned)
metadata['auc_scores'] = metadata.get('auc_scores', {})
metadata['auc_scores']['gradient_boosting'] = float(gb_auc)
metadata['last_updated'] = datetime.now().isoformat()

with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✅ Updated metadata")

# Show ensemble weights
print("\n   Current ensemble weights:")
for model, weight in metadata['ensemble_weights'].items():
    print(f"   - {model}: {weight:.4f}")

print("\n" + "="*80)
print("✅ GRADIENT BOOSTING TRAINING COMPLETE!")
print("="*80)
print("\n📋 Next Steps:")
print("   1. Update app/app.py to load 'gb_model.pkl'")
print("   2. Test the ensemble with new predictions")
print("   3. Optionally tune ensemble weights for better performance")
print("\n   Quick integration (add to app.py load_models):")
print('   model_files = {..., "gb": "gb_model.pkl"}')
print("\n" + "="*80)
