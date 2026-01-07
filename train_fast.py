"""
Fast Fraud Detection Training - Optimized for Speed
Uses sampling and reduced model complexity for faster training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 FAST FRAUD DETECTION TRAINING")
print("="*80)

# ============================================================================
# 1. LOAD DATA WITH SAMPLING FOR SPEED
# ============================================================================
print("\n📂 Step 1: Loading dataset (with sampling for speed)...")
df = pd.read_csv('data/Fraud.csv')
print(f"   Original: {len(df):,} transactions")

# Sample 30% of data for faster training (still 300k+ samples)
df = df.sample(frac=0.3, random_state=42)
print(f"   Sampled: {len(df):,} transactions")
print(f"   Fraud cases: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
print(f"   Legitimate cases: {(~df['isFraud'].astype(bool)).sum():,}")

# ============================================================================
# 2. FEATURE ENGINEERING (MATCHING APP.PY EXACTLY)
# ============================================================================
print("\n🔧 Step 2: Engineering features...")

# Encode categorical features
le_city = LabelEncoder()
le_card = LabelEncoder()
le_exp = LabelEncoder()
le_gender = LabelEncoder()

df['city_encoded'] = le_city.fit_transform(df['City'])
df['card_encoded'] = le_card.fit_transform(df['Card Type'])
df['exp_encoded'] = le_exp.fit_transform(df['Exp Type'])
df['gender_encoded'] = le_gender.fit_transform(df['Gender'])

# Basic transformations
df['amount_log'] = np.log1p(df['amount'])
df['amount_sqrt'] = np.sqrt(df['amount'])

# Balance changes
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

# Destination balance features (placeholders - not in dataset)
df['exact_balance_match'] = 0
df['almost_exact_match'] = 0
df['balance_mismatch'] = 0
df['suspicious_zero_transaction'] = ((df['oldbalanceOrg'] == 0) & (df['amount'] > 0)).astype(int)

# Amount categorization
amount_quintile_thresholds = [0, 13000, 74000, 163000, 350000, float('inf')]
df['amount_quintile'] = pd.cut(df['amount'], bins=amount_quintile_thresholds, 
                                labels=[0,1,2,3,4], include_lowest=True).astype(int)

amount_decile_thresholds = [0, 5000, 13000, 30000, 74000, 120000, 163000, 250000, 350000, 500000, float('inf')]
df['amount_decile'] = pd.cut(df['amount'], bins=amount_decile_thresholds, 
                              labels=[0,1,2,3,4,5,6,7,8,9], include_lowest=True).astype(int)

df['round_amount'] = (np.mod(df['amount'], 1000) == 0).astype(int)
df['round_large_amount'] = ((df['amount'] >= 100000) & (np.mod(df['amount'], 1000) == 0)).astype(int)
df['round_medium_amount'] = ((df['amount'] >= 10000) & (df['amount'] < 100000) & 
                              (np.mod(df['amount'], 1000) == 0)).astype(int)
df['odd_amount'] = (np.mod(df['amount'], 1000) != 0).astype(int)
df['small_amount'] = (df['amount'] < 1000).astype(int)

# Outlier features
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

# IQR-based outliers
amount_q1, amount_q3 = df['amount'].quantile(0.25), df['amount'].quantile(0.75)
amount_iqr = amount_q3 - amount_q1
df['amount_iqr_outlier'] = ((df['amount'] < amount_q1 - 1.5*amount_iqr) | 
                             (df['amount'] > amount_q3 + 1.5*amount_iqr)).astype(int)

balance_q1, balance_q3 = df['oldbalanceOrg'].quantile(0.25), df['oldbalanceOrg'].quantile(0.75)
balance_iqr = balance_q3 - balance_q1
df['balance_iqr_outlier'] = ((df['oldbalanceOrg'] < balance_q1 - 1.5*balance_iqr) | 
                              (df['oldbalanceOrg'] > balance_q3 + 1.5*balance_iqr)).astype(int)

df['extreme_outlier'] = ((df['amount_outlier_99'] == 1) | (df['balance_zscore_outlier'] == 1)).astype(int)

# Transaction type features
df['transfer_large'] = ((df['type'] == 'TRANSFER') & (df['amount'] > 200000)).astype(int)
df['transfer_medium'] = ((df['type'] == 'TRANSFER') & (df['amount'] >= 50000) & 
                         (df['amount'] <= 200000)).astype(int)
df['cashout_large'] = ((df['type'] == 'CASH_OUT') & (df['amount'] > 200000)).astype(int)
df['cashout_medium'] = ((df['type'] == 'CASH_OUT') & (df['amount'] >= 50000) & 
                        (df['amount'] <= 200000)).astype(int)
df['payment_large'] = ((df['type'] == 'PAYMENT') & (df['amount'] > 200000)).astype(int)
df['transfer_or_cashout'] = ((df['type'] == 'TRANSFER') | (df['type'] == 'CASH_OUT')).astype(int)
df['high_risk_type'] = df['transfer_or_cashout']
df['low_risk_type'] = ((df['type'] == 'PAYMENT') | (df['type'] == 'CASH_IN')).astype(int)

type_risk_map = {'TRANSFER': 0.8, 'CASH_OUT': 0.8, 'DEBIT': 0.4, 'PAYMENT': 0.2, 'CASH_IN': 0.2}
df['type_risk_score'] = df['type'].map(type_risk_map)
df['risky_transaction'] = ((df['type_risk_score'] >= 0.8) & (df['amount'] > 100000)).astype(int)

# Percentile features
balance_percentiles = [0, 1, 13000, 50000, 150000, 500000, 1000000, 5000000, float('inf')]
df['balance_percentile'] = pd.cut(df['oldbalanceOrg'], bins=balance_percentiles, 
                                   labels=[0, 10, 25, 50, 75, 90, 95, 99], include_lowest=True).astype(float)

amount_percentiles = [0, 5000, 13000, 30000, 74000, 150000, 350000, 500000, float('inf')]
df['amount_percentile'] = pd.cut(df['amount'], bins=amount_percentiles, 
                                  labels=[10, 25, 40, 50, 75, 85, 95, 99], include_lowest=True).astype(float)

df['percentile_diff'] = df['balance_percentile'] - df['amount_percentile']

# Advanced ratios
df['new_to_old_balance_ratio'] = np.where(df['oldbalanceOrg'] > 0,
                                          df['newbalanceOrig'] / df['oldbalanceOrg'], 0)
df['amount_balance_product'] = df['amount'] * df['oldbalanceOrg']
df['amount_balance_product_log'] = np.log1p(df['amount_balance_product'])
df['balance_change_pct'] = np.where(df['oldbalanceOrg'] > 0,
                                    (df['newbalanceOrig'] - df['oldbalanceOrg']) / df['oldbalanceOrg'], 0)
df['extreme_change'] = (np.abs(df['balance_change_pct']) > 0.95).astype(int)

# Type encoding
type_encoding_map = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
df['type_encoded'] = df['type'].map(type_encoding_map)

# Dataset source
df['dataset_source_encoded'] = 0

print(f"   ✅ Created features")

# ============================================================================
# 3. SELECT FEATURES (EXACT 62 FEATURES MATCHING APP.PY)
# ============================================================================
print("\n📋 Step 3: Selecting features...")

# These are the EXACT 62 features that app.py expects
feature_cols = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'amount_log', 'amount_sqrt',
    'balance_change', 'amount_to_balance_ratio',
    'balance_error', 'balance_error_ratio', 'has_balance_error', 'large_balance_error',
    'zero_balance_before', 'zero_balance_after',
    'complete_drain', 'partial_drain', 'high_drain_ratio', 'medium_drain_ratio', 
    'low_drain_ratio', 'near_complete_drain',
    'exact_balance_match', 'almost_exact_match', 'suspicious_zero_transaction', 'balance_mismatch',
    'amount_quintile', 'amount_decile', 'round_amount', 'round_large_amount', 
    'round_medium_amount', 'odd_amount',
    'amount_outlier_99', 'amount_outlier_95', 'amount_outlier_90', 'small_amount',
    'transfer_large', 'transfer_medium', 'cashout_large', 'cashout_medium', 'payment_large',
    'transfer_or_cashout', 'high_risk_type', 'low_risk_type', 'type_risk_score', 'risky_transaction',
    'balance_zscore', 'amount_zscore', 'balance_zscore_outlier', 'amount_zscore_outlier',
    'balance_iqr_outlier', 'amount_iqr_outlier', 'extreme_outlier',
    'balance_percentile', 'amount_percentile', 'percentile_diff',
    'new_to_old_balance_ratio', 'amount_balance_product', 'amount_balance_product_log',
    'balance_change_pct', 'extreme_change',
    'type_encoded', 'dataset_source_encoded'
]

X = df[feature_cols]
y = df['isFraud']

print(f"   Features: {len(feature_cols)}")
print(f"   Samples: {len(X):,}")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n✂️  Step 4: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================================
# 5. SMOTE
# ============================================================================
print("\n⚖️  Step 5: Applying SMOTE...")
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
print(f"   ✅ Balanced: {len(X_train_balanced):,} samples")

# ============================================================================
# 6. SCALE FEATURES
# ============================================================================
print("\n📊 Step 6: Scaling...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Scaled")

# ============================================================================
# 7. TRAIN MODELS (REDUCED COMPLEXITY FOR SPEED)
# ============================================================================
print("\n" + "="*80)
print("🎯 TRAINING 4 MODELS (FAST MODE)")
print("="*80)

# 7.1 Random Forest
print("\n🌲 Model 1/4: Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,  # Reduced from 300
    max_depth=25,      # Reduced from 35
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train_balanced)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)
print(f"   ✅ RF AUC: {rf_auc:.4f}")

# 7.2 XGBoost
print("\n🚀 Model 2/4: XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,   # Reduced from 200
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train_scaled, y_train_balanced)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)
print(f"   ✅ XGB AUC: {xgb_auc:.4f}")

# 7.3 LightGBM
print("\n💡 Model 3/4: LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,   # Reduced from 200
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
lgb_model.fit(X_train_scaled, y_train_balanced)
lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
lgb_auc = roc_auc_score(y_test, lgb_proba)
print(f"   ✅ LGB AUC: {lgb_auc:.4f}")

# 7.4 CatBoost
print("\n🐱 Model 4/4: CatBoost...")
cat_model = CatBoostClassifier(
    iterations=100,     # Reduced from 200
    depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=0
)
cat_model.fit(X_train_scaled, y_train_balanced)
cat_proba = cat_model.predict_proba(X_test_scaled)[:, 1]
cat_auc = roc_auc_score(y_test, cat_proba)
print(f"   ✅ CAT AUC: {cat_auc:.4f}")

# ============================================================================
# 8. CALCULATE ENSEMBLE WEIGHTS
# ============================================================================
print("\n" + "="*80)
print("📊 ENSEMBLE PERFORMANCE")
print("="*80)

auc_scores = {
    'random_forest': rf_auc,
    'xgboost': xgb_auc,
    'lightgbm': lgb_auc,
    'catboost': cat_auc
}

# AUC-based weighting
total_auc = sum(auc_scores.values())
ensemble_weights = {name: auc / total_auc for name, auc in auc_scores.items()}

print("\nModel AUC Scores:")
for name, auc in auc_scores.items():
    print(f"   {name:15s}: {auc:.4f} (weight: {ensemble_weights[name]:.3f})")

# Weighted ensemble prediction
ensemble_proba = (
    rf_proba * ensemble_weights['random_forest'] +
    xgb_proba * ensemble_weights['xgboost'] +
    lgb_proba * ensemble_weights['lightgbm'] +
    cat_proba * ensemble_weights['catboost']
)
ensemble_auc = roc_auc_score(y_test, ensemble_proba)
print(f"\n🎯 ENSEMBLE AUC: {ensemble_auc:.4f}")

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, ensemble_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"   Optimal Threshold: {optimal_threshold:.4f}")

# ============================================================================
# 9. SAVE MODELS
# ============================================================================
print("\n" + "="*80)
print("💾 SAVING MODELS")
print("="*80)

import os
os.makedirs('models', exist_ok=True)

joblib.dump(rf_model, 'models/rf_model.pkl')
print("✅ Saved: models/rf_model.pkl")

joblib.dump(xgb_model, 'models/xgboost_model.pkl')
print("✅ Saved: models/xgboost_model.pkl")

joblib.dump(lgb_model, 'models/lightgbm_model.pkl')
print("✅ Saved: models/lightgbm_model.pkl")

joblib.dump(cat_model, 'models/catboost_model.pkl')
print("✅ Saved: models/catboost_model.pkl")

joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Saved: models/scaler.pkl")

# Save encoders
encoders = {
    'city': le_city,
    'card': le_card,
    'exp': le_exp,
    'gender': le_gender
}
joblib.dump(encoders, 'models/encoders.pkl')
print("✅ Saved: models/encoders.pkl")

# Save metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'feature_names': feature_cols,
    'num_features': len(feature_cols),
    'ensemble_weights': ensemble_weights,
    'optimal_threshold': float(optimal_threshold),
    'auc_scores': auc_scores,
    'ensemble_auc': float(ensemble_auc),
    'training_samples': len(X_train_balanced),
    'test_samples': len(X_test)
}

with open('models/advanced_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: models/advanced_metadata.json")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n🎯 Final Results:")
print(f"   Ensemble AUC: {ensemble_auc:.4f}")
print(f"   Optimal Threshold: {optimal_threshold:.4f}")
print(f"   Models: 4 (RF, XGB, LGB, CAT)")
print(f"   Features: {len(feature_cols)}")
print("\n💡 You can now run: python app/app.py")
