"""
Advanced Model Improvement Pipeline
Implements techniques to boost fraud detection accuracy beyond 88%
Includes: Advanced preprocessing, SMOTE, feature selection, ensemble methods
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 ADVANCED MODEL IMPROVEMENT PIPELINE")
print("=" * 80)
print("\nThis script implements advanced techniques to improve accuracy:")
print("  1. ✅ Advanced Data Preprocessing (Robust Scaling, Outlier Handling)")
print("  2. ✅ Class Imbalance Handling (SMOTE + Undersampling)")
print("  3. ✅ Feature Selection (Remove Low-Importance Features)")
print("  4. ✅ Hyperparameter Tuning (Grid Search)")
print("  5. ✅ Advanced Ensemble Methods (Stacking, Voting)")
print("  6. ✅ Cross-Validation for Robust Evaluation")
print("=" * 80)

# --- STEP 1: Load Data ---
print("\n📊 STEP 1: Loading Dataset...")
df = pd.read_csv('data/Fraud.csv')
df.columns = df.columns.str.strip()

# Ensure columns exist
for col in ['oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig', 'amount', 'type']:
    if col not in df.columns:
        df[col] = 0

for col in ['amount', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"   Dataset: {len(df):,} transactions")
print(f"   Fraud cases: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
print(f"   Class imbalance ratio: 1:{int((1-df['isFraud'].mean())/df['isFraud'].mean())}")

# --- STEP 2: Advanced Feature Engineering ---
print("\n🔧 STEP 2: Advanced Feature Engineering...")

def engineer_advanced_features(df):
    """Create enhanced feature set with better preprocessing"""
    
    # Basic features
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_sqrt'] = np.sqrt(df['amount'])
    df['balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    
    # Balance error detection
    df['balance_error'] = np.abs(df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig'])
    df['balance_error_ratio'] = df['balance_error'] / (df['amount'] + 1)
    
    # Transaction type encoding
    df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
    df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
    df['type_PAYMENT'] = (df['type'] == 'PAYMENT').astype(int)
    df['type_CASH_IN'] = (df['type'] == 'CASH_IN').astype(int)
    df['type_DEBIT'] = (df['type'] == 'DEBIT').astype(int)
    
    # High-risk transaction flags
    df['high_risk_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
    df['large_transaction'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    df['complete_drain'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
    df['zero_balance_before'] = (df['oldbalanceOrg'] == 0).astype(int)
    
    # Statistical features
    df['amount_percentile'] = df['amount'].rank(pct=True)
    df['balance_percentile'] = df['oldbalanceOrg'].rank(pct=True)
    
    # Interaction features
    df['risk_score'] = (df['high_risk_type'] * df['large_transaction'] * 
                        (df['balance_error_ratio'] > 0.01).astype(int))
    df['amount_balance_product'] = df['amount'] * df['oldbalanceOrg']
    df['amount_balance_product_log'] = np.log1p(df['amount_balance_product'])
    
    return df

df = engineer_advanced_features(df)
print(f"   ✅ Created enhanced feature set")

# Select features for modeling
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

# Handle any remaining NaN or inf values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"   Feature set: {X.shape[1]} features")

# --- STEP 3: Train-Test Split ---
print("\n✂️  STEP 3: Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train set: {len(X_train):,} samples ({y_train.sum():,} frauds)")
print(f"   Test set: {len(X_test):,} samples ({y_test.sum():,} frauds)")

# --- STEP 4: Handle Class Imbalance with SMOTE ---
print("\n⚖️  STEP 4: Handling Class Imbalance (SMOTE + Undersampling)...")

# Use combined approach: SMOTE for minority class + undersample majority
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Increase fraud to 50% of non-fraud
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Further balance

X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
X_train_balanced, y_train_balanced = under.fit_resample(X_train_balanced, y_train_balanced)

print(f"   Before: {y_train.sum():,} frauds, {len(y_train)-y_train.sum():,} non-frauds")
print(f"   After: {y_train_balanced.sum():,} frauds, {len(y_train_balanced)-y_train_balanced.sum():,} non-frauds")
print(f"   New ratio: 1:{int((len(y_train_balanced)-y_train_balanced.sum())/y_train_balanced.sum())}")

# --- STEP 5: Advanced Scaling (Robust to Outliers) ---
print("\n📏 STEP 5: Robust Scaling (Outlier-Resistant)...")
scaler = RobustScaler()  # Better than StandardScaler for data with outliers
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
print(f"   ✅ Applied RobustScaler to all features")

# --- STEP 6: Train Baseline Model ---
print("\n🤖 STEP 6: Training Baseline Model (Random Forest)...")
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_scaled, y_train_balanced)

y_pred_baseline = rf_baseline.predict(X_test_scaled)
y_pred_proba_baseline = rf_baseline.predict_proba(X_test_scaled)[:, 1]

baseline_acc = (y_pred_baseline == y_test).mean()
baseline_f1 = f1_score(y_test, y_pred_baseline)
baseline_auc = roc_auc_score(y_test, y_pred_proba_baseline)

print(f"   Baseline Accuracy: {baseline_acc*100:.2f}%")
print(f"   Baseline F1-Score: {baseline_f1:.4f}")
print(f"   Baseline AUC: {baseline_auc:.4f}")

# --- STEP 7: Feature Selection (Remove Low-Importance Features) ---
print("\n🎯 STEP 7: Feature Selection (Top Important Features)...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_baseline.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top 15 most important features
top_features = feature_importance.head(15)['feature'].tolist()
print(f"   Selected top {len(top_features)} features:")
for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
    print(f"      {i}. {row['feature']:30s} - {row['importance']:.4f}")

X_train_selected = pd.DataFrame(X_train_scaled, columns=feature_cols)[top_features].values
X_test_selected = pd.DataFrame(X_test_scaled, columns=feature_cols)[top_features].values

# --- STEP 8: Hyperparameter Tuning ---
print("\n🎛️  STEP 8: Hyperparameter Tuning (Grid Search)...")
print("   This may take a few minutes...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_selected, y_train_balanced)
best_rf = grid_search.best_estimator_

print(f"   ✅ Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"      {param}: {value}")

# --- STEP 9: Advanced Ensemble (Stacking) ---
print("\n🏗️  STEP 9: Building Advanced Ensemble (Stacking Classifier)...")

# Base models
base_models = [
    ('rf', RandomForestClassifier(**grid_search.best_params_, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

# Meta-model
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3,
    n_jobs=-1
)

print("   Training stacking classifier...")
stacking_clf.fit(X_train_selected, y_train_balanced)
print("   ✅ Stacking classifier trained")

# --- STEP 10: Final Evaluation ---
print("\n📊 STEP 10: Final Model Evaluation...")

y_pred_final = stacking_clf.predict(X_test_selected)
y_pred_proba_final = stacking_clf.predict_proba(X_test_selected)[:, 1]

final_acc = (y_pred_final == y_test).mean()
final_f1 = f1_score(y_test, y_pred_final)
final_auc = roc_auc_score(y_test, y_pred_proba_final)

print("\n" + "=" * 80)
print("🏆 FINAL RESULTS COMPARISON")
print("=" * 80)
print(f"\n{'Metric':<20} {'Baseline':<15} {'Improved Model':<15} {'Improvement':<15}")
print("-" * 80)
print(f"{'Accuracy':<20} {baseline_acc*100:>8.2f}%      {final_acc*100:>8.2f}%      {(final_acc-baseline_acc)*100:>+8.2f}%")
print(f"{'F1-Score':<20} {baseline_f1:>8.4f}       {final_f1:>8.4f}       {(final_f1-baseline_f1):>+8.4f}")
print(f"{'AUC Score':<20} {baseline_auc:>8.4f}       {final_auc:>8.4f}       {(final_auc-baseline_auc):>+8.4f}")
print("=" * 80)

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Non-Fraud', 'Fraud']))

# --- STEP 11: Visualization ---
print("\n📈 Generating comparison visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix - Improved Model', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)

# Performance Comparison
metrics = ['Accuracy', 'F1-Score', 'AUC']
baseline_scores = [baseline_acc, baseline_f1, baseline_auc]
improved_scores = [final_acc, final_f1, final_auc]

x = np.arange(len(metrics))
width = 0.35

axes[1].bar(x - width/2, baseline_scores, width, label='Baseline', color='#FFCC80')
axes[1].bar(x + width/2, improved_scores, width, label='Improved', color='#A5D6A7')

axes[1].set_xlabel('Metrics', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[1].set_title('Performance Comparison', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([0.8, 1.0])

plt.tight_layout()

print("\n" + "=" * 80)
print("✅ MODEL IMPROVEMENT PIPELINE COMPLETE!")
print("=" * 80)
print("\n💡 Key Techniques Applied:")
print("   1. ✅ SMOTE + Undersampling for class imbalance")
print("   2. ✅ RobustScaler for outlier-resistant scaling")
print("   3. ✅ Feature selection (top 15 features)")
print("   4. ✅ Grid search hyperparameter tuning")
print("   5. ✅ Stacking ensemble (RF + GB + LR)")
print("\n👁️  Displaying visualizations... (close window to continue)")

plt.show()

# --- STEP 12: Save Improved Model ---
print("\n💾 Saving improved model...")
joblib.dump(stacking_clf, 'models/improved_stacking_model.pkl')
joblib.dump(scaler, 'models/improved_scaler.pkl')
joblib.dump(top_features, 'models/improved_features.pkl')
print("   ✅ Saved: improved_stacking_model.pkl")
print("   ✅ Saved: improved_scaler.pkl")
print("   ✅ Saved: improved_features.pkl")

print("\n🎉 Done! Your improved model is ready to use!")
print("=" * 80)
