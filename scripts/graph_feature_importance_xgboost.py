"""
Feature Importance Analysis for XGBoost Model
Shows which features/columns are most important for fraud detection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import xgboost as xgb

print("=" * 70)
print("📊 Feature Importance - XGBoost Model")
print("=" * 70)

# Load XGBoost model
print("\n🤖 Loading XGBoost model...")
try:
    xgb_model = joblib.load('models/xgboost_model.pkl')
    print("   ✅ XGBoost model loaded")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    exit(1)

# Load metadata to get feature names
print("\n📋 Loading feature names...")
try:
    with open('models/advanced_metadata.json', 'r') as f:
        metadata = json.load(f)
        feature_names = metadata.get("feature_names", [])
    print(f"   ✅ Loaded {len(feature_names)} feature names")
except Exception as e:
    print(f"   ❌ Error loading metadata: {e}")
    exit(1)

# Get feature importances from the model
print("\n🔍 Extracting feature importances...")
try:
    importances = xgb_model.feature_importances_
    
    # Create dataframe with feature names and importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"   ✅ Extracted importance scores for {len(importances)} features")
    
except Exception as e:
    print(f"   ❌ Error extracting importances: {e}")
    exit(1)

# Display top features in console
print("\n" + "=" * 70)
print("🏆 TOP 20 MOST IMPORTANT FEATURES (XGBoost)")
print("=" * 70)
for idx, row in feature_importance_df.head(20).iterrows():
    print(f"{row['Feature']:40s} | {row['Importance']:.6f} ({row['Importance']*100:.2f}%)")

# Create visualization
print("\n📈 Generating feature importance visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Graph 1: Top 20 Features (Horizontal Bar Chart)
top_20 = feature_importance_df.head(20)

ax1.barh(range(len(top_20)), top_20['Importance'], color='#2E86AB')
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels(top_20['Feature'], fontsize=10)
ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax1.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.invert_yaxis()  # Highest importance at top

# Add value labels on bars
for i, (idx, row) in enumerate(top_20.iterrows()):
    ax1.text(row['Importance'], i, f' {row["Importance"]:.4f}', 
             va='center', fontsize=9, fontweight='bold')

# Graph 2: Feature Importance Distribution (Cumulative)
cumsum = feature_importance_df['Importance'].cumsum()
ax2.plot(range(1, len(cumsum) + 1), cumsum, linewidth=3, color='#2E86AB', marker='o', 
         markersize=4, markevery=5)
ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='50% Threshold')
ax2.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='80% Threshold')
ax2.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Threshold')

ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='lower right', fontsize=10)

# Find how many features capture 50%, 80%, 95%
n_50 = (cumsum >= 0.5).idxmax() + 1 if not cumsum.empty else 0
n_80 = (cumsum >= 0.8).idxmax() + 1 if not cumsum.empty else 0
n_95 = (cumsum >= 0.95).idxmax() + 1 if not cumsum.empty else 0


# Add text box with statistics
textstr = f'Features for:\n50% importance: {n_50}\n80% importance: {n_80}\n95% importance: {n_95}\n\nTotal features: {len(feature_names)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.98, 0.02, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

fig.suptitle('XGBoost Feature Importance Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

print("   ✅ Visualization generated")

# Print summary statistics
print("\n" + "=" * 70)
print("📈 FEATURE IMPORTANCE SUMMARY (XGBoost)")
print("=" * 70)
print(f"Total Features: {len(feature_names)}")
print(f"Features capturing 50% importance: {n_50}")
print(f"Features capturing 80% importance: {n_80}")
print(f"Features capturing 95% importance: {n_95}")
print(f"\nTop 3 Features:")
for i, (idx, row) in enumerate(feature_importance_df.head(3).iterrows(), 1):
    print(f"  {i}. {row['Feature']:40s} - {row['Importance']*100:.2f}%")

print("\n" + "=" * 70)
print("✅ FEATURE IMPORTANCE ANALYSIS COMPLETE!")
print("=" * 70)
print("\n👁️  Displaying graphs... (close window to exit)")

plt.show()

print("\n🎉 Done! Thank you for using the fraud detection visualizer.")
print("=" * 70)
