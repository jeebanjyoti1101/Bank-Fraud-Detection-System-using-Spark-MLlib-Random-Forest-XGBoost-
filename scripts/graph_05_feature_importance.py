"""
Graph 5: Feature Importance
Shows most important features for fraud detection
"""
import matplotlib.pyplot as plt
import joblib
import numpy as np
import json

print("📈 Loading models...")

models = {}
for name, path in [('rf', 'models/rf_model.pkl'), ('xgb', 'models/xgboost_model.pkl'), 
                   ('lgb', 'models/lightgbm_model.pkl'), ('cat', 'models/catboost_model.pkl')]:
    try:
        models[name] = joblib.load(path)
        print(f"✅ Loaded {name.upper()}")
    except:
        pass

with open('models/advanced_metadata.json', 'r') as f:
    feature_cols = json.load(f).get("feature_names", [])

print("\n📈 Generating Feature Importance plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, (name, model) in enumerate(models.items()):
    if idx >= 4:
        break
    
    ax = axes[idx]
    try:
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_score'):
            importance_dict = model.get_score(importance_type='weight')
            importances = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_cols))]
        else:
            importances = np.zeros(len(feature_cols))
        
        # Get top 15 features
        indices = np.argsort(importances)[-15:]
        top_features = [feature_cols[i] if i < len(feature_cols) else f'Feature {i}' for i in indices]
        top_importances = [importances[i] for i in indices]
        
        # Plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax.barh(range(len(top_features)), top_importances, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Top 15 Features - {name.upper()}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        print(f"  ✅ {name.upper()} feature importance generated")
    except Exception as e:
        ax.text(0.5, 0.5, f'Error\n{name.upper()}', ha='center', va='center', fontsize=12)
        print(f"  ⚠️ Error with {name.upper()}: {e}")

plt.tight_layout()
print("\n✅ Displaying feature importance...")
plt.show()
