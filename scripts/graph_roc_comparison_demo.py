"""
ROC Curve Comparison - All Models (Demo/Simulated Data)
Displays ROC curves for RF, XGBoost, LightGBM, and CatBoost
"""
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("📈 ROC Curve Comparison (Simulated Data)")
print("=" * 70)

# Generate simulated ROC curve data
print("\n📊 Generating simulated ROC curves...")

np.random.seed(42)

# Create smooth ROC curves with AUC between 0.90-0.92
def generate_roc_curve(auc_score):
    """Generate a smooth ROC curve for given AUC score"""
    # Create FPR points
    fpr = np.linspace(0, 1, 100)
    
    # Generate TPR to achieve desired AUC
    # Use a smooth sigmoid-like curve
    tpr = []
    for f in fpr:
        # Create a curve that stays above the diagonal
        base_tpr = f + (auc_score - 0.5) * 2 * (1 - f)
        # Add some natural variation
        noise = np.random.normal(0, 0.01)
        tpr_val = min(1.0, max(f, base_tpr + noise))
        tpr.append(tpr_val)
    
    tpr = np.array(tpr)
    # Ensure it starts at 0,0 and ends at 1,1
    tpr[0] = 0.0
    tpr[-1] = 1.0
    
    # Smooth the curve
    from scipy.ndimage import gaussian_filter1d
    tpr = gaussian_filter1d(tpr, sigma=2)
    
    return fpr, tpr

# Model data with AUC between 0.90-0.92
models_data = [
    {"name": "Random Forest", "auc": 0.9145, "color": "#90CAF9"},
    {"name": "XGBoost", "auc": 0.9198, "color": "#A5D6A7"},
    {"name": "LightGBM", "auc": 0.9167, "color": "#FFCC80"},
    {"name": "CatBoost", "auc": 0.9178, "color": "#CE93D8"}
]

print("   ✅ ROC curves generated for 4 models")

# Create visualization
print("\n📈 Creating ROC curve comparison plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot ROC curve for each model
for model in models_data:
    fpr, tpr = generate_roc_curve(model["auc"])
    ax.plot(fpr, tpr, linewidth=3, label=f'{model["name"]} (AUC = {model["auc"]:.4f})', 
            color=model["color"], alpha=0.8)

# Plot diagonal line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5000)', alpha=0.6)

# Styling
ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curve Comparison - All Models', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()

print("   ✅ ROC curve comparison plot created")
print("\n" + "=" * 70)
print("✅ ROC CURVE GENERATION COMPLETE!")
print("=" * 70)
print("\n📌 NOTE: This is simulated/demo data for visualization purposes.")
print("👁️  Displaying graph... (close window to exit)")

plt.show()

print("\n🎉 Done! Thank you for using the fraud detection visualizer.")
print("=" * 70)
