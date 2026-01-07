"""
Generate Training Performance Dashboard
Creates a 2x2 dashboard showing Accuracy, Precision, Recall, and F1-Score comparisons
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# Model performance data (including Gradient Boosting)
models = ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Gradient Boosting']
accuracy = [93.12, 93.98, 94.25, 94.01, 93.87]
precision = [92.45, 93.21, 93.56, 93.34, 92.98]
recall = [91.56, 92.76, 93.12, 92.89, 92.45]
f1_score = [92.00, 92.98, 93.34, 93.11, 92.71]

# Colors for each model
colors = ['#5DADE2', '#F39C12', '#2ECC71', '#E74C3C', '#9B59B6']

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Training Performance Dashboard', fontsize=20, fontweight='bold', y=0.995)

# =====================================================================
# 1. Model Accuracy Comparison (Top Left)
# =====================================================================
ax1 = axes[0, 0]
bars1 = ax1.barh(models, accuracy, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, accuracy)):
    ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')

ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlim([88, 95])
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# =====================================================================
# 2. Model Precision Comparison (Top Right)
# =====================================================================
ax2 = axes[0, 1]
bars2 = ax2.barh(models, precision, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, precision)):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')

ax2.set_xlabel('Precision (%)', fontsize=12, fontweight='bold')
ax2.set_title('Model Precision Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlim([86, 95])
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# =====================================================================
# 3. Model Recall Comparison (Bottom Left)
# =====================================================================
ax3 = axes[1, 0]
bars3 = ax3.barh(models, recall, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars3, recall)):
    ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')

ax3.set_xlabel('Recall (%)', fontsize=12, fontweight='bold')
ax3.set_title('Model Recall Comparison', fontsize=14, fontweight='bold', pad=15)
ax3.set_xlim([85, 94])
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

# =====================================================================
# 4. Model F1-Score Comparison (Bottom Right)
# =====================================================================
ax4 = axes[1, 1]
bars4 = ax4.barh(models, f1_score, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars4, f1_score)):
    ax4.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')

ax4.set_xlabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax4.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold', pad=15)
ax4.set_xlim([85, 94])
ax4.grid(axis='x', alpha=0.3)
ax4.invert_yaxis()

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save the dashboard
plt.savefig('graphs/training_performance_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Training Performance Dashboard saved: graphs/training_performance_dashboard.png")
plt.close()

print("\n📊 Dashboard Details:")
print(f"  • Models included: {len(models)}")
print(f"  • Metrics displayed: Accuracy, Precision, Recall, F1-Score")
print(f"  • Best Overall: LightGBM (94.25% accuracy)")
print(f"  • File: graphs/training_performance_dashboard.png")
print("\n🎉 Dashboard generation complete!")
