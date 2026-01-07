"""
Model Performance Comparison (Demo/Simulated Data)
Shows simulated performance metrics with all models above 90%
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("📊 Model Performance Comparison (Simulated Data)")
print("=" * 70)

# --- Generate Simulated Performance Data ---
print("\n📈 Generating simulated performance metrics...")

# Simulated data with all metrics between 90-92%
performance_data = [
    {
        "Model": "Random Forest",
        "AUC": 0.9145,
        "F1-Score": 0.9087,
        "Precision": 0.9123,
        "Recall": 0.9056,
        "Accuracy": 0.9178
    },
    {
        "Model": "XGBoost",
        "AUC": 0.9198,
        "F1-Score": 0.9134,
        "Precision": 0.9167,
        "Recall": 0.9112,
        "Accuracy": 0.9189
    },
    {
        "Model": "LightGBM",
        "AUC": 0.9167,
        "F1-Score": 0.9098,
        "Precision": 0.9145,
        "Recall": 0.9089,
        "Accuracy": 0.9156
    },
    {
        "Model": "CatBoost",
        "AUC": 0.9178,
        "F1-Score": 0.9156,
        "Precision": 0.9178,
        "Recall": 0.9134,
        "Accuracy": 0.9195
    }
]

performance_df = pd.DataFrame(performance_data).set_index('Model')

print("\n" + "=" * 70)
print("🏆 SIMULATED PERFORMANCE METRICS")
print("=" * 70)
print(performance_df.to_string(float_format="%.4f"))
print("=" * 70)

# --- Generate Visualization ---
print("\n📈 Generating performance comparison chart...")

fig, ax = plt.subplots(figsize=(14, 8))

# Define gentle, light document-friendly colors for each metric
colors = ['#90CAF9', '#A5D6A7', '#FFCC80', '#CE93D8', '#80DEEA']
performance_df.plot(kind='bar', ax=ax, color=colors, width=0.8)

# Styling the chart
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=0, labelsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

# Set y-axis to start from 0 as normal
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for legend

print("   ✅ Chart generated.")
print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE!")
print("=" * 70)
print("\n📌 NOTE: This is simulated/demo data for visualization purposes.")
print("👁️  Displaying graph... (close window to exit)")

plt.show()

print("\n🎉 Done! Thank you for using the fraud detection visualizer.")
print("=" * 70)
