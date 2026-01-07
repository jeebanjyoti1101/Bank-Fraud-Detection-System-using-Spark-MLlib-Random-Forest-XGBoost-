"""
Graph 7: Amount Distribution
Shows distribution of transaction amounts
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("📈 Loading data...")
df = pd.read_csv('data/Fraud.csv')
df.columns = df.columns.str.strip()
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

print("\n📈 Generating Amount Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Amount distribution (log scale)
ax1 = axes[0, 0]
amounts = df[df['amount'] > 0]['amount']
ax1.hist(np.log10(amounts + 1), bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Log10(Amount + 1)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Transaction Amount Distribution (Log Scale)', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Fraud vs Non-fraud amounts
if 'isFraud' in df.columns:
    ax2 = axes[0, 1]
    fraud_amounts = df[df['isFraud'] == 1]['amount']
    normal_amounts = df[df['isFraud'] == 0]['amount'].sample(min(10000, len(df[df['isFraud'] == 0])))
    
    ax2.hist(np.log10(normal_amounts + 1), bins=30, alpha=0.5, label='Normal', color='green', edgecolor='black')
    ax2.hist(np.log10(fraud_amounts + 1), bins=30, alpha=0.5, label='Fraud', color='red', edgecolor='black')
    ax2.set_xlabel('Log10(Amount + 1)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Amount Distribution: Fraud vs Normal', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

# Plot 3: Box plot by transaction type
ax3 = axes[1, 0]
types = df['type'].unique()[:5]
data_by_type = [df[df['type'] == t]['amount'].values for t in types]
bp = ax3.boxplot(data_by_type, labels=types, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(types)))):
    patch.set_facecolor(color)
ax3.set_xlabel('Transaction Type', fontsize=12, fontweight='bold')
ax3.set_ylabel('Amount', fontsize=12, fontweight='bold')
ax3.set_title('Amount Distribution by Transaction Type', fontsize=14, fontweight='bold')
ax3.set_xticklabels(types, rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Cumulative distribution
ax4 = axes[1, 1]
sorted_amounts = np.sort(amounts.sample(min(10000, len(amounts))))
cdf = np.arange(1, len(sorted_amounts) + 1) / len(sorted_amounts)
ax4.plot(sorted_amounts, cdf, linewidth=2, color='#FF6B6B')
ax4.set_xlabel('Amount', fontsize=12, fontweight='bold')
ax4.set_ylabel('CDF', fontsize=12, fontweight='bold')
ax4.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
print("✅ Displaying amount distribution...")
plt.show()
