"""
Graph 6: Transaction Type Distribution
Shows distribution of fraud by transaction type
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("📈 Loading data...")
df = pd.read_csv('data/Fraud.csv')
df.columns = df.columns.str.strip()

print("\n📈 Generating Transaction Type Distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Transaction type counts
type_counts = df['type'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
bars = ax1.bar(range(len(type_counts)), type_counts.values, color=colors)
ax1.set_xticks(range(len(type_counts)))
ax1.set_xticklabels(type_counts.index, rotation=45, ha='right', fontsize=12)
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Transaction Type Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(type_counts.values):
    ax1.text(i, v + max(type_counts.values) * 0.01, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

# Plot 2: Fraud rate by transaction type
if 'isFraud' in df.columns:
    fraud_by_type = df.groupby('type')['isFraud'].agg(['sum', 'count'])
    fraud_by_type['fraud_rate'] = (fraud_by_type['sum'] / fraud_by_type['count'] * 100)
    
    bars = ax2.bar(range(len(fraud_by_type)), fraud_by_type['fraud_rate'].values, color=colors)
    ax2.set_xticks(range(len(fraud_by_type)))
    ax2.set_xticklabels(fraud_by_type.index, rotation=45, ha='right', fontsize=12)
    ax2.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Fraud Rate by Transaction Type', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(fraud_by_type['fraud_rate'].values):
        ax2.text(i, v + max(fraud_by_type['fraud_rate'].values) * 0.01, f'{v:.2f}%', 
                ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
print("✅ Displaying transaction type distribution...")
plt.show()
