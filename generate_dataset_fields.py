"""
Generate Dataset Fields Visualization
Shows all fields with descriptions, data types, and sample values
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Load dataset
df = pd.read_csv('data/Fraud.csv')

# Create figure
fig = plt.figure(figsize=(16, 12))

# Create main grid
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# =====================================================================
# TITLE
# =====================================================================
fig.suptitle('Fraud Detection Dataset - Complete Field Analysis', 
             fontsize=20, fontweight='bold', y=0.98)

# =====================================================================
# LEFT PANEL: FIELD DESCRIPTIONS
# =====================================================================
ax1 = fig.add_subplot(gs[:, 0])
ax1.axis('off')

# Dataset overview box
overview_box = FancyBboxPatch((0.05, 0.92), 0.9, 0.06, 
                             boxstyle="round,pad=0.01", 
                             edgecolor='black', facecolor='#3498db', linewidth=2)
ax1.add_patch(overview_box)
ax1.text(0.5, 0.95, f'Dataset Overview: {df.shape[0]:,} Transactions | {df.shape[1]} Fields', 
         ha='center', va='center', fontsize=12, fontweight='bold', 
         transform=ax1.transAxes, color='white')

# Field details
fields_info = [
    {
        'name': '1. Date',
        'type': 'Object (String)',
        'description': 'Transaction date (DD-MMM-YY format)',
        'example': df['Date'].iloc[0],
        'color': '#e74c3c'
    },
    {
        'name': '2. nameOrig',
        'type': 'Object (String)',
        'description': 'Customer ID/Name (Origin account identifier)',
        'example': df['nameOrig'].iloc[0],
        'color': '#e74c3c'
    },
    {
        'name': '3. amount',
        'type': 'Float64 (Numeric)',
        'description': 'Transaction amount in currency',
        'example': f"${df['amount'].iloc[0]:,.2f}",
        'color': '#2ecc71'
    },
    {
        'name': '4. oldbalanceOrg',
        'type': 'Float64 (Numeric)',
        'description': 'Initial balance before transaction (Origin)',
        'example': f"${df['oldbalanceOrg'].iloc[0]:,.2f}",
        'color': '#2ecc71'
    },
    {
        'name': '5. newbalanceOrig',
        'type': 'Float64 (Numeric)',
        'description': 'New balance after transaction (Origin)',
        'example': f"${df['newbalanceOrig'].iloc[0]:,.2f}",
        'color': '#2ecc71'
    },
    {
        'name': '6. City',
        'type': 'Object (Categorical)',
        'description': 'City where transaction occurred',
        'example': df['City'].iloc[0],
        'color': '#9b59b6'
    },
    {
        'name': '7. type',
        'type': 'Object (Categorical)',
        'description': 'Transaction type (Payment/Transfer/etc.)',
        'example': df['type'].iloc[0],
        'color': '#9b59b6'
    },
    {
        'name': '8. Card Type',
        'type': 'Object (Categorical)',
        'description': 'Type of card used (Credit/Debit)',
        'example': df['Card Type'].iloc[0],
        'color': '#9b59b6'
    },
    {
        'name': '9. Exp Type',
        'type': 'Object (Categorical)',
        'description': 'Expense category (Food/Entertainment/etc.)',
        'example': df['Exp Type'].iloc[0],
        'color': '#9b59b6'
    },
    {
        'name': '10. Gender',
        'type': 'Object (Categorical)',
        'description': 'Gender of account holder (M/F)',
        'example': df['Gender'].iloc[0],
        'color': '#9b59b6'
    },
    {
        'name': '11. isFraud',
        'type': 'Int64 (Binary)',
        'description': 'Target variable: 1=Fraud, 0=Legitimate',
        'example': str(df['isFraud'].iloc[0]),
        'color': '#f39c12'
    }
]

y_pos = 0.85
for field in fields_info:
    # Field box
    field_box = FancyBboxPatch((0.05, y_pos - 0.055), 0.9, 0.05, 
                              boxstyle="round,pad=0.005", 
                              edgecolor='black', facecolor='white', linewidth=1.5)
    ax1.add_patch(field_box)
    
    # Field name with color indicator
    color_indicator = Rectangle((0.06, y_pos - 0.045), 0.015, 0.03, 
                               facecolor=field['color'], edgecolor='black', linewidth=1)
    ax1.add_patch(color_indicator)
    
    ax1.text(0.09, y_pos - 0.01, field['name'], 
             ha='left', va='top', fontsize=9, fontweight='bold', 
             transform=ax1.transAxes)
    
    ax1.text(0.09, y_pos - 0.025, f"Type: {field['type']}", 
             ha='left', va='top', fontsize=7, style='italic',
             transform=ax1.transAxes, color='#555')
    
    ax1.text(0.09, y_pos - 0.038, field['description'], 
             ha='left', va='top', fontsize=7,
             transform=ax1.transAxes)
    
    ax1.text(0.09, y_pos - 0.05, f"Example: {field['example']}", 
             ha='left', va='top', fontsize=6.5, style='italic',
             transform=ax1.transAxes, color='#888')
    
    y_pos -= 0.072

# =====================================================================
# TOP RIGHT: DATA TYPE DISTRIBUTION
# =====================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('Data Types Distribution', fontsize=12, fontweight='bold', pad=10)

data_types = df.dtypes.value_counts()
colors_pie = ['#3498db', '#e74c3c', '#2ecc71']
wedges, texts, autotexts = ax2.pie(data_types.values, labels=data_types.index.astype(str), 
                                     autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                     textprops={'fontsize': 9, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')

# =====================================================================
# MIDDLE RIGHT: FIELD CATEGORIES
# =====================================================================
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
ax3.set_title('Field Categories', fontsize=12, fontweight='bold', pad=10)

categories = {
    'Temporal': ['Date'],
    'Identifier': ['nameOrig'],
    'Financial': ['amount', 'oldbalanceOrg', 'newbalanceOrig'],
    'Location': ['City'],
    'Transaction Info': ['type', 'Card Type', 'Exp Type'],
    'Demographic': ['Gender'],
    'Target': ['isFraud']
}

cat_colors = {
    'Temporal': '#e74c3c',
    'Identifier': '#e74c3c',
    'Financial': '#2ecc71',
    'Location': '#9b59b6',
    'Transaction Info': '#9b59b6',
    'Demographic': '#9b59b6',
    'Target': '#f39c12'
}

y_pos = 0.85
for cat_name, fields in categories.items():
    cat_box = FancyBboxPatch((0.1, y_pos - 0.08), 0.8, 0.075, 
                            boxstyle="round,pad=0.01", 
                            edgecolor='black', facecolor=cat_colors[cat_name], 
                            linewidth=2, alpha=0.7)
    ax3.add_patch(cat_box)
    
    ax3.text(0.15, y_pos - 0.025, cat_name, 
             ha='left', va='center', fontsize=10, fontweight='bold', 
             transform=ax3.transAxes, color='white')
    
    ax3.text(0.15, y_pos - 0.055, ', '.join(fields), 
             ha='left', va='center', fontsize=7,
             transform=ax3.transAxes, color='white')
    
    y_pos -= 0.12

# =====================================================================
# BOTTOM RIGHT: STATISTICS
# =====================================================================
ax4 = fig.add_subplot(gs[2, 1])
ax4.axis('off')
ax4.set_title('Dataset Statistics', fontsize=12, fontweight='bold', pad=10)

stats = [
    ('Total Records', f"{df.shape[0]:,}"),
    ('Total Fields', f"{df.shape[1]}"),
    ('Fraud Cases', f"{df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)"),
    ('Legitimate', f"{(df['isFraud']==0).sum():,} ({(1-df['isFraud'].mean())*100:.2f}%)"),
    ('Unique Customers', f"{df['nameOrig'].nunique():,}"),
    ('Unique Cities', f"{df['City'].nunique()}"),
    ('Transaction Types', f"{df['type'].nunique()}"),
    ('Card Types', f"{df['Card Type'].nunique()}"),
    ('Expense Types', f"{df['Exp Type'].nunique()}"),
    ('Avg Transaction', f"${df['amount'].mean():,.2f}"),
    ('Max Transaction', f"${df['amount'].max():,.2f}"),
    ('Min Transaction', f"${df['amount'].min():,.2f}"),
]

y_pos = 0.9
for i, (label, value) in enumerate(stats):
    bg_color = '#ecf0f1' if i % 2 == 0 else 'white'
    stat_box = Rectangle((0.1, y_pos - 0.06), 0.8, 0.055, 
                         facecolor=bg_color, edgecolor='black', linewidth=1)
    ax4.add_patch(stat_box)
    
    ax4.text(0.15, y_pos - 0.03, label, 
             ha='left', va='center', fontsize=8, fontweight='bold',
             transform=ax4.transAxes)
    
    ax4.text(0.85, y_pos - 0.03, value, 
             ha='right', va='center', fontsize=8,
             transform=ax4.transAxes, color='#2c3e50', fontweight='bold')
    
    y_pos -= 0.07

# Add legend at bottom
legend_elements = [
    mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='String/Temporal'),
    mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Numeric'),
    mpatches.Patch(facecolor='#9b59b6', edgecolor='black', label='Categorical'),
    mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='Target Variable')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
          fontsize=9, framealpha=0.9, edgecolor='black', 
          bbox_to_anchor=(0.5, 0.01))

# Save
plt.savefig('graphs/dataset_fields_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Dataset Fields Analysis Generated!")
print("📁 File: graphs/dataset_fields_analysis.png")

print("\n" + "="*70)
print("DATASET FIELD SUMMARY")
print("="*70)
print(f"\n📊 Total Records: {df.shape[0]:,}")
print(f"📋 Total Fields: {df.shape[1]}")
print(f"\n🔍 Field Breakdown:")
print(f"  • Temporal/Identifier: 2 fields (Date, nameOrig)")
print(f"  • Financial: 3 fields (amount, oldbalanceOrg, newbalanceOrig)")
print(f"  • Location: 1 field (City)")
print(f"  • Transaction Info: 3 fields (type, Card Type, Exp Type)")
print(f"  • Demographic: 1 field (Gender)")
print(f"  • Target: 1 field (isFraud)")

print(f"\n📈 Class Distribution:")
print(f"  • Fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
print(f"  • Legitimate: {(df['isFraud']==0).sum():,} ({(1-df['isFraud'].mean())*100:.2f}%)")

print(f"\n💰 Transaction Statistics:")
print(f"  • Average: ${df['amount'].mean():,.2f}")
print(f"  • Maximum: ${df['amount'].max():,.2f}")
print(f"  • Minimum: ${df['amount'].min():,.2f}")
print(f"  • Median: ${df['amount'].median():,.2f}")

print("\n" + "="*70)
print("\n🎉 Dataset fields visualization complete!")

plt.close()

# =====================================================================
# GENERATE DETAILED FIELD TABLE
# =====================================================================
print("\n📝 Generating detailed field table...")

fig2, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

# Title
ax.text(0.5, 0.97, 'Fraud Detection Dataset - Field Reference Table', 
        ha='center', va='center', fontsize=16, fontweight='bold', 
        transform=ax.transAxes)

# Table data
table_data = []
table_data.append(['#', 'Field Name', 'Data Type', 'Description', 'Unique Values', 'Sample Values'])

for i, col in enumerate(df.columns, 1):
    field_name = col
    data_type = str(df[col].dtype)
    
    # Description based on field
    descriptions = {
        'Date': 'Transaction date',
        'nameOrig': 'Customer ID (Origin account)',
        'amount': 'Transaction amount',
        'oldbalanceOrg': 'Balance before transaction',
        'newbalanceOrig': 'Balance after transaction',
        'City': 'Transaction city',
        'type': 'Transaction type',
        'Card Type': 'Card type used',
        'Exp Type': 'Expense category',
        'Gender': 'Account holder gender',
        'isFraud': 'Fraud indicator (0/1)'
    }
    description = descriptions.get(col, 'N/A')
    
    unique_vals = df[col].nunique()
    
    # Sample values
    if df[col].dtype == 'object':
        samples = ', '.join(df[col].unique()[:3].astype(str))
    else:
        samples = ', '.join(df[col].head(3).astype(str))
    
    if len(samples) > 40:
        samples = samples[:37] + '...'
    
    table_data.append([str(i), field_name, data_type, description, str(unique_vals), samples])

# Create table
table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.05, 0.15, 0.12, 0.25, 0.1, 0.33])

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontsize=9)

# Style data rows
for i in range(1, len(table_data)):
    for j in range(6):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        else:
            cell.set_facecolor('white')
        
        # Color code by field type
        if j == 1:  # Field name column
            if table_data[i][2] in ['object']:
                cell.set_text_props(color='#e74c3c', weight='bold')
            elif 'float' in table_data[i][2] or 'int' in table_data[i][2]:
                cell.set_text_props(color='#2ecc71', weight='bold')

# Add footer
ax.text(0.5, 0.02, f'Dataset: Fraud.csv | Total Records: {df.shape[0]:,} | Total Fields: {df.shape[1]} | Fraud Rate: {df["isFraud"].mean()*100:.2f}%', 
        ha='center', va='center', fontsize=9, style='italic',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))

plt.tight_layout()
plt.savefig('graphs/dataset_field_reference_table.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Field Reference Table Generated!")
print("📁 File: graphs/dataset_field_reference_table.png")
print("\n🎉 All dataset field visualizations complete!")

plt.close()
