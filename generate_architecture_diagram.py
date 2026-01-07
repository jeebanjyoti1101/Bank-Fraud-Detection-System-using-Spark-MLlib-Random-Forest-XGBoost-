"""
Generate System Architecture Diagram for Fraud Detection System
Creates a comprehensive visual representation of the system architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Fraud Detection System Architecture', 
        ha='center', va='center', fontsize=20, fontweight='bold')

# Colors
color_ui = '#3498db'
color_api = '#2ecc71'
color_processing = '#f39c12'
color_ml = '#e74c3c'
color_data = '#9b59b6'
color_output = '#1abc9c'

# =====================================================================
# LAYER 1: USER INTERFACE (Top)
# =====================================================================
# Web Interface
ui_box = FancyBboxPatch((0.5, 8), 2, 0.6, 
                        boxstyle="round,pad=0.05", 
                        edgecolor='black', facecolor=color_ui, linewidth=2)
ax.add_patch(ui_box)
ax.text(1.5, 8.3, 'Web Interface\n(HTML/CSS/JS)', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# REST API
api_box = FancyBboxPatch((3, 8), 2, 0.6, 
                        boxstyle="round,pad=0.05", 
                        edgecolor='black', facecolor=color_ui, linewidth=2)
ax.add_patch(api_box)
ax.text(4, 8.3, 'REST API\n(POST /api/predict)', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Mobile/External
mobile_box = FancyBboxPatch((5.5, 8), 2, 0.6, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor=color_ui, linewidth=2)
ax.add_patch(mobile_box)
ax.text(6.5, 8.3, 'External Systems\n(Mobile/API Clients)', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrow to Flask App
arrow1 = FancyArrowPatch((1.5, 8), (4, 7.2), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color='black')
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((4, 8), (4, 7.2), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color='black')
ax.add_patch(arrow2)
arrow3 = FancyArrowPatch((6.5, 8), (4, 7.2), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color='black')
ax.add_patch(arrow3)

# =====================================================================
# LAYER 2: APPLICATION SERVER
# =====================================================================
flask_box = FancyBboxPatch((2, 6.5), 4, 0.6, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color_api, linewidth=2)
ax.add_patch(flask_box)
ax.text(4, 6.8, 'Flask Application Server (app.py)\nPort: 5001 | Python 3.13', 
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Arrow to Feature Engineering
arrow4 = FancyArrowPatch((4, 6.5), (4, 5.9), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color='black')
ax.add_patch(arrow4)

# =====================================================================
# LAYER 3: DATA PROCESSING
# =====================================================================
# Feature Engineering
feature_box = FancyBboxPatch((1.5, 5.2), 5, 0.6, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color_processing, linewidth=2)
ax.add_patch(feature_box)
ax.text(4, 5.5, 'Feature Engineering Pipeline\n60 Engineered Features | StandardScaler Normalization', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Feature details boxes
features = [
    ('Amount\nFeatures', 0.8, 4.3),
    ('Balance\nFeatures', 2.2, 4.3),
    ('Time\nFeatures', 3.6, 4.3),
    ('Type\nFeatures', 5, 4.3),
    ('Risk\nIndicators', 6.4, 4.3)
]

for feat_name, x, y in features:
    feat_box = FancyBboxPatch((x, y), 1.1, 0.5, 
                             boxstyle="round,pad=0.03", 
                             edgecolor='black', facecolor='#ecf0f1', linewidth=1.5)
    ax.add_patch(feat_box)
    ax.text(x + 0.55, y + 0.25, feat_name, 
            ha='center', va='center', fontsize=7, fontweight='bold')

# Arrow from features to feature engineering
for feat_name, x, y in features:
    arrow = FancyArrowPatch((x + 0.55, y + 0.5), (4, 5.2), 
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color='gray', alpha=0.6)
    ax.add_patch(arrow)

# Arrow to ML Models
arrow5 = FancyArrowPatch((4, 5.2), (4, 3.9), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color='black')
ax.add_patch(arrow5)

# =====================================================================
# LAYER 4: MACHINE LEARNING MODELS
# =====================================================================
# ML Models Container
ml_container = FancyBboxPatch((0.3, 2.2), 7.4, 1.6, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='black', facecolor='#ecf0f1', 
                             linewidth=2, linestyle='--', alpha=0.3)
ax.add_patch(ml_container)
ax.text(4, 3.7, '5-Model Ensemble Architecture', 
        ha='center', va='center', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))

# Individual Models
models = [
    ('Random Forest\n300 Trees', 0.8, 2.5, '#5DADE2'),
    ('XGBoost\n300 Rounds', 2.1, 2.5, '#F39C12'),
    ('LightGBM\n300 Rounds', 3.4, 2.5, '#2ECC71'),
    ('CatBoost\n200 Rounds', 4.7, 2.5, '#E74C3C'),
    ('Gradient Boost\n200 Epochs', 6, 2.5, '#9B59B6')
]

for model_name, x, y, color in models:
    model_box = FancyBboxPatch((x, y), 1.2, 0.7, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(model_box)
    ax.text(x + 0.6, y + 0.35, model_name, 
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    # Arrow to ensemble
    arrow = FancyArrowPatch((x + 0.6, y), (4, 1.8), 
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color='black')
    ax.add_patch(arrow)

# =====================================================================
# LAYER 5: ENSEMBLE & DECISION
# =====================================================================
# Ensemble Aggregation
ensemble_box = FancyBboxPatch((2.5, 1.2), 3, 0.5, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='black', facecolor=color_ml, linewidth=2)
ax.add_patch(ensemble_box)
ax.text(4, 1.45, 'Weighted Ensemble Aggregation\nWeights: 0.20 each | Accuracy: 94.46%', 
        ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Arrow to output
arrow6 = FancyArrowPatch((4, 1.2), (4, 0.8), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color='black')
ax.add_patch(arrow6)

# =====================================================================
# LAYER 6: OUTPUT & RESPONSE
# =====================================================================
# Prediction Output
output_box = FancyBboxPatch((1.5, 0.2), 5, 0.5, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor=color_output, linewidth=2)
ax.add_patch(output_box)
ax.text(4, 0.45, 'Prediction Response\nFraud Probability | Individual Model Scores | Risk Assessment | Key Factors', 
        ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# =====================================================================
# SIDE PANEL: DATA STORAGE
# =====================================================================
# Models Storage
storage_box = FancyBboxPatch((8.2, 5), 1.5, 3, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color_data, linewidth=2)
ax.add_patch(storage_box)
ax.text(8.95, 7.5, 'Model\nStorage', 
        ha='center', va='top', fontsize=10, fontweight='bold', color='white')

storage_items = [
    'rf_model.pkl',
    'xgboost_model.pkl',
    'lightgbm_model.pkl',
    'catboost_model.pkl',
    'gb_model.pkl',
    'scaler.pkl',
    'encoders.pkl',
    'metadata.json'
]

y_pos = 7
for item in storage_items:
    ax.text(8.95, y_pos, f'• {item}', 
            ha='center', va='center', fontsize=6, color='white')
    y_pos -= 0.25

# Arrow from storage to models
arrow7 = FancyArrowPatch((8.2, 6.5), (7.7, 3.2), 
                        arrowstyle='<->', mutation_scale=15, 
                        linewidth=2, color='black', linestyle='--')
ax.add_patch(arrow7)

# =====================================================================
# SIDE PANEL: TRAINING DATA
# =====================================================================
data_box = FancyBboxPatch((8.2, 1.5), 1.5, 2.5, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor='#34495e', linewidth=2)
ax.add_patch(data_box)
ax.text(8.95, 3.7, 'Training\nData', 
        ha='center', va='top', fontsize=10, fontweight='bold', color='white')

data_items = [
    'Dataset:',
    'Fraud.csv',
    '',
    'Size:',
    '1,048,574 txns',
    '',
    'Fraud Rate:',
    '16.76%',
    '',
    'Features:',
    '11 raw',
    '60 engineered'
]

y_pos = 3.4
for item in data_items:
    fontweight = 'bold' if ':' in item else 'normal'
    ax.text(8.95, y_pos, item, 
            ha='center', va='center', fontsize=6, color='white', fontweight=fontweight)
    y_pos -= 0.17

# =====================================================================
# BOTTOM: TECHNOLOGY STACK
# =====================================================================
tech_box = FancyBboxPatch((0.2, 0.02), 7.6, 0.12, 
                         boxstyle="round,pad=0.01", 
                         edgecolor='black', facecolor='#2c3e50', linewidth=1)
ax.add_patch(tech_box)

tech_stack = 'Technology Stack: Python 3.13 | Flask 2.3.0 | Scikit-learn 1.1.0 | XGBoost 1.6.2 | LightGBM | CatBoost | Pandas | NumPy | Matplotlib | Bootstrap 5'
ax.text(4, 0.08, tech_stack, 
        ha='center', va='center', fontsize=6, color='white', fontweight='bold')

# =====================================================================
# LEGEND
# =====================================================================
legend_elements = [
    mpatches.Patch(facecolor=color_ui, edgecolor='black', label='User Interface'),
    mpatches.Patch(facecolor=color_api, edgecolor='black', label='Application Layer'),
    mpatches.Patch(facecolor=color_processing, edgecolor='black', label='Data Processing'),
    mpatches.Patch(facecolor=color_ml, edgecolor='black', label='ML Models/Ensemble'),
    mpatches.Patch(facecolor=color_data, edgecolor='black', label='Storage'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output/Response')
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=7, 
         framealpha=0.9, edgecolor='black', title='Components', title_fontsize=8)

# Save
plt.tight_layout()
plt.savefig('graphs/system_architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ System Architecture Diagram saved: graphs/system_architecture_diagram.png")
plt.close()

print("\n📐 Architecture Diagram Generated!")
print("\n🏗️ System Components:")
print("  Layer 1: User Interface (Web, API, External)")
print("  Layer 2: Flask Application Server")
print("  Layer 3: Feature Engineering Pipeline (60 features)")
print("  Layer 4: 5-Model ML Ensemble")
print("  Layer 5: Weighted Ensemble Aggregation")
print("  Layer 6: Prediction Output & Response")
print("  Side: Model Storage & Training Data")
print("\n📊 Key Metrics:")
print("  • 5 ML Models: RF, XGBoost, LightGBM, CatBoost, Gradient Boosting")
print("  • Ensemble Accuracy: 94.46%")
print("  • Training Data: 1,048,574 transactions")
print("  • 60 Engineered Features")
print("\n📁 File: graphs/system_architecture_diagram.png")
