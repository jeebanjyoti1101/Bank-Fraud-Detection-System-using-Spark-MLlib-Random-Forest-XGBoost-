"""
Generate Complete Workflow Diagram for Fraud Detection System
Shows end-to-end process from data collection to prediction output
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
color_input = '#3498db'
color_process = '#2ecc71'
color_train = '#f39c12'
color_predict = '#e74c3c'
color_output = '#9b59b6'
color_storage = '#1abc9c'

# Title
title_box = FancyBboxPatch((5, 13), 8, 0.6, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor='#2c3e50', linewidth=3)
ax.add_patch(title_box)
ax.text(9, 13.3, 'Fraud Detection System - Complete Workflow', 
        ha='center', va='center', fontsize=18, fontweight='bold', color='white')

# =====================================================================
# PHASE 1: DATA COLLECTION & PREPARATION
# =====================================================================
ax.text(2, 12.3, 'PHASE 1: DATA COLLECTION', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black'))

# Raw Data Sources
sources = [
    ('Transaction\nRecords', 0.5, 11),
    ('User\nBehavior', 2.5, 11),
    ('Account\nHistory', 4.5, 11)
]

for label, x, y in sources:
    box = FancyBboxPatch((x, y), 1.5, 0.8, 
                        boxstyle="round,pad=0.05", 
                        edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(box)
    ax.text(x + 0.75, y + 0.4, label, ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Arrow down
    arrow = FancyArrowPatch((x + 0.75, y), (x + 0.75, 10), 
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)

# Data Integration
integration_box = FancyBboxPatch((0.5, 9.2), 5.5, 0.7, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='black', facecolor=color_process, linewidth=2)
ax.add_patch(integration_box)
ax.text(3.25, 9.55, 'Data Integration & Consolidation\nDataset: Fraud.csv (1,048,574 transactions)', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrow to preprocessing
arrow1 = FancyArrowPatch((3.25, 9.2), (3.25, 8.5), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow1)

# =====================================================================
# PHASE 2: DATA PREPROCESSING
# =====================================================================
ax.text(2, 8.7, 'PHASE 2: PREPROCESSING', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black'))

# Preprocessing steps
preprocess_steps = [
    ('Data\nCleaning', 0.5, 7.5),
    ('Missing\nValues', 2, 7.5),
    ('Outlier\nDetection', 3.5, 7.5),
    ('Normalization', 5, 7.5)
]

for label, x, y in preprocess_steps:
    box = FancyBboxPatch((x, y), 1.3, 0.6, 
                        boxstyle="round,pad=0.05", 
                        edgecolor='black', facecolor=color_process, linewidth=2)
    ax.add_patch(box)
    ax.text(x + 0.65, y + 0.3, label, ha='center', va='center', 
            fontsize=8, fontweight='bold', color='white')
    
    # Arrow down
    arrow = FancyArrowPatch((x + 0.65, y), (x + 0.65, 6.7), 
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color='black')
    ax.add_patch(arrow)

# Feature Engineering
feature_box = FancyBboxPatch((0.5, 6.0), 5.5, 0.6, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color_train, linewidth=2)
ax.add_patch(feature_box)
ax.text(3.25, 6.3, 'Feature Engineering: 60 Features Created\n(Amount, Balance, Time, Type, Risk Indicators)', 
        ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Arrow to training
arrow2 = FancyArrowPatch((3.25, 6.0), (3.25, 5.3), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow2)

# =====================================================================
# PHASE 3: MODEL TRAINING
# =====================================================================
ax.text(2, 5.5, 'PHASE 3: MODEL TRAINING', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))

# Data Split
split_box = FancyBboxPatch((0.5, 4.5), 5.5, 0.5, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(split_box)
ax.text(3.25, 4.75, 'Train-Test Split: 80% Training / 20% Testing', 
        ha='center', va='center', fontsize=9, fontweight='bold')

# Training arrows
arrow3a = FancyArrowPatch((1.5, 4.5), (1.5, 3.8), 
                         arrowstyle='->', mutation_scale=15, 
                         linewidth=2, color='black')
ax.add_patch(arrow3a)
ax.text(0.8, 4.1, 'Train', fontsize=8, fontweight='bold')

arrow3b = FancyArrowPatch((5, 4.5), (5, 3.8), 
                         arrowstyle='->', mutation_scale=15, 
                         linewidth=2, color='black')
ax.add_patch(arrow3b)
ax.text(5.5, 4.1, 'Test', fontsize=8, fontweight='bold')

# Individual Models
models_training = [
    ('Random Forest\n300 Trees\n93.12% Acc', 0.3, 2.8, '#5DADE2'),
    ('XGBoost\n300 Rounds\n93.98% Acc', 1.5, 2.8, '#F39C12'),
    ('LightGBM\n300 Rounds\n94.25% Acc', 2.7, 2.8, '#2ECC71'),
    ('CatBoost\n200 Rounds\n94.01% Acc', 3.9, 2.8, '#E74C3C'),
    ('Gradient Boost\n200 Epochs\n93.87% Acc', 5.1, 2.8, '#9B59B6')
]

for label, x, y, color in models_training:
    box = FancyBboxPatch((x, y), 1.1, 0.9, 
                        boxstyle="round,pad=0.05", 
                        edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + 0.55, y + 0.45, label, ha='center', va='center', 
            fontsize=7, fontweight='bold', color='white')
    
    # Arrow to ensemble
    arrow = FancyArrowPatch((x + 0.55, y), (3.25, 2.0), 
                           arrowstyle='->', mutation_scale=12, 
                           linewidth=1.5, color='black')
    ax.add_patch(arrow)

# Model Saving
save_box = FancyBboxPatch((7, 3.2), 1.8, 1.2, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color_storage, linewidth=2)
ax.add_patch(save_box)
ax.text(7.9, 4.1, 'Model Storage', ha='center', va='center', 
        fontsize=9, fontweight='bold', color='white')
save_files = ['rf_model.pkl', 'xgboost_model.pkl', 'lightgbm_model.pkl', 
              'catboost_model.pkl', 'gb_model.pkl']
y_pos = 3.8
for f in save_files:
    ax.text(7.9, y_pos, f'• {f}', ha='center', va='center', 
            fontsize=6, color='white')
    y_pos -= 0.15

# Arrow from models to storage
arrow4 = FancyArrowPatch((6.2, 3.3), (7.0, 3.8), 
                        arrowstyle='->', mutation_scale=15, 
                        linewidth=2, color='black', linestyle='--')
ax.add_patch(arrow4)

# Ensemble
ensemble_box = FancyBboxPatch((1.5, 1.3), 3.5, 0.6, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='black', facecolor=color_predict, linewidth=3)
ax.add_patch(ensemble_box)
ax.text(3.25, 1.6, 'Weighted Ensemble (0.20 each)\nAccuracy: 94.46% | AUC: 0.9487', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrow to deployment
arrow5 = FancyArrowPatch((5, 1.6), (7, 1.6), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=3, color='black')
ax.add_patch(arrow5)

# =====================================================================
# PHASE 4: DEPLOYMENT
# =====================================================================
ax.text(11, 12.3, 'PHASE 4: DEPLOYMENT', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black'))

# Flask Application
flask_box = FancyBboxPatch((9.5, 10.8), 3, 1, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_process, linewidth=3)
ax.add_patch(flask_box)
ax.text(11, 11.5, 'Flask Application Server', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='white')
ax.text(11, 11.15, 'app.py | Port: 5001 | Python 3.13', ha='center', va='center', 
        fontsize=8, color='white')

# Load Models
arrow6 = FancyArrowPatch((7.9, 4.4), (10.5, 10.8), 
                        arrowstyle='->', mutation_scale=15, 
                        linewidth=2, color='black', linestyle='--')
ax.add_patch(arrow6)
ax.text(9, 7.5, 'Load\nModels', ha='center', va='center', 
        fontsize=8, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

# Arrow down to API
arrow7 = FancyArrowPatch((11, 10.8), (11, 10), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow7)

# API Endpoints
api_box = FancyBboxPatch((9.5, 9.2), 3, 0.7, 
                        boxstyle="round,pad=0.05", 
                        edgecolor='black', facecolor=color_output, linewidth=2)
ax.add_patch(api_box)
ax.text(11, 9.7, 'REST API Endpoints', ha='center', va='center', 
        fontsize=9, fontweight='bold', color='white')
ax.text(11, 9.4, 'POST /api/predict | GET /', ha='center', va='center', 
        fontsize=7, color='white')

# =====================================================================
# PHASE 5: PREDICTION WORKFLOW
# =====================================================================
ax.text(11, 8.7, 'PHASE 5: REAL-TIME PREDICTION', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lavender', edgecolor='black'))

# User Input
user_box = FancyBboxPatch((9.5, 7.5), 3, 0.9, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color_input, linewidth=2)
ax.add_patch(user_box)
ax.text(11, 8.15, 'User Input Transaction', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='white')
inputs = ['Amount', 'Type', 'Old Balance', 'New Balance']
ax.text(11, 7.75, ' | '.join(inputs), ha='center', va='center', 
        fontsize=7, color='white')

# Arrow to processing
arrow8 = FancyArrowPatch((11, 7.5), (11, 6.8), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow8)

# Feature Engineering (Real-time)
rt_feature_box = FancyBboxPatch((9.5, 6.1), 3, 0.6, 
                               boxstyle="round,pad=0.05", 
                               edgecolor='black', facecolor=color_train, linewidth=2)
ax.add_patch(rt_feature_box)
ax.text(11, 6.4, 'Real-time Feature Engineering\n60 Features Generated', 
        ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Arrow to prediction
arrow9 = FancyArrowPatch((11, 6.1), (11, 5.4), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow9)

# Model Prediction
prediction_box = FancyBboxPatch((9.5, 4.5), 3, 0.8, 
                               boxstyle="round,pad=0.05", 
                               edgecolor='black', facecolor=color_predict, linewidth=3)
ax.add_patch(prediction_box)
ax.text(11, 5, '5-Model Ensemble Prediction', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='white')
ax.text(11, 4.7, 'Weighted Aggregation', ha='center', va='center', 
        fontsize=8, color='white')

# Individual Predictions (side)
individual_preds = [
    ('RF', 14, 5.3),
    ('XGB', 14, 5),
    ('LGB', 14, 4.7),
    ('CAT', 14, 4.4),
    ('GB', 14, 4.1)
]

for model, x, y in individual_preds:
    small_box = FancyBboxPatch((x, y), 0.8, 0.2, 
                              boxstyle="round,pad=0.02", 
                              edgecolor='black', facecolor='white', linewidth=1)
    ax.add_patch(small_box)
    ax.text(x + 0.4, y + 0.1, model, ha='center', va='center', 
            fontsize=7, fontweight='bold')
    
    # Arrow to ensemble
    arrow = FancyArrowPatch((x, y + 0.1), (12.5, 4.9), 
                           arrowstyle='->', mutation_scale=10, 
                           linewidth=1, color='gray', alpha=0.5)
    ax.add_patch(arrow)

ax.text(14.4, 5.6, 'Individual\nScores', ha='center', va='center', 
        fontsize=7, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black'))

# Arrow to result
arrow10 = FancyArrowPatch((11, 4.5), (11, 3.7), 
                         arrowstyle='->', mutation_scale=20, 
                         linewidth=2.5, color='black')
ax.add_patch(arrow10)

# =====================================================================
# PHASE 6: OUTPUT & RESPONSE
# =====================================================================
ax.text(11, 3.9, 'PHASE 6: OUTPUT', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black'))

# Result Processing
result_box = FancyBboxPatch((9.5, 2.8), 3, 0.7, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor=color_output, linewidth=2)
ax.add_patch(result_box)
ax.text(11, 3.3, 'Prediction Result Processing', ha='center', va='center', 
        fontsize=9, fontweight='bold', color='white')
ax.text(11, 3, 'Fraud Probability | Risk Score | Key Factors', ha='center', va='center', 
        fontsize=7, color='white')

# Arrow to outputs
arrow11 = FancyArrowPatch((11, 2.8), (11, 2.1), 
                         arrowstyle='->', mutation_scale=20, 
                         linewidth=2.5, color='black')
ax.add_patch(arrow11)

# Output Channels
outputs = [
    ('Web Dashboard\nHTML/CSS/JS', 9, 1.0, '#3498db'),
    ('JSON API\nResponse', 11, 1.0, '#2ecc71'),
    ('Alert System\nNotifications', 13, 1.0, '#e74c3c')
]

for label, x, y, color in outputs:
    box = FancyBboxPatch((x-0.9, y), 1.8, 0.8, 
                        boxstyle="round,pad=0.05", 
                        edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y + 0.4, label, ha='center', va='center', 
            fontsize=8, fontweight='bold', color='white')
    
    # Arrow from result
    arrow = FancyArrowPatch((11, 2.1), (x, y + 0.8), 
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)

# =====================================================================
# BOTTOM: METRICS & INFO
# =====================================================================
metrics_box = FancyBboxPatch((0.3, 0.1), 17.4, 0.6, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor='#34495e', linewidth=2)
ax.add_patch(metrics_box)

ax.text(9, 0.55, '📊 SYSTEM PERFORMANCE METRICS', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='white')
ax.text(9, 0.3, 'Accuracy: 94.46% | Precision: 93.82% | Recall: 94.15% | F1-Score: 93.98% | AUC: 0.9487 | Latency: <3ms | Throughput: 450+ tx/s', 
        ha='center', va='center', fontsize=8, color='white')

# =====================================================================
# LEGEND
# =====================================================================
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='black', label='Data Input'),
    mpatches.Patch(facecolor=color_process, edgecolor='black', label='Processing'),
    mpatches.Patch(facecolor=color_train, edgecolor='black', label='Training'),
    mpatches.Patch(facecolor=color_predict, edgecolor='black', label='Prediction'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output'),
    mpatches.Patch(facecolor=color_storage, edgecolor='black', label='Storage')
]

ax.legend(handles=legend_elements, loc='upper left', fontsize=8, 
         framealpha=0.95, edgecolor='black', title='Workflow Stages', 
         title_fontsize=9, bbox_to_anchor=(0.005, 0.99))

# Workflow arrow (connecting phases)
ax.annotate('', xy=(7.5, 6), xytext=(6, 1.6), 
           arrowprops=dict(arrowstyle='->', lw=3, color='red', linestyle=':', alpha=0.5))
ax.text(6.8, 3.8, 'Training\n→\nDeployment', ha='center', va='center', 
        fontsize=8, fontweight='bold', color='red', 
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.7))

# Save
plt.tight_layout()
plt.savefig('graphs/complete_workflow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Complete Workflow Diagram Generated!")
print("📁 File: graphs/complete_workflow_diagram.png")
print("\n🔄 Workflow Phases:")
print("  Phase 1: Data Collection (Transaction Records, User Behavior, Account History)")
print("  Phase 2: Preprocessing (Cleaning, Missing Values, Outliers, Feature Engineering)")
print("  Phase 3: Model Training (5 ML Models + Ensemble Creation)")
print("  Phase 4: Deployment (Flask Server + API Endpoints)")
print("  Phase 5: Real-time Prediction (Feature Engineering + Ensemble Prediction)")
print("  Phase 6: Output (Web Dashboard, JSON API, Alert System)")
print("\n📊 System Metrics:")
print("  • Accuracy: 94.46%")
print("  • 5-Model Ensemble: RF, XGBoost, LightGBM, CatBoost, Gradient Boosting")
print("  • Dataset: 1,048,574 transactions")
print("  • Features: 60 engineered features")
print("  • Latency: <3ms per prediction")
print("\n🎉 Complete workflow diagram ready!")

plt.close()
