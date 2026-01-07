"""
Generate Fraud Detection System Architecture Diagram
Similar style to medical disease prediction workflow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patches import Polygon
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(12, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

# Colors
color_blue = '#4A90E2'
color_green = '#50C878'
color_orange = '#FF8C42'
color_purple = '#9B59B6'
color_gray = '#95A5A6'

# =====================================================================
# TOP: DATA SOURCES
# =====================================================================

# Transaction Data Icon (Left)
transaction_icon = Rectangle((0.5, 14.5), 1, 0.8, 
                             edgecolor='black', facecolor=color_blue, linewidth=2)
ax.add_patch(transaction_icon)
ax.text(1, 15.1, '💳', fontsize=30, ha='center', va='center')
ax.text(1, 14.2, 'Transaction\nData', fontsize=9, ha='center', va='center', fontweight='bold')

# Financial Records Icon (Right)
records_icon = Rectangle((0.5, 13), 1, 0.8, 
                         edgecolor='black', facecolor=color_blue, linewidth=2)
ax.add_patch(records_icon)
ax.text(1, 13.6, '📊', fontsize=30, ha='center', va='center')
ax.text(1, 12.7, 'Financial\nRecords', fontsize=9, ha='center', va='center', fontweight='bold')

# Arrows from data sources to database
arrow1 = FancyArrowPatch((1.5, 14.9), (2.2, 13.5), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((1.5, 13.4), (2.2, 13.5), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow2)

# =====================================================================
# DATABASE
# =====================================================================

# Database cylinder
cylinder = Polygon([(2.3, 13.8), (3.7, 13.8), (3.7, 13.2), (2.3, 13.2)], 
                  closed=True, edgecolor='black', facecolor=color_blue, linewidth=2)
ax.add_patch(cylinder)

# Database top ellipse
ellipse_top = mpatches.Ellipse((3, 13.8), 1.4, 0.3, 
                               edgecolor='black', facecolor=color_blue, linewidth=2)
ax.add_patch(ellipse_top)
ellipse_top2 = mpatches.Ellipse((3, 13.8), 1.4, 0.3, 
                                edgecolor='black', facecolor='white', linewidth=1, alpha=0.3)
ax.add_patch(ellipse_top2)

# Database icon
ax.text(3, 13.5, '⚕️', fontsize=35, ha='center', va='center')
ax.text(3, 12.9, 'Fraud Dataset\n1M+ Records', fontsize=8, ha='center', va='center', fontweight='bold')

# Arrow to preprocessing
arrow3 = FancyArrowPatch((3.7, 13.5), (4.5, 13.5), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow3)

# =====================================================================
# PREPROCESSING
# =====================================================================

# Preprocessing cylinder
prep_cylinder = Polygon([(4.6, 13.8), (6.0, 13.8), (6.0, 13.2), (4.6, 13.2)], 
                       closed=True, edgecolor='black', facecolor=color_green, linewidth=2)
ax.add_patch(prep_cylinder)

prep_ellipse_top = mpatches.Ellipse((5.3, 13.8), 1.4, 0.3, 
                                    edgecolor='black', facecolor=color_green, linewidth=2)
ax.add_patch(prep_ellipse_top)

# Gear icon for preprocessing
ax.text(5.3, 13.5, '⚙️', fontsize=35, ha='center', va='center')
ax.text(5.3, 12.9, 'Preprocessing', fontsize=9, ha='center', va='center', fontweight='bold')

# Arrow to Multifiller
arrow4 = FancyArrowPatch((6.0, 13.5), (7.2, 13.5), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow4)

# =====================================================================
# MULTIFILLER
# =====================================================================

multifiller_box = FancyBboxPatch((7.2, 13.1), 2, 0.8, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(multifiller_box)
ax.text(8.2, 13.5, 'Data Cleaning\n& Validation', fontsize=9, ha='center', va='center', fontweight='bold')

# Arrow down to Feature Extraction
arrow5 = FancyArrowPatch((5.3, 12.9), (5.3, 12.1), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow5)

# =====================================================================
# FEATURE EXTRACTION
# =====================================================================

feature_box = FancyBboxPatch((3.5, 11.3), 3.6, 0.7, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(feature_box)
ax.text(5.3, 11.65, 'Feature Extraction\n60 Engineered Features', 
        fontsize=9, ha='center', va='center', fontweight='bold')

# Arrow down to ML Techniques
arrow6 = FancyArrowPatch((5.3, 11.3), (5.3, 10.5), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow6)

# =====================================================================
# MACHINE LEARNING TECHNIQUES BOX (Side)
# =====================================================================

ml_list_box = FancyBboxPatch((7.5, 8.5), 2.2, 3, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(ml_list_box)

ml_algorithms = [
    'Random Forest',
    'XGBoost',
    'LightGBM',
    'CatBoost',
    'Gradient Boosting'
]

ax.text(8.6, 11.2, 'ML Models', fontsize=10, ha='center', va='center', fontweight='bold')

y_pos = 10.7
for algo in ml_algorithms:
    ax.text(8.6, y_pos, f'• {algo}', fontsize=8, ha='center', va='center')
    y_pos -= 0.4

# Arrow from ML list to ML Techniques
arrow7 = FancyArrowPatch((7.5, 10), (7.1, 10), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow7)

# =====================================================================
# MACHINE LEARNING TECHNIQUES
# =====================================================================

ml_box = FancyBboxPatch((3.5, 9.5), 3.6, 0.9, 
                       boxstyle="round,pad=0.05", 
                       edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(ml_box)
ax.text(5.3, 10.1, 'Machine Learning', fontsize=10, ha='center', va='center', fontweight='bold')
ax.text(5.3, 9.75, 'Ensemble Techniques', fontsize=9, ha='center', va='center')

# Arrow to Performance Evaluation
arrow8 = FancyArrowPatch((5.3, 9.5), (5.3, 8.7), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow8)

# =====================================================================
# PERFORMANCE EVALUATION
# =====================================================================

perf_box = FancyBboxPatch((3.5, 7.5), 3.6, 1.1, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(perf_box)
ax.text(5.3, 8.3, 'Performance Evaluation', fontsize=10, ha='center', va='center', fontweight='bold')

# Mini bar chart
bar_x = [4.3, 4.8, 5.3, 5.8, 6.3]
bar_heights = [0.5, 0.6, 0.55, 0.52, 0.48]
bar_colors = ['#3498db', '#f39c12', '#2ecc71', '#e74c3c', '#9b59b6']

for x, h, c in zip(bar_x, bar_heights, bar_colors):
    rect = Rectangle((x-0.15, 7.7), 0.3, h, facecolor=c, edgecolor='black', linewidth=1)
    ax.add_patch(rect)

# Arrow to Selection
arrow9 = FancyArrowPatch((5.3, 7.5), (5.3, 6.7), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow9)

# =====================================================================
# SELECTION (BEST MODEL)
# =====================================================================

selection_box = FancyBboxPatch((3.5, 6.0), 3.6, 0.6, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(selection_box)
ax.text(5.3, 6.3, 'Selection (Best Ensemble)\nAccuracy: 94.46%', 
        fontsize=9, ha='center', va='center', fontweight='bold')

# Arrow to Prediction
arrow10 = FancyArrowPatch((5.3, 6.0), (5.3, 5.2), 
                         arrowstyle='->', mutation_scale=20, 
                         linewidth=2.5, color='black')
ax.add_patch(arrow10)

# =====================================================================
# FRAUD PREDICTION (CENTER)
# =====================================================================

prediction_box = FancyBboxPatch((3.8, 3.5), 3, 1.6, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor='white', linewidth=3)
ax.add_patch(prediction_box)

# Person icon with detection symbol
ax.text(5.3, 4.7, '🎯', fontsize=50, ha='center', va='center')
ax.text(5.3, 3.85, 'Fraud Detection\nPrediction', 
        fontsize=11, ha='center', va='center', fontweight='bold')

# =====================================================================
# PATIENT/USER (Left Bottom)
# =====================================================================

# User icon box
user_box = Rectangle((0.5, 3.5), 1.5, 1.3, 
                     edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(user_box)

# User in bed icon
ax.text(1.25, 4.5, '👤', fontsize=40, ha='center', va='center')
ax.text(1.25, 3.85, '💳', fontsize=25, ha='center', va='center')
ax.text(1.25, 3.3, 'Transaction\nUser', fontsize=8, ha='center', va='center', fontweight='bold')

# =====================================================================
# HEALTH RECORD (Left Middle)
# =====================================================================

record_box = FancyBboxPatch((0.8, 5.3), 1.5, 0.5, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(record_box)

# Monitor icon
ax.text(1.1, 5.55, '📱', fontsize=20, ha='center', va='center')
ax.text(1.85, 5.55, 'Transaction\nRecord', fontsize=7, ha='center', va='center', fontweight='bold')

# Red arrow from record to prediction
arrow11 = FancyArrowPatch((2.3, 5.55), (3.8, 4.7), 
                         arrowstyle='->', mutation_scale=25, 
                         linewidth=3, color='#e74c3c')
ax.add_patch(arrow11)

# Arrow from user to record
arrow12 = FancyArrowPatch((1.25, 4.8), (1.55, 5.3), 
                         arrowstyle='->', mutation_scale=20, 
                         linewidth=2, color='black')
ax.add_patch(arrow12)

# =====================================================================
# RESPONSE FLOW (Bottom)
# =====================================================================

# Alert/Dashboard (Left)
alert_box = FancyBboxPatch((0.5, 1.5), 1.5, 1.2, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color_green, linewidth=2)
ax.add_patch(alert_box)
ax.text(1.25, 2.4, '📊', fontsize=35, ha='center', va='center')
ax.text(1.25, 1.85, 'Real-time\nAlert', fontsize=8, ha='center', va='center', 
        fontweight='bold', color='white')

# Web Interface (Center)
web_box = FancyBboxPatch((3, 1.5), 2, 1.2, 
                        boxstyle="round,pad=0.05", 
                        edgecolor='black', facecolor=color_orange, linewidth=2)
ax.add_patch(web_box)
ax.text(4, 2.4, '🌐', fontsize=35, ha='center', va='center')
ax.text(4, 1.85, 'Web\nDashboard', fontsize=9, ha='center', va='center', 
        fontweight='bold', color='white')

# API Response (Right)
api_box = FancyBboxPatch((6, 1.5), 2, 1.2, 
                        boxstyle="round,pad=0.05", 
                        edgecolor='black', facecolor=color_purple, linewidth=2)
ax.add_patch(api_box)
ax.text(7, 2.4, '🔌', fontsize=35, ha='center', va='center')
ax.text(7, 1.85, 'API\nResponse', fontsize=9, ha='center', va='center', 
        fontweight='bold', color='white')

# Arrows from prediction to outputs
arrow13 = FancyArrowPatch((3.8, 4.0), (1.25, 2.7), 
                         arrowstyle='->', mutation_scale=20, 
                         linewidth=2.5, color='black')
ax.add_patch(arrow13)

arrow14 = FancyArrowPatch((5.3, 3.5), (4, 2.7), 
                         arrowstyle='->', mutation_scale=20, 
                         linewidth=2.5, color='black')
ax.add_patch(arrow14)

arrow15 = FancyArrowPatch((6.8, 4.0), (7, 2.7), 
                         arrowstyle='->', mutation_scale=20, 
                         linewidth=2.5, color='black')
ax.add_patch(arrow15)

# =====================================================================
# BOTTOM: SYSTEM INFO
# =====================================================================

info_box = FancyBboxPatch((0.5, 0.3), 8, 0.8, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(info_box)

ax.text(4.5, 0.95, 'Fraud Detection System Architecture', 
        fontsize=12, ha='center', va='center', fontweight='bold')
ax.text(4.5, 0.65, 'Technology: Flask + Python | 5 ML Models | 94.46% Accuracy', 
        fontsize=9, ha='center', va='center')
ax.text(4.5, 0.4, 'Dataset: 1,048,574 Transactions | 60 Features | Real-time Processing', 
        fontsize=8, ha='center', va='center', style='italic')

# Save
plt.tight_layout()
plt.savefig('graphs/fraud_detection_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Fraud Detection Architecture Diagram saved!")
print("📁 File: graphs/fraud_detection_architecture.png")
print("\n🏗️ Architecture Flow:")
print("  1. Transaction Data & Financial Records")
print("  2. → Fraud Dataset (1M+ Records)")
print("  3. → Preprocessing & Data Cleaning")
print("  4. → Feature Extraction (60 Features)")
print("  5. → Machine Learning (5-Model Ensemble)")
print("  6. → Performance Evaluation")
print("  7. → Best Model Selection (94.46% Accuracy)")
print("  8. → Fraud Detection Prediction")
print("  9. → Output: Alerts, Web Dashboard, API Response")
print("\n🎉 Architecture diagram generated successfully!")

plt.close()
