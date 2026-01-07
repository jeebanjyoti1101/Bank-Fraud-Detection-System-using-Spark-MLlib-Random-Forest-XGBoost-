# Interactive Graph Scripts for Fraud Detection

This directory contains Python scripts for generating fraud detection visualizations. All graphs display interactively in windows rather than saving to files.

## 🚀 Quick Start - Show All Graphs Sequentially

**Run the master script to see all graphs one after another:**

```bash
python scripts/show_all_graphs.py
```

Simply close each window to automatically display the next graph! Perfect for reviewing all visualizations in one session.

---

## Available Graphs

### 1. **ROC Curves - Individual Models** (`graph_01_roc_individual.py`)
- Shows ROC curve for each model in separate subplots
- Displays AUC scores for RF, XGBoost, LightGBM, and CatBoost

### 2. **Combined ROC Curve** (`graph_02_roc_combined.py`)
- Compares all models on a single ROC plot
- Easy comparison of model performance

### 3. **Precision-Recall Curve** (`graph_03_precision_recall.py`)
- Shows precision vs recall tradeoff
- Displays Average Precision (AP) scores

### 4. **Confusion Matrix Heatmap** (`graph_04_confusion_matrix.py`)
- Confusion matrices for all four models
- Shows True Positives, False Positives, True Negatives, False Negatives

### 5. **Feature Importance** (`graph_05_feature_importance.py`)
- Top 15 most important features for each model
- Helps understand what drives predictions

### 6. **Transaction Type Distribution** (`graph_06_transaction_distribution.py`)
- Distribution of transaction types
- Fraud rate by transaction type

### 7. **Amount Distribution** (`graph_07_amount_distribution.py`)
- Transaction amount distributions (log scale)
- Comparison of fraud vs normal transaction amounts
- Box plots by transaction type
- Cumulative distribution function

### 8. **Performance Metrics Comparison** (`graph_08_performance_metrics.py`)
- Bar charts comparing Accuracy, Precision, Recall, F1-Score
- Side-by-side model comparison

## How to Run

### 🎯 Option 1: Show All Graphs Sequentially (Recommended)

```bash
python scripts/show_all_graphs.py
```

- **Best for**: Viewing all graphs in one session
- **How it works**: Each graph appears in a window. Close the window to see the next graph automatically
- **Total graphs**: 8 comprehensive visualizations
- **Time**: ~2-3 minutes to view all

### 📊 Option 2: Run Individual Graphs

```bash
# Run from the project root directory
python scripts/graph_01_roc_individual.py
python scripts/graph_02_roc_combined.py
python scripts/graph_03_precision_recall.py
python scripts/graph_04_confusion_matrix.py
python scripts/graph_05_feature_importance.py
python scripts/graph_06_transaction_distribution.py
python scripts/graph_07_amount_distribution.py
python scripts/graph_08_performance_metrics.py
```

### Interactive Display

- Each script will open a matplotlib window
- You can zoom, pan, and save the plot manually
- Close the window to end the script

## Requirements

All scripts require:
- pandas
- numpy
- matplotlib
- scikit-learn
- joblib

Some scripts also use:
- seaborn (for heatmaps)
- xgboost, lightgbm, catboost (model libraries)

## Notes

- Scripts use a sample of 5,000 transactions for faster processing
- Feature engineering is applied to match the trained models
- Missing columns are handled automatically with default values
- Close the matplotlib window to run the next script

## Tips

1. **Start with simple graphs** like transaction distribution before complex ones
2. **Run one at a time** to avoid overwhelming your system
3. **Maximize the window** for better visibility
4. **Use the toolbar** in the matplotlib window to zoom and save manually if needed
