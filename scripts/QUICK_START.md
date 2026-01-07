# 🎨 Sequential Graph Viewer - Quick Guide

## What You Get

A single command that shows **8 comprehensive fraud detection graphs** one after another:

```bash
python scripts/show_all_graphs.py
```

## How It Works

1. **Run the command** - Script loads models and data once
2. **First graph appears** - View it, zoom, explore
3. **Close the window** - Next graph automatically appears
4. **Repeat** - Continue through all 8 graphs
5. **Done!** - All visualizations reviewed

## The 8 Graphs You'll See

1. **ROC Curves (Individual)** - 4 models, separate subplots with AUC scores
2. **ROC Curves (Combined)** - All models on one plot for comparison
3. **Precision-Recall Curves** - Shows precision vs recall tradeoff
4. **Confusion Matrices** - TP, FP, TN, FN for each model
5. **Feature Importance** - Top 15 features driving predictions
6. **Transaction Distribution** - Transaction types and fraud rates
7. **Amount Analysis** - 4 different views of transaction amounts
8. **Performance Metrics** - Accuracy, Precision, Recall, F1-Score comparison

## Tips

- **Take your time** - Each window stays open until you close it
- **Use toolbar** - Zoom, pan, save individual graphs if needed
- **Stop anytime** - Press Ctrl+C in terminal to exit early
- **Re-run anytime** - Just run the command again

## Individual Graphs (Optional)

Want just one specific graph? Run individual scripts:

```bash
python scripts/graph_01_roc_individual.py
python scripts/graph_02_roc_combined.py
python scripts/graph_03_precision_recall.py
python scripts/graph_04_confusion_matrix.py
python scripts/graph_05_feature_importance.py
python scripts/graph_06_transaction_distribution.py
python scripts/graph_07_amount_distribution.py
python scripts/graph_08_performance_metrics.py
```

## Performance

- **Sample Size**: 5,000 transactions (for speed)
- **Load Time**: ~10-20 seconds initial data preparation
- **Display Time**: Instant once data is ready
- **Full Feature Engineering**: All 60 features computed correctly

---

**Enjoy exploring your fraud detection model performance! 📊✨**
