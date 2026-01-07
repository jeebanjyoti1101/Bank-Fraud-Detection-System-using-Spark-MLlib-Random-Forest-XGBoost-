# Gradient Boosting Integration

## Overview
Added **sklearn's GradientBoostingClassifier** to the fraud detection ensemble, bringing the total to 5 models:
1. Random Forest (RF)
2. XGBoost (XGB)
3. LightGBM (LGB)
4. CatBoost (CAT)
5. **Gradient Boosting (GB)** ← NEW

## Why Gradient Boosting?

### Advantages
- **Interpretable**: Provides clear feature importances and decision paths
- **Robust**: Built-in regularization prevents overfitting
- **Complementary**: Uses different boosting strategy than XGB/LGB/CAT
- **Stable**: Less sensitive to hyperparameters than XGBoost
- **Pure Python**: No external dependencies beyond scikit-learn

### Differences from XGBoost/LightGBM/CatBoost
| Feature | Gradient Boosting | XGBoost | LightGBM | CatBoost |
|---------|------------------|---------|----------|----------|
| **Speed** | Moderate | Very Fast | Fastest | Fast |
| **Memory** | Low | Medium | Low | High |
| **Regularization** | L2 only | L1 + L2 | L1 + L2 | L2 |
| **Tree Growth** | Level-wise | Level-wise | Leaf-wise | Symmetric |
| **Categorical Handling** | Manual encoding | Manual encoding | Native | Native |

## Training Configuration

### Hyperparameters
```python
GradientBoostingClassifier(
    n_estimators=200,        # 200 boosting rounds (epochs)
    max_depth=5,             # Shallow trees (prevents overfitting)
    learning_rate=0.1,       # Moderate learning rate
    subsample=0.8,           # 80% row sampling per tree
    min_samples_split=100,   # Robust splits
    min_samples_leaf=50,     # Minimum leaf size
    max_features='sqrt',     # Feature subsampling
    random_state=42
)
```

### Training Details
- **Epochs**: 200 boosting rounds
- **Training time**: ~15-25 minutes (moderate speed)
- **Memory usage**: Lower than XGB/CAT
- **Class balancing**: Natural imbalance ratio (16.76% fraud)

## How to Train

### Step 1: Train the model
```cmd
python train_with_gradient_boosting.py
```

This will:
1. Load existing metadata and scaler from `models/`
2. Engineer 60 features (same as existing models)
3. Train GradientBoostingClassifier with 200 rounds
4. Evaluate on test set and show AUC score
5. Save model to `models/gb_model.pkl`
6. Update `models/advanced_metadata.json` with GB weights

### Step 2: Verify integration
The app (`app/app.py`) has already been updated to:
- Load `gb_model.pkl` automatically
- Include GB predictions in ensemble
- Show GB probability in API responses

### Step 3: Test the ensemble
```cmd
cd app
python app.py
```

Then test via API:
```bash
curl -X POST http://localhost:5001/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"type\":\"TRANSFER\",\"amount\":500000,\"oldbalanceOrg\":100000,\"newbalanceOrig\":50000,\"oldbalanceDest\":0,\"newbalanceDest\":0}"
```

Response will include:
```json
{
  "components": {
    "rf_probability": 85.2,
    "xgb_probability": 87.1,
    "lgb_probability": 86.5,
    "cat_probability": 88.0,
    "gb_probability": 84.7
  },
  "fraud_probability": 86.3,
  "is_fraud": true,
  "risk_level": "HIGH"
}
```

## Ensemble Weights

### Default Weights (Equal)
```json
{
  "rf": 0.20,
  "xgb": 0.20,
  "lgb": 0.20,
  "cat": 0.20,
  "gb": 0.20
}
```

### How to Tune Weights
1. **By AUC Score** (recommended):
   - Train all models
   - Weight each model by its AUC score
   - Normalize weights to sum to 1.0

2. **By Grid Search**:
   - Try different weight combinations
   - Evaluate ensemble AUC on validation set
   - Pick weights that maximize performance

3. **Manual Tuning**:
   - Edit `models/advanced_metadata.json`
   - Adjust `ensemble_weights` section
   - Restart app to load new weights

Example optimized weights:
```json
{
  "rf": 0.18,
  "xgb": 0.22,
  "lgb": 0.23,
  "cat": 0.22,
  "gb": 0.15
}
```

## Performance Comparison

### Expected Metrics
| Model | AUC Score | Training Time | Memory |
|-------|-----------|---------------|--------|
| Random Forest | 0.87-0.89 | 20-30 min | Medium |
| XGBoost | 0.88-0.90 | 10-15 min | High |
| LightGBM | 0.88-0.90 | 5-10 min | Low |
| CatBoost | 0.88-0.90 | 15-25 min | High |
| **Gradient Boosting** | **0.85-0.87** | **15-25 min** | **Low** |
| **Ensemble (5 models)** | **0.89-0.91** | **N/A** | **N/A** |

### Why GB has slightly lower AUC
- More conservative (less prone to overfitting)
- Fewer hyperparameters tuned
- Provides diversity to ensemble

**Note**: Lower individual AUC can still improve ensemble performance through model diversity!

## Troubleshooting

### Issue: `gb_model.pkl` not found
**Solution**: Run `python train_with_gradient_boosting.py` first

### Issue: Feature mismatch errors
**Solution**: Ensure you've trained base models first with `train_improved.py` or `train_fast.py`

### Issue: Memory error during training
**Solution**: Reduce dataset size or use smaller `n_estimators` (e.g., 100 instead of 200)

### Issue: Slow predictions
**Solution**: Gradient Boosting is slower than tree-based models. Consider reducing max_depth or n_estimators for production

## Next Steps

### Optional Improvements
1. **Tune Hyperparameters**:
   - Grid search for optimal max_depth, learning_rate, n_estimators
   - Use cross-validation to find best configuration

2. **Feature Selection**:
   - Use GB's feature_importances_ to identify top features
   - Retrain with only important features for speed

3. **Ensemble Optimization**:
   - Use stacking (meta-learner) instead of weighted averaging
   - Train a LogisticRegression on top of 5 models

4. **Production Optimization**:
   - Convert to ONNX for faster inference
   - Use quantization to reduce model size

## Files Modified/Created

### New Files
- `train_with_gradient_boosting.py` - Training script
- `GB_INTEGRATION.md` - This documentation

### Modified Files
- `app/app.py` - Added GB to load_models() and ensemble_predict()
- `models/advanced_metadata.json` - Added GB weights and AUC score
- `models/gb_model.pkl` - Trained model (created after running training script)

## Summary

✅ **What was added**: Sklearn's GradientBoostingClassifier with 200 epochs  
✅ **Integration**: Fully integrated into app.py ensemble  
✅ **Benefits**: Increased model diversity, better robustness  
✅ **Trade-offs**: Slightly slower inference, lower individual AUC  
✅ **Result**: Expected ensemble AUC improvement of +0.5-1.0%  

---

**Questions?** Check the main README.md or review the training script for details.
