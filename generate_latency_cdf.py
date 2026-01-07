"""
Generate Latency CDF (Cumulative Distribution Function) Graph
Shows prediction latency distribution for all models
"""
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.preprocessing import RobustScaler

print("⏱️  Generating Latency CDF Graph...")
print("="*70)

# Load models
models = {}
model_names = {
    'rf': 'RF',
    'xgb': 'XGB',
    'lgb': 'LGB',
    'cat': 'Cat'
}

model_files = {
    'rf': 'models/rf_model.pkl',
    'xgb': 'models/xgboost_model.pkl',
    'lgb': 'models/lightgbm_model.pkl',
    'cat': 'models/catboost_model.pkl'
}

for name, label in model_names.items():
    try:
        models[name] = joblib.load(model_files[name])
        print(f"✅ Loaded {label}")
    except Exception as e:
        print(f"⚠️  Could not load {label}: {e}")

if not models:
    print("❌ No models loaded. Exiting.")
    exit()

# Load scaler
try:
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Loaded scaler")
except:
    scaler = RobustScaler()
    print("⚠️  Using default scaler")

print("\n📊 Running latency benchmark...")

# Generate synthetic test data (60 features matching the model)
np.random.seed(42)
n_samples = 1000
n_features = 60

# Create realistic test data
X_test = np.random.randn(n_samples, n_features)
X_test[:, 0] = np.random.uniform(100, 500000, n_samples)  # amount
X_test[:, 1] = np.random.uniform(0, 1000000, n_samples)   # oldbalance
X_test[:, 2] = np.random.uniform(0, 1000000, n_samples)   # newbalance

# Scale the data
try:
    X_test_scaled = scaler.transform(X_test)
except:
    X_test_scaled = X_test

# Measure latency for each model
latencies = {}
for name, model in models.items():
    print(f"   Testing {model_names[name]}...", end=' ')
    model_latencies = []
    
    # Warmup
    for _ in range(10):
        try:
            _ = model.predict_proba(X_test_scaled[:1])
        except:
            pass
    
    # Actual measurements
    for i in range(n_samples):
        start = time.perf_counter()
        try:
            _ = model.predict_proba(X_test_scaled[i:i+1])
        except:
            pass
        end = time.perf_counter()
        latency_ms = (end - start) * 1000  # Convert to milliseconds
        model_latencies.append(latency_ms)
    
    latencies[name] = sorted(model_latencies)
    
    # Calculate percentiles
    p50 = np.percentile(model_latencies, 50)
    p95 = np.percentile(model_latencies, 95)
    p99 = np.percentile(model_latencies, 99)
    
    print(f"P50={p50:.2f}ms, P95={p95:.2f}ms, P99={p99:.2f}ms")

print("\n📈 Generating CDF plot...")

# Create the CDF plot
plt.figure(figsize=(10, 6))

colors = {
    'rf': '#1f77b4',   # Blue
    'xgb': '#ff7f0e',  # Orange
    'lgb': '#2ca02c',  # Green
    'cat': '#d62728'   # Red
}

for name, lats in latencies.items():
    # Calculate CDF
    y = np.arange(1, len(lats) + 1) / len(lats) * 100
    plt.plot(lats, y, label=model_names[name], color=colors[name], linewidth=2)

# Styling
plt.xlabel('Latency (ms)', fontsize=12, fontweight='bold')
plt.ylabel('Cumulative Probability (%)', fontsize=12, fontweight='bold')
plt.title('Latency CDF - Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
plt.xlim(0, max([max(l) for l in latencies.values()]) * 1.1)
plt.ylim(0, 100)

# Add percentile reference lines
plt.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
plt.axhline(y=95, color='gray', linestyle=':', linewidth=1, alpha=0.5)
plt.axhline(y=99, color='gray', linestyle=':', linewidth=1, alpha=0.5)

plt.tight_layout()

# Save the plot
output_path = 'graphs/latency_cdf.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Latency CDF saved to: {output_path}")

# Show the plot
plt.show()

print("\n" + "="*70)
print("📊 Latency Statistics Summary:")
print("="*70)
print(f"{'Model':<10} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Mean (ms)':<12}")
print("-"*70)

for name, lats in latencies.items():
    p50 = np.percentile(lats, 50)
    p95 = np.percentile(lats, 95)
    p99 = np.percentile(lats, 99)
    mean = np.mean(lats)
    print(f"{model_names[name]:<10} {p50:<12.2f} {p95:<12.2f} {p99:<12.2f} {mean:<12.2f}")

print("="*70)
print("\n✨ Graph generation complete!")
