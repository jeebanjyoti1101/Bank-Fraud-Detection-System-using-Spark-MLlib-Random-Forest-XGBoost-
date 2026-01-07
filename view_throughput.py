"""
Quick Throughput Comparison Viewer
Opens the existing throughput comparison graphs
"""
import os
import webbrowser
from pathlib import Path

# Find the throughput graphs
graph_paths = [
    ("Throughput Comparison", "graphs/throughput_comparison.png"),
    ("Throughput Over Time", "graphs/throughput_over_time.png"),
    ("Throughput Comparison (Evaluation)", "evaluation_results/throughput_comparison.png")
]

print("\n" + "="*70)
print("⚡ THROUGHPUT PERFORMANCE - FRAUD DETECTION MODELS")
print("="*70 + "\n")

found_graphs = []
for name, path in graph_paths:
    if os.path.exists(path):
        abs_path = os.path.abspath(path)
        found_graphs.append((name, abs_path))
        print(f"✅ Found: {name}")
        print(f"   Path: {path}\n")

if found_graphs:
    print(f"📊 Opening {len(found_graphs)} throughput graph(s)...\n")
    
    for name, abs_path in found_graphs[:2]:  # Open first 2 to avoid overwhelming
        try:
            webbrowser.open(f'file://{abs_path}')
            print(f"   🖼️  Opened: {name}")
        except Exception as e:
            print(f"   ⚠️  Could not open {name}: {e}")
    
    print()
else:
    print("❌ No throughput graphs found.\n")

print("="*70)
print("\n💡 About Throughput Performance:")
print("   • Measures transactions processed per second")
print("   • Shows prediction speed for each model")
print("   • Important for real-time fraud detection")
print("   • Higher throughput = Faster predictions")
print("\n📈 Typical Performance:")
print("   • LightGBM: ~10,000-50,000 TPS (fastest)")
print("   • XGBoost: ~5,000-20,000 TPS")
print("   • Random Forest: ~2,000-10,000 TPS")
print("   • CatBoost: ~1,000-5,000 TPS")
print("   • Ensemble: Combined throughput with parallel processing")
print("="*70 + "\n")
