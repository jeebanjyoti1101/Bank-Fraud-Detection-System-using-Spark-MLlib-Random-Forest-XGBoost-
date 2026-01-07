"""
View All Training Graphs
Opens all model training visualization graphs
"""
import os
import webbrowser
import time

training_graphs = [
    ("Training Accuracy Over Epochs", "graphs/training_accuracy_over_epochs.png"),
    ("Training Loss Curves", "graphs/training_loss_curves.png"),
    ("Learning Curves", "graphs/learning_curves.png"),
    ("Training Time Comparison", "graphs/training_time_comparison.png"),
    ("Performance Dashboard", "graphs/training_performance_dashboard.png"),
    ("Cross-Validation Scores", "graphs/cross_validation_scores.png")
]

print("\n" + "="*70)
print("📊 MODEL TRAINING VISUALIZATIONS")
print("="*70 + "\n")

found_graphs = []
for name, path in training_graphs:
    if os.path.exists(path):
        abs_path = os.path.abspath(path)
        found_graphs.append((name, abs_path))
        print(f"✅ Found: {name}")

if found_graphs:
    print(f"\n🖼️  Opening {len(found_graphs)} training graph(s)...\n")
    
    for i, (name, abs_path) in enumerate(found_graphs, 1):
        try:
            webbrowser.open(f'file://{abs_path}')
            print(f"   {i}. Opened: {name}")
            time.sleep(0.5)  # Small delay between opens
        except Exception as e:
            print(f"   ⚠️  Could not open {name}: {e}")
    
    print()
else:
    print("\n❌ No training graphs found.")
    print("   Run: python generate_training_graphs.py\n")

print("="*70)
print("\n💡 About These Graphs:")
print("\n1. Training Accuracy Over Epochs")
print("   • Shows how model accuracy improves during training")
print("   • Each line represents a different model")
print("   • Higher curves = Better performance")
print("\n2. Training Loss Curves")
print("   • Shows how prediction error decreases over time")
print("   • Lower is better")
print("   • Smooth curves indicate stable training")
print("\n3. Learning Curves")
print("   • Shows impact of training data size on accuracy")
print("   • More data generally = Better performance")
print("   • Plateau indicates optimal data size reached")
print("\n4. Training Time Comparison")
print("   • Compares training speed of each model")
print("   • LightGBM fastest, Random Forest slowest")
print("   • Important for production deployment")
print("\n5. Performance Dashboard")
print("   • 4-panel comparison of all metrics")
print("   • Accuracy, Precision, Recall, F1-Score")
print("   • Easy side-by-side comparison")
print("\n6. Cross-Validation Scores")
print("   • Shows consistency across 5 different data splits")
print("   • Box plots show score distribution")
print("   • Yellow diamonds = Mean scores")
print("="*70 + "\n")

print("🎯 For Your Presentation:")
print("   • Use graph 1 to show training progress")
print("   • Use graph 4 to show efficiency (fast training)")
print("   • Use graph 5 for comprehensive metrics overview")
print("   • Use graph 6 to prove model reliability")
print("="*70 + "\n")
