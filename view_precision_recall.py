"""
Quick Precision-Recall Curve Viewer
Opens the existing precision-recall curve graph
"""
import os
import webbrowser
from pathlib import Path

# Find the precision-recall curve
graph_paths = [
    "graphs/precision_recall_curve.png",
    "evaluation_results/precision_recall_curve.png"
]

print("\n" + "="*70)
print("📊 PRECISION-RECALL CURVE - FRAUD DETECTION MODELS")
print("="*70 + "\n")

found = False
for path in graph_paths:
    if os.path.exists(path):
        abs_path = os.path.abspath(path)
        print(f"✅ Found graph at: {path}")
        print(f"   Full path: {abs_path}\n")
        
        # Open the image
        try:
            webbrowser.open(f'file://{abs_path}')
            print("🖼️  Opening graph in your default image viewer...\n")
            found = True
            break
        except Exception as e:
            print(f"⚠️  Could not open automatically: {e}")
            print(f"   Please open manually: {abs_path}\n")
            found = True
            break

if not found:
    print("❌ Precision-Recall curve not found.")
    print("   Run: python scripts\\graph_03_precision_recall.py\n")

print("="*70)
print("\n💡 About Precision-Recall Curve:")
print("   • Shows trade-off between precision and recall")
print("   • Higher curve = Better model performance")
print("   • Area Under Curve (AUC-PR) measures overall quality")
print("   • Important for imbalanced datasets (like fraud detection)")
print("="*70 + "\n")
