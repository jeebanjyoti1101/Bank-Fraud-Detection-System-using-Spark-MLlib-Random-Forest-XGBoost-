"""
Test Fraud Detection Predictions
Tests the API with known fraud and legitimate transaction patterns
"""
import requests
import json

API_URL = "http://localhost:5001/api/predict"

# Test Cases
test_cases = [
    {
        "name": "🚨 CLEAR FRAUD: Complete Drain + Large Amount",
        "data": {
            "type": "TRANSFER",
            "amount": 500000,
            "oldbalanceOrg": 500000,
            "newbalanceOrig": 0,  # Complete drain
            "oldbalanceDest": 0,
            "newbalanceDest": 500000
        },
        "expected": "FRAUD"
    },
    {
        "name": "✅ LEGITIMATE: Normal Payment",
        "data": {
            "type": "PAYMENT",
            "amount": 50,
            "oldbalanceOrg": 10000,
            "newbalanceOrig": 9950,
            "oldbalanceDest": 5000,
            "newbalanceDest": 5050
        },
        "expected": "LEGITIMATE"
    },
    {
        "name": "🚨 FRAUD: Large Transfer with Zero Destination",
        "data": {
            "type": "TRANSFER",
            "amount": 350000,
            "oldbalanceOrg": 400000,
            "newbalanceOrig": 50000,
            "oldbalanceDest": 0,
            "newbalanceDest": 0  # Destination doesn't change (suspicious)
        },
        "expected": "FRAUD"
    },
    {
        "name": "🚨 FRAUD: Cash Out Complete Balance",
        "data": {
            "type": "CASH_OUT",
            "amount": 1000000,
            "oldbalanceOrg": 1000000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 1000000
        },
        "expected": "FRAUD"
    },
    {
        "name": "✅ LEGITIMATE: Small Cash In",
        "data": {
            "type": "CASH_IN",
            "amount": 1000,
            "oldbalanceOrg": 5000,
            "newbalanceOrig": 6000,
            "oldbalanceDest": 0,
            "newbalanceDest": 0
        },
        "expected": "LEGITIMATE"
    },
    {
        "name": "⚠️ SUSPICIOUS: Balance Error + Large Amount",
        "data": {
            "type": "TRANSFER",
            "amount": 200000,
            "oldbalanceOrg": 300000,
            "newbalanceOrig": 150000,  # Should be 100000 (balance error!)
            "oldbalanceDest": 100000,
            "newbalanceDest": 300000
        },
        "expected": "FRAUD"
    },
    {
        "name": "✅ LEGITIMATE: Regular Debit",
        "data": {
            "type": "DEBIT",
            "amount": 500,
            "oldbalanceOrg": 15000,
            "newbalanceOrig": 14500,
            "oldbalanceDest": 0,
            "newbalanceDest": 0
        },
        "expected": "LEGITIMATE"
    }
]

print("="*80)
print("🧪 TESTING FRAUD DETECTION PREDICTIONS")
print("="*80)

results = []
for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['name']}")
    print(f"   Input: {test['data']['type']} | Amount: ${test['data']['amount']:,}")
    
    try:
        response = requests.post(API_URL, json=test['data'], headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            is_fraud = result.get('is_fraud')
            fraud_prob = result.get('fraud_probability', 0)
            components = result.get('components', {})
            
            # Get individual model predictions
            rf_prob = components.get('rf_probability', 'N/A')
            xgb_prob = components.get('xgb_probability', 'N/A')
            lgb_prob = components.get('lgb_probability', 'N/A')
            cat_prob = components.get('cat_probability', 'N/A')
            
            prediction = "FRAUD" if is_fraud else "LEGITIMATE"
            match = "✅ CORRECT" if prediction == test['expected'] else "❌ WRONG"
            
            print(f"   Prediction: {prediction} ({fraud_prob:.2f}%) {match}")
            print(f"   Models: RF={rf_prob}% | XGB={xgb_prob}% | LGB={lgb_prob}% | CAT={cat_prob}%")
            
            results.append({
                'test': test['name'],
                'expected': test['expected'],
                'predicted': prediction,
                'correct': prediction == test['expected'],
                'fraud_probability': fraud_prob
            })
        else:
            print(f"   ❌ ERROR: {response.status_code} - {response.text}")
            results.append({'test': test['name'], 'correct': False})
    
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        results.append({'test': test['name'], 'correct': False})

# Summary
print("\n" + "="*80)
print("📊 RESULTS SUMMARY")
print("="*80)
correct = sum(1 for r in results if r.get('correct'))
total = len(results)
accuracy = (correct / total * 100) if total > 0 else 0

print(f"✅ Correct Predictions: {correct}/{total} ({accuracy:.1f}%)")
print(f"❌ Wrong Predictions: {total - correct}/{total}")

if accuracy >= 85:
    print("\n🎉 EXCELLENT! Models are working correctly!")
elif accuracy >= 70:
    print("\n👍 GOOD! Most predictions are accurate.")
else:
    print("\n⚠️ WARNING! Predictions need improvement.")

print("="*80)
