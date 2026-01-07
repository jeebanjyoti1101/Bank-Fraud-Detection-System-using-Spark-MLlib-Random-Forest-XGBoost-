"""
Test the improved fraud detection system with real-world fraud scenarios
"""

import requests
import json

BASE_URL = "http://localhost:5001"

# Test cases covering various fraud patterns
test_cases = [
    {
        "name": "Large Transfer with Zero Destination (CRITICAL FRAUD)",
        "data": {
            "type": "TRANSFER",
            "amount": 350000.0,
            "oldbalanceOrg": 350000.0,
            "newbalanceOrig": 0.0
        },
        "expected": "FRAUD",
        "reason": "Complete account drain with large transfer - classic fraud"
    },
    {
        "name": "Large Cash Out with Balance Error (FRAUD)",
        "data": {
            "type": "CASH_OUT",
            "amount": 500000.0,
            "oldbalanceOrg": 500000.0,
            "newbalanceOrig": 10000.0
        },
        "expected": "FRAUD",
        "reason": "Large amount with suspicious balance inconsistency"
    },
    {
        "name": "High-Value Transfer with Complete Drain (FRAUD)",
        "data": {
            "type": "TRANSFER",
            "amount": 1000000.0,
            "oldbalanceOrg": 1000000.0,
            "newbalanceOrig": 0.0
        },
        "expected": "FRAUD",
        "reason": "Very large amount draining entire account"
    },
    {
        "name": "Medium Transfer with Partial Drain (SUSPICIOUS)",
        "data": {
            "type": "TRANSFER",
            "amount": 250000.0,
            "oldbalanceOrg": 300000.0,
            "newbalanceOrig": 50000.0
        },
        "expected": "FRAUD",
        "reason": "Large transfer draining most of balance"
    },
    {
        "name": "Normal Payment (LEGITIMATE)",
        "data": {
            "type": "PAYMENT",
            "amount": 5000.0,
            "oldbalanceOrg": 100000.0,
            "newbalanceOrig": 95000.0
        },
        "expected": "LEGITIMATE",
        "reason": "Normal payment with proper balance"
    },
    {
        "name": "Small Cash In (LEGITIMATE)",
        "data": {
            "type": "CASH_IN",
            "amount": 2000.0,
            "oldbalanceOrg": 50000.0,
            "newbalanceOrig": 52000.0
        },
        "expected": "LEGITIMATE",
        "reason": "Small cash in deposit"
    },
    {
        "name": "Moderate Debit (LEGITIMATE)",
        "data": {
            "type": "DEBIT",
            "amount": 15000.0,
            "oldbalanceOrg": 200000.0,
            "newbalanceOrig": 185000.0
        },
        "expected": "LEGITIMATE",
        "reason": "Normal debit transaction"
    },
    {
        "name": "Cash Out with Balance Error (FRAUD)",
        "data": {
            "type": "CASH_OUT",
            "amount": 400000.0,
            "oldbalanceOrg": 450000.0,
            "newbalanceOrig": 100000.0
        },
        "expected": "FRAUD",
        "reason": "Balance doesn't match transaction amount"
    },
    {
        "name": "Transfer with Exact Zero Destination (CRITICAL FRAUD)",
        "data": {
            "type": "TRANSFER",
            "amount": 750000.0,
            "oldbalanceOrg": 800000.0,
            "newbalanceOrig": 50000.0
        },
        "expected": "FRAUD",
        "reason": "Large transfer likely to zero destination account"
    },
    {
        "name": "Small Payment (LEGITIMATE)",
        "data": {
            "type": "PAYMENT",
            "amount": 500.0,
            "oldbalanceOrg": 25000.0,
            "newbalanceOrig": 24500.0
        },
        "expected": "LEGITIMATE",
        "reason": "Small routine payment"
    }
]

def test_prediction(test_case):
    """Test a single prediction case"""
    try:
        response = requests.post(f"{BASE_URL}/predict", json=test_case["data"])
        result = response.json()
        
        # Extract prediction
        prediction = result.get('prediction', 'UNKNOWN')
        probability = result.get('probability', 0)
        
        # Check if correct
        is_correct = prediction == test_case["expected"]
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        
        print(f"\n{'='*80}")
        print(f"Test: {test_case['name']}")
        print(f"{'='*80}")
        print(f"Input:")
        print(f"  Type: {test_case['data']['type']}")
        print(f"  Amount: ${test_case['data']['amount']:,.2f}")
        print(f"  Balance Before: ${test_case['data']['oldbalanceOrg']:,.2f}")
        print(f"  Balance After: ${test_case['data']['newbalanceOrig']:,.2f}")
        print(f"\nExpected: {test_case['expected']}")
        print(f"Reason: {test_case['reason']}")
        print(f"\nPrediction: {prediction} ({probability:.2f}%)")
        print(f"Status: {status}")
        
        # Show model breakdown if available
        if 'model_predictions' in result:
            print(f"\nModel Breakdown:")
            for model, pred in result['model_predictions'].items():
                print(f"  {model}: {pred:.2f}%")
        
        return is_correct
        
    except Exception as e:
        print(f"\n❌ ERROR testing {test_case['name']}: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("🧪 TESTING IMPROVED FRAUD DETECTION SYSTEM")
    print("="*80)
    print(f"Target: 90%+ accuracy on fraud detection")
    print(f"Total tests: {len(test_cases)}")
    
    results = []
    for test_case in test_cases:
        is_correct = test_prediction(test_case)
        results.append(is_correct)
    
    # Summary
    correct_count = sum(results)
    accuracy = (correct_count / len(results)) * 100
    
    print(f"\n{'='*80}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Wrong: {len(results) - correct_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 90:
        print(f"\n🎉 SUCCESS! Achieved {accuracy:.1f}% accuracy (target: 90%+)")
    else:
        print(f"\n⚠️  Below target. Got {accuracy:.1f}%, need 90%+")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
