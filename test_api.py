import requests
import json

# Test cases
tests = [
    {
        "name": "Test 1: Normal Payment",
        "data": {
            "type": "PAYMENT",
            "amount": 500,
            "oldbalanceOrg": 1000,
            "newbalanceOrig": 500,
            "oldbalanceDest": 0,
            "newbalanceDest": 0
        },
        "expected": "LEGITIMATE"
    },
    {
        "name": "Test 2: Small Transfer",
        "data": {
            "type": "TRANSFER",
            "amount": 200,
            "oldbalanceOrg": 1000,
            "newbalanceOrig": 800,
            "oldbalanceDest": 0,
            "newbalanceDest": 0
        },
        "expected": "LEGITIMATE"
    },
    {
        "name": "Test 3: Complete Drain (Large Amount)",
        "data": {
            "type": "TRANSFER",
            "amount": 50000,
            "oldbalanceOrg": 50000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 0
        },
        "expected": "FRAUD (HIGH RISK)"
    },
    {
        "name": "Test 4: Impossible Balance Increase",
        "data": {
            "type": "TRANSFER",
            "amount": 200,
            "oldbalanceOrg": 200,
            "newbalanceOrig": 500,
            "oldbalanceDest": 0,
            "newbalanceDest": 0
        },
        "expected": "FRAUD (CRITICAL)"
    },
    {
        "name": "Test 5: Zero Balance Transaction",
        "data": {
            "type": "TRANSFER",
            "amount": 1000,
            "oldbalanceOrg": 0,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 0
        },
        "expected": "FRAUD (CRITICAL)"
    }
]

print("="*80)
print("🧪 FRAUD DETECTION API TESTS")
print("="*80)

url = "http://localhost:5001/api/predict"

for test in tests:
    print(f"\n{test['name']}")
    print(f"Expected: {test['expected']}")
    print("-" * 40)
    
    try:
        response = requests.post(url, json=test['data'], timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Result:")
            print(f"   Is Fraud: {result.get('is_fraud')}")
            print(f"   Fraud Probability: {result.get('fraud_probability')}%")
            print(f"   Risk Level: {result.get('risk_level')}")
            print(f"\n📝 Full Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running on port 5001?")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*80)
print("✅ Tests Complete!")
print("="*80)
