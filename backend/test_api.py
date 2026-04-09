"""
Example API test file for the Next Word Prediction endpoint
This demonstrates how to test the /api/predict endpoint
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:5000/api/predict"  # Change port if needed

# Example test cases
test_cases = [
    {
        "text": "the quick brown",
        "genre": "fiction",
        "description": "Classic phrase"
    },
    {
        "text": "machine learning is",
        "genre": "technical",
        "description": "Technical text"
    },
    {
        "text": "deep neural network",
        "genre": "academic",
        "description": "AI terminology"
    },
    {
        "text": "python is a",
        "genre": "technical",
        "description": "Programming language"
    },
    {
        "text": "data science",
        "genre": "business",
        "description": "Business/analytics"
    },
    {
        "text": "",  # Empty text - tests default predictions
        "genre": "fiction",
        "description": "Empty input test"
    }
]

def test_predict_endpoint():
    """Test the prediction endpoint with various inputs"""
    
    print("=" * 70)
    print("Testing Next Word Prediction API")
    print("=" * 70)
    
    for test in test_cases:
        print(f"\n[TEST] {test['description']}")
        print(f"Input: '{test['text']}' (genre: {test['genre']})")
        print("-" * 70)
        
        # Prepare request
        payload = {
            "text": test["text"],
            "genre": test["genre"]
        }
        
        try:
            # Make API request
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "success":
                    print(f"✓ Status: {data['status'].upper()}")
                    print(f"  Model: {data.get('model', 'Unknown')}")
                    print(f"  Predictions ({len(data['predictions'])} total):")
                    
                    # Display predictions
                    for pred in data["predictions"]:
                        pred_type = "PROBABLE" if pred["type"] == "probable" else "CREATIVE"
                        confidence = pred.get("confidence", 0)
                        bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
                        print(f"    {pred['rank']:>3}. {bar} {pred['word']:15} ({confidence:.4f}) [{pred_type}]")
                else:
                    print(f"✗ Error: {data.get('message', 'Unknown error')}")
            else:
                print(f"✗ HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"✗ Connection Error: Cannot reach {API_URL}")
            print("  Make sure the Flask app is running: python app.py")
            break
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)

def test_response_format():
    """Validate response format"""
    
    print("\n" + "=" * 70)
    print("Validating Response Format")
    print("=" * 70)
    
    payload = {
        "text": "machine learning",
        "genre": "fiction"
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()
        
        print("\nResponse Structure:")
        print(json.dumps(data, indent=2))
        
        # Validate structure
        assert "status" in data, "Missing 'status' field"
        assert "predictions" in data, "Missing 'predictions' field"
        assert isinstance(data["predictions"], list), "Predictions should be a list"
        
        # Validate each prediction
        for pred in data["predictions"]:
            required_fields = ["id", "word", "rank", "type"]
            for field in required_fields:
                assert field in pred, f"Missing '{field}' in prediction"
        
        print("\n✓ Response format is valid!")
        
    except Exception as e:
        print(f"\n✗ Validation failed: {str(e)}")

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "NEXT WORD PREDICTION API - TEST SUITE" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Note about running the app
    print("\nNOTE: Make sure the Flask app is running!")
    print("Run in another terminal: python app.py")
    print("\nWaiting for API to be available...")
    
    import time
    import sys
    
    # Give user time to start the app
    print("\nPress Enter to continue when the Flask app is running (Ctrl+C to stop)...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        sys.exit(0)
    
    # Run tests
    test_predict_endpoint()
    test_response_format()
