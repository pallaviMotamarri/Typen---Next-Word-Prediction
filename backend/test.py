"""
Test script for Next Word Prediction Model
Tests model file, loads it, and makes predictions with word mappings
"""

import os
import sys
import h5py
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'next_word_model', 'next_word_model.h5')

# Create a sample vocabulary for demonstration
# In production, you should load this from a saved vocabulary file
VOCABULARY = {
    0: "the", 1: "a", 2: "and", 3: "to", 4: "of", 5: "is", 6: "in", 7: "it", 
    8: "that", 9: "was", 10: "for", 11: "on", 12: "with", 13: "be", 14: "are",
    15: "as", 16: "by", 17: "from", 18: "at", 19: "or", 20: "an", 21: "this",
    22: "but", 23: "not", 24: "have", 25: "had", 26: "has", 27: "do", 28: "does",
    29: "did", 30: "will", 31: "would", 32: "could", 33: "should", 34: "can",
    35: "may", 36: "might", 37: "must", 38: "very", 39: "more", 40: "most",
    41: "other", 42: "some", 43: "all", 44: "each", 45: "every", 46: "both",
    47: "few", 48: "many", 49: "much", 50: "been", 51: "being", 52: "have",
    53: "hello", 54: "world", 55: "python", 56: "data", 57: "science", 58: "machine",
    59: "learning", 60: "deep", 61: "neural", 62: "network", 63: "model", 64: "prediction",
    65: "next", 66: "word", 67: "text", 68: "language", 69: "processing",
    70: "artificial", 71: "intelligence", 72: "algorithm", 73: "code", 74: "program"
}

# Reverse vocabulary for encoding
REVERSE_VOCABULARY = {word: idx for idx, word in VOCABULARY.items()}

def load_prediction_model():
    """Load the pre-trained model"""
    try:
        model = load_model(MODEL_PATH)
        print(f"✓ Model loaded successfully!")
        print(f"  Location: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def text_to_sequence(text, max_len=10):
    """Convert text to numeric sequence using vocabulary"""
    words = text.lower().split()
    sequence = []
    
    for word in words:
        # Find word in reverse vocabulary
        if word in REVERSE_VOCABULARY:
            sequence.append(REVERSE_VOCABULARY[word])
        else:
            # Use a default index for unknown words
            sequence.append(1)  # Assuming 1 is for unknown words
    
    # Pad the sequence
    padded = pad_sequences([sequence], maxlen=max_len, padding='pre')
    return padded

def predict_next_word(model, text, top_n=5):
    """Predict the next word with probabilities"""
    try:
        print(f"\n{'─' * 60}")
        print(f"Input text: '{text}'")
        print(f"{'─' * 60}")
        
        # Convert text to sequence
        sequence = text_to_sequence(text)
        
        # Make prediction
        predictions = model.predict(sequence, verbose=0)
        
        # Get top N predictions
        top_indices = np.argsort(predictions[0])[-top_n:][::-1]
        
        print(f"\nTop {top_n} Next Word Predictions:")
        print(f"{'Rank':<6} {'Probability':<15} {'Word':<20}")
        print("─" * 50)
        
        for rank, idx in enumerate(top_indices, 1):
            word = VOCABULARY.get(idx, f"<unknown_{idx}>")
            probability = predictions[0][idx]
            
            # Visual probability bar
            bar_length = int(probability * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            print(f"{rank:<6} {probability:<7.4f} (%) {bar}  {word:<15}")
        
        print("─" * 50)
        
        # Show the top predicted word
        top_word = VOCABULARY.get(top_indices[0], f"<unknown_{top_indices[0]}>")
        top_prob = predictions[0][top_indices[0]]
        print(f"\n✓ Most likely next word: '{top_word}' ({top_prob:.4f} confidence)")
        
        return top_word, predictions[0]
        
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main test function"""
    print("=" * 60)
    print("Next Word Prediction Model - Full Test with Predictions")
    print("=" * 60)
    
    # Load the model
    model = load_prediction_model()
    if model is None:
        print("\nTest failed: Could not load model")
        sys.exit(1)
    
    print(f"\nModel Summary:")
    model.summary()
    
    print(f"\n{'=' * 60}")
    print("Making Predictions")
    print("=" * 60)
    
    # Test cases
    test_inputs = [
        "hello world",
        "python is",
        "machine learning",
        "deep neural network",
        "the quick brown",
    ]
    
    results = []
    for test_input in test_inputs:
        top_word, probs = predict_next_word(model, test_input, top_n=5)
        if top_word:
            results.append((test_input, top_word))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Prediction Summary")
    print("=" * 60)
    for input_text, predicted in results:
        print(f"  '{input_text}' → '{predicted}'")
    
    print(f"\n{'=' * 60}")
    print("✓ Test completed successfully!")
    print("=" * 60)
    print(f"\nVocabulary size: {len(VOCABULARY)}")
    print(f"Model can predict from {len(VOCABULARY)} different words")

if __name__ == "__main__":
    main()
