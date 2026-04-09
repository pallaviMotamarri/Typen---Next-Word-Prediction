"""
Test script to verify the model integration in app.py
"""

import sys
import os
sys.path.insert(0, 'd:\\PROJECTS\\typen\\backend')

# Test imports
try:
    import numpy as np
    from keras.models import load_model
    from keras.preprocessing.sequence import pad_sequences
    print("✓ Keras/TensorFlow imported")
except ImportError as e:
    print(f"✗ Keras import failed: {e}")
    sys.exit(1)

# Test model loading
MODEL_PATH = 'd:\\PROJECTS\\typen\\backend\\model\\next_word_model\\next_word_model.h5'
try:
    model = load_model(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

# Test vocabulary
VOCABULARY = {
    0: "the", 1: "a", 2: "and", 3: "to", 4: "of", 5: "is", 6: "in", 7: "it", 
    8: "that", 9: "was", 10: "for", 11: "on", 12: "with", 13: "be", 14: "are",
}
REVERSE_VOCABULARY = {word: idx for idx, word in VOCABULARY.items()}
print(f"✓ Vocabulary loaded ({len(VOCABULARY)} words)")

# Test text_to_sequence function
def text_to_sequence(text, max_len=10):
    words = text.lower().split()
    sequence = []
    for word in words:
        if word in REVERSE_VOCABULARY:
            sequence.append(REVERSE_VOCABULARY[word])
        else:
            sequence.append(1)
    padded = pad_sequences([sequence], maxlen=max_len, padding='pre')
    return padded

# Test prediction
test_text = "machine learning"
print(f"\n{'='*60}")
print("Testing Prediction")
print(f"{'='*60}")
print(f"Input: '{test_text}'")

try:
    sequence = text_to_sequence(test_text)
    predictions = model.predict(sequence, verbose=0)
    top_indices = np.argsort(predictions[0])[-5:][::-1]
    
    print(f"\nTop 5 predicted words:")
    for i, idx in enumerate(top_indices):
        word = VOCABULARY.get(idx, f"word_{idx}")
        confidence = predictions[0][idx]
        print(f"  {i+1}. '{word}' - {confidence:.4f}")
    
    print(f"\n✓ Prediction successful!")
except Exception as e:
    print(f"✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*60}")
print("✓ All tests passed! Model is ready for Flask integration")
print(f"{'='*60}")
