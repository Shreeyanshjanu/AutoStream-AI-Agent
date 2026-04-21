"""
Quick intent classification test - minimal API calls
"""

from intent_classifier import get_intent_classifier, Intent


def quick_test():
    """Test with just 3 examples to avoid rate limits"""
    
    print("\n" + "=" * 70)
    print("QUICK INTENT CLASSIFIER TEST (3 examples)")
    print("=" * 70 + "\n")
    
    classifier = get_intent_classifier()
    
    test_cases = [
        ("Hey, how are you?", Intent.CASUAL),
        ("What's the Pro plan price?", Intent.INQUIRY),
        ("I want to sign up for the Pro plan for YouTube", Intent.HIGH_INTENT),
    ]
    
    passed = 0
    
    for message, expected in test_cases:
        actual = classifier.classify(message)
        status = "✅" if actual == expected else "❌"
        
        print(f"{status} Message: \"{message}\"")
        print(f"   Expected: {expected.value:12} | Got: {actual.value}\n")
        
        if actual == expected:
            passed += 1
    
    print("=" * 70)
    print(f"Results: {passed}/3 tests passed")
    if passed == 3:
        print("✅ QUICK TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    quick_test()