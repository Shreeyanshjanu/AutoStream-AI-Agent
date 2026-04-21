"""
Comprehensive test suite for Intent Classifier
"""

from intent_classifier import get_intent_classifier, Intent


def test_intent_classification():
    """Test intent classification with various messages"""
    
    print("\n" + "=" * 80)
    print("INTENT CLASSIFIER COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    classifier = get_intent_classifier()
    
    # Test cases organized by intent
    test_cases = {
        Intent.CASUAL: [
            "Hey!",
            "How are you?",
            "Hello there",
            "Thanks so much!",
            "What's up?",
            "Good morning",
            "Bye!",
            "See you later",
            "That's great",
        ],
        Intent.INQUIRY: [
            "What's the Pro plan price?",
            "Can I get 4K video quality?",
            "What's your refund policy?",
            "How many videos per month?",
            "Do you have AI captions?",
            "What's included in the Basic plan?",
            "Is there 24/7 support?",
            "Can I cancel anytime?",
            "What file formats do you support?",
            "How much storage do I get?",
        ],
        Intent.HIGH_INTENT: [
            "I want to try the Pro plan for my YouTube channel",
            "Sign me up for the Basic plan",
            "I'm ready to get started with AutoStream",
            "Let's go with the Pro plan for my Instagram",
            "I'd like to sign up immediately",
            "I want to start editing videos for TikTok",
            "Can I upgrade to Pro right now?",
            "I'm ready to become a customer",
            "Let me sign up for the yearly plan",
            "I want to try this out for my podcast",
        ],
    }
    
    total_passed = 0
    total_failed = 0
    results_by_intent = {}
    
    for expected_intent, messages in test_cases.items():
        print(f"\n{'=' * 80}")
        print(f"Testing {expected_intent.value.upper()} Intent")
        print(f"{'=' * 80}\n")
        
        intent_passed = 0
        intent_failed = 0
        
        for message in messages:
            actual_intent = classifier.classify(message)
            
            is_match = actual_intent == expected_intent
            status = "✅" if is_match else "❌"
            
            print(f"{status} \"{message}\"")
            print(f"   Expected: {expected_intent.value:12} | Got: {actual_intent.value:12}")
            
            if is_match:
                intent_passed += 1
                total_passed += 1
            else:
                intent_failed += 1
                total_failed += 1
            
            print()
        
        results_by_intent[expected_intent.value] = {
            "passed": intent_passed,
            "failed": intent_failed,
            "total": len(messages)
        }
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80 + "\n")
    
    for intent_type, results in results_by_intent.items():
        percentage = (results["passed"] / results["total"] * 100) if results["total"] > 0 else 0
        print(f"{intent_type.upper():12} | {results['passed']:2}/{results['total']:2} passed | {percentage:5.1f}%")
    
    print("\n" + "-" * 80)
    total_tests = total_passed + total_failed
    overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"{'OVERALL':12} | {total_passed:2}/{total_tests:2} passed | {overall_percentage:5.1f}%")
    print("=" * 80)
    
    if total_failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total_failed} test(s) failed")
    
    return total_failed == 0


if __name__ == "__main__":
    success = test_intent_classification()
    exit(0 if success else 1)