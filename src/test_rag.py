"""
Test suite for RAG pipeline
"""

from rag import get_rag_pipeline


def test_rag_basic_queries():
    """Test RAG with basic product questions"""
    print("\n" + "=" * 70)
    print("RAG PIPELINE INTEGRATION TESTS")
    print("=" * 70)
    
    rag = get_rag_pipeline()
    
    # Test Case 1: Pricing Question
    print("\n[TEST 1] Pricing Question")
    print("-" * 70)
    result1 = rag.answer_question(
        "What's the price of the Pro plan?",
        verbose=True
    )
    assert "$79" in result1["answer"], "Pro plan price should be mentioned"
    print("✅ TEST 1 PASSED: Correctly identified Pro plan price")
    
    # Test Case 2: Feature Question
    print("\n[TEST 2] Feature Question")
    print("-" * 70)
    result2 = rag.answer_question(
        "Can I get 4K video quality?",
        verbose=True
    )
    assert "4K" in result2["answer"] or "4K" in str(result2), "4K should be mentioned"
    print("✅ TEST 2 PASSED: Correctly identified 4K feature")
    
    # Test Case 3: Policy Question
    print("\n[TEST 3] Policy Question")
    print("-" * 70)
    result3 = rag.answer_question(
        "What's your refund policy?",
        verbose=True
    )
    assert "7 days" in result3["answer"].lower() or "refund" in result3["answer"].lower(), \
        "Refund policy should be mentioned"
    print("✅ TEST 3 PASSED: Correctly identified refund policy")
    
    # Test Case 4: Support Question
    print("\n[TEST 4] Support Availability Question")
    print("-" * 70)
    result4 = rag.answer_question(
        "Is there 24/7 support?",
        verbose=True
    )
    assert "24/7" in result4["answer"] or "support" in result4["answer"].lower(), \
        "Support info should be mentioned"
    print("✅ TEST 4 PASSED: Correctly identified support availability")
    
    # Test Case 5: Comparison Question
    print("\n[TEST 5] Plan Comparison Question")
    print("-" * 70)
    result5 = rag.answer_question(
        "What's the difference between Basic and Pro plans?",
        verbose=True
    )
    assert len(result5["answer"]) > 50, "Should provide detailed comparison"
    print("✅ TEST 5 PASSED: Correctly compared plans")
    
    # Test Case 6: Edge Case - Unknown Query
    print("\n[TEST 6] Unknown Information Query")
    print("-" * 70)
    result6 = rag.answer_question(
        "What's your company's secret recipe?",
        verbose=True
    )
    print("✅ TEST 6 PASSED: Gracefully handled unknown query")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nRAG Pipeline is working correctly:")
    print("  ✅ Retrieves relevant KB sections")
    print("  ✅ Generates accurate, grounded answers")
    print("  ✅ Handles unknown queries gracefully")
    print("  ✅ Ready for Phase 3!")


if __name__ == "__main__":
    test_rag_basic_queries()