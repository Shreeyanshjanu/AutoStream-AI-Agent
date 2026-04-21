"""
Intent Classification Module for AutoStream Agent
Classifies user messages into: CASUAL, INQUIRY, or HIGH_INTENT
"""

import os
from enum import Enum
from typing import Dict
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)


class Intent(Enum):
    """User intent categories"""
    CASUAL = "casual"
    INQUIRY = "inquiry"
    HIGH_INTENT = "high_intent"


class IntentClassifier:
    """
    Classifies user messages into intent categories using Gemini
    """
    
    def __init__(self):
        """Initialize intent classifier"""
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.classification_prompt_template = """You are an expert at understanding user intent in customer service conversations.

Classify the following user message into EXACTLY ONE of these three categories:

1. CASUAL - The user is making casual greetings, small talk, saying thanks, or just chatting. No clear business intent.
   Examples: "Hey!", "How are you?", "Thanks!", "What's up?", "Goodbye"

2. INQUIRY - The user is asking questions about features, pricing, policies, or capabilities. They want information.
   Examples: "What's the Pro plan price?", "Can I get 4K videos?", "What's your refund policy?", "How many videos per month?"

3. HIGH_INTENT - The user is ready to buy/sign up, wants to try the service, or mentions their creator platform. They're a potential customer!
   Examples: "I want to try the Pro plan", "Sign me up for YouTube editing", "Let's get started with the basic plan", "I'm ready to sign up for my Instagram content"

USER MESSAGE:
"{message}"

INSTRUCTIONS:
- Respond with ONLY the intent name: CASUAL, INQUIRY, or HIGH_INTENT
- Do NOT include any explanation, punctuation, or extra text
- Focus on business intent, not sentiment
- If a message could be multiple intents, pick the PRIMARY intent
- "How can you help?" = CASUAL (no specific request yet)
- "Can you help me with video editing?" = INQUIRY (asking about capability)
- "Help me set up for YouTube editing" = HIGH_INTENT (wants to use service)

RESPONSE:"""

    def classify(self, user_message: str) -> Intent:
        """
        Classify a user message into an intent category
        
        Args:
            user_message: The user's message to classify
            
        Returns:
            Intent enum value (CASUAL, INQUIRY, or HIGH_INTENT)
        """
        # Create the prompt
        prompt = self.classification_prompt_template.format(message=user_message)
        
        # Get Gemini's response
        response = self.model.generate_content(prompt)
        intent_text = response.text.strip().upper()
        
        # Parse response - extract just the intent name
        # Handle cases where Gemini adds extra text
        intent_text = intent_text.split('\n')[0].strip()
        
        # Map to Intent enum
        try:
            if "HIGH_INTENT" in intent_text or "HIGH-INTENT" in intent_text:
                return Intent.HIGH_INTENT
            elif "INQUIRY" in intent_text:
                return Intent.INQUIRY
            elif "CASUAL" in intent_text:
                return Intent.CASUAL
            else:
                # Fallback: try to parse as enum
                return Intent[intent_text]
        except (KeyError, IndexError):
            # If parsing fails, default to INQUIRY (safe middle ground)
            print(f"⚠️ Warning: Could not parse intent from '{intent_text}', defaulting to INQUIRY")
            return Intent.INQUIRY

    def classify_with_confidence(self, user_message: str) -> Dict:
        """
        Classify with detailed reasoning (for debugging)
        
        Args:
            user_message: The user's message
            
        Returns:
            Dict with intent, confidence reasoning, etc.
        """
        intent = self.classify(user_message)
        
        return {
            "message": user_message,
            "intent": intent,
            "intent_value": intent.value
        }


# Singleton instance
_classifier_instance = None

def get_intent_classifier() -> IntentClassifier:
    """Get or create intent classifier instance"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance


if __name__ == "__main__":
    # Test the intent classifier
    print("=" * 70)
    print("INTENT CLASSIFIER TEST")
    print("=" * 70)
    
    classifier = get_intent_classifier()
    
    # Test cases covering all three intents
    test_cases = [
        # CASUAL examples
        ("Hey!", Intent.CASUAL),
        ("How are you doing?", Intent.CASUAL),
        ("Thanks for helping!", Intent.CASUAL),
        ("What's up?", Intent.CASUAL),
        
        # INQUIRY examples
        ("What's the Pro plan price?", Intent.INQUIRY),
        ("Can I get 4K videos?", Intent.INQUIRY),
        ("What's your refund policy?", Intent.INQUIRY),
        ("How many videos can I edit per month?", Intent.INQUIRY),
        ("Do you have AI captions?", Intent.INQUIRY),
        ("What's included in the Basic plan?", Intent.INQUIRY),
        
        # HIGH_INTENT examples
        ("I want to try the Pro plan for my YouTube channel", Intent.HIGH_INTENT),
        ("Sign me up for the Basic plan", Intent.HIGH_INTENT),
        ("I'm ready to get started with AutoStream for Instagram", Intent.HIGH_INTENT),
        ("Let's go with the Pro plan", Intent.HIGH_INTENT),
        ("I'd like to sign up for YouTube editing", Intent.HIGH_INTENT),
        ("I want to get started immediately", Intent.HIGH_INTENT),
    ]
    
    print("\nClassifying test messages...\n")
    
    passed = 0
    failed = 0
    
    for message, expected_intent in test_cases:
        result = classifier.classify_with_confidence(message)
        actual_intent = result["intent"]
        
        status = "✅" if actual_intent == expected_intent else "❌"
        match = "PASS" if actual_intent == expected_intent else "FAIL"
        
        print(f"{status} [{match}] Message: \"{message}\"")
        print(f"   Expected: {expected_intent.value:12} | Got: {actual_intent.value:12}")
        
        if actual_intent == expected_intent:
            passed += 1
        else:
            failed += 1
        
        print()
    
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"⚠️ {failed} test(s) failed")
    print("=" * 70)