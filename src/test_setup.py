# Test script to verify setup
import sys
from dotenv import load_dotenv
import os

print("✅ Phase 1 Setup Test")
print(f"Python version: {sys.version}")
print(f"Virtual environment active: {sys.prefix}")

# Load .env
load_dotenv()

# Check for API keys
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GOOGLE_API_KEY")
claude_key = os.getenv("ANTHROPIC_API_KEY")

if openai_key:
    print(f"✅ OpenAI API key loaded (key starts with: {openai_key[:10]}...)")
elif gemini_key:
    print(f"✅ Gemini API key loaded (key starts with: {gemini_key[:10]}...)")
elif claude_key:
    print(f"✅ Claude API key loaded (key starts with: {claude_key[:10]}...)")
else:
    print("⚠️ No API key found in .env file")

print("\n✅ All setup complete! Ready for Phase 2.")