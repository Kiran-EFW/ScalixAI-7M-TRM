#!/usr/bin/env python3
"""
Test script to verify Gemini 2.5 Flash configuration
"""

import os
from gemini_integration import GeminiConfig, GeminiAPI

def test_gemini_connection():
    """Test basic Gemini API connectivity"""

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not found")
        return False

    print(f"🔑 Using API key: {api_key[:20]}...")
    print("🤖 Testing Gemini 2.5 Flash connection...")

    # Initialize Gemini API
    config = GeminiConfig(api_key=api_key)
    api = GeminiAPI(config)

    print(f"📝 Model: {config.model}")
    print(f"🔗 Base URL: {config.base_url}")

    # Test simple prompt
    test_prompt = "Hello! Please respond with exactly 'Gemini 2.5 Flash is working!' if you can hear me."

    print("📤 Sending test message...")
    response = api.generate_response(test_prompt)

    if response:
        print("✅ Gemini API connected successfully!")
        print(f"📥 Response: {response}")
        return True
    else:
        print("❌ Failed to get response from Gemini API")
        return False

if __name__ == "__main__":
    success = test_gemini_connection()
    exit(0 if success else 1)
