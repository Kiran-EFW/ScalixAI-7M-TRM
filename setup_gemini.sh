#!/bin/bash
# Setup script for Gemini 2.5 Flash API configuration

echo "🤖 Setting up Gemini 2.5 Flash for ScalixAI-7M-TRM..."

# Set the API key (from previous logs)
export GEMINI_API_KEY="AIzaSyBqC3-mhSRhRsCcXEuRSA0urefK8TVdWKE"

echo "✅ Gemini API key configured"
echo "🔑 API Key: ${GEMINI_API_KEY:0:20}..."
echo "🤖 Model: gemini-2.0-flash-thinking-exp (Gemini 2.5 Flash)"
echo ""
echo "To use in your shell session, run:"
echo "source setup_gemini.sh"
echo ""
echo "Or add this to your ~/.bashrc:"
echo "export GEMINI_API_KEY=\"$GEMINI_API_KEY\""
