#!/usr/bin/env python3
"""
Check current Gemini API rate limit status
"""

import os
import sys
from datetime import datetime
from gemini_integration import GeminiConfig, GeminiAPI

def check_rate_limits():
    """Check and display current rate limit status"""

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not found")
        print("Run: source setup_gemini.sh")
        return

    print("🔍 Checking Gemini API Rate Limits...")
    print("=" * 50)

    # Initialize API
    config = GeminiConfig(api_key=api_key)
    api = GeminiAPI(config)

    # Get status
    status = api.get_rate_limit_status()

    print(f"📊 Current Rate Limit Status (as of {datetime.now().strftime('%H:%M:%S')})")
    print()

    # Request counts
    print("📈 Request Usage:")
    minute_req = status.get('minute_requests', 0)
    hour_req = status.get('hour_requests', 0)
    day_req = status.get('day_requests', 0)

    print(f"  • Last minute: {minute_req}/30 requests ({minute_req/30*100:.1f}%)")
    print(f"  • Last hour: {hour_req}/1,000 requests ({hour_req/1000*100:.1f}%)")
    print(f"  • Last day: {day_req}/5,000 requests ({day_req/5000*100:.1f}%)")
    print()

    # Token usage
    minute_tokens = status.get('minute_tokens', 0)
    print("🎫 Token Usage:")
    print(f"  • Last minute: {minute_tokens:,}/100,000 tokens ({minute_tokens/100000*100:.1f}%)")
    print()

    # Burst status
    burst_count = status.get('burst_count', 0)
    print("💥 Burst Status:")
    print(f"  • Burst requests: {burst_count}/5 ({burst_count/5*100:.1f}%)")
    print()

    # Cooldown status
    cooldown_active = status.get('cooldown_active', False)
    cooldown_remaining = status.get('cooldown_remaining', 0)

    if cooldown_active:
        print("⏰ Cooldown Status:")
        print(f"  • ACTIVE - {cooldown_remaining:.1f} seconds remaining")
        print("  • Status: 🟡 Rate limited - waiting")
    else:
        print("✅ Cooldown Status:")
        print("  • INACTIVE - API ready for requests")

    print()
    print("🛡️  Safety Measures:")
    print("  • 90% minute limit protection")
    print("  • 80% hour limit protection")
    print("  • 70% day limit protection")
    print("  • Automatic cooldown on rate limit hits")
    print("  • Burst rate limiting")
    print("  • Conservative daily limits (5,000 requests/day)")

    print()
    print("💡 Recommendations:")

    if day_req > 4000:
        print("  • 🚨 HIGH USAGE: Consider pausing for 24 hours")
    elif day_req > 3000:
        print("  • ⚠️  MODERATE: Monitor closely, reduce frequency if needed")
    elif day_req > 1000:
        print("  • 📊 NORMAL: Usage within safe limits")
    else:
        print("  • ✅ LOW: Safe to continue")

    if cooldown_active:
        print(f"  • ⏳ Wait {cooldown_remaining:.0f} more seconds before making requests")

    print()
    print("🔄 Run this command anytime: python check_rate_limits.py")

if __name__ == "__main__":
    check_rate_limits()
