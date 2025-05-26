#!/usr/bin/env python3
"""
Simple test script to check MemFuse server connectivity
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

def test_memfuse_connection():
    """Test if MemFuse server is running and accessible"""
    memfuse_base_url = os.getenv("MEMFUSE_BASE_URL", "http://127.0.0.1:8000")
    
    print(f"Testing MemFuse connection to: {memfuse_base_url}")
    
    try:
        # Try to reach the health endpoint
        response = requests.get(f"{memfuse_base_url}/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ MemFuse server is running and accessible!")
            return True
        else:
            print(f"❌ MemFuse server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to MemFuse server. Is it running?")
        print("   Make sure to start the MemFuse server first.")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection to MemFuse server timed out.")
        return False
    except Exception as e:
        print(f"❌ Error connecting to MemFuse server: {e}")
        return False

def test_api_keys():
    """Test if required API keys are set"""
    print("\nTesting API key configuration:")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    
    if openai_api_key:
        print("✅ OPENAI_API_KEY is set")
    else:
        print("❌ OPENAI_API_KEY is not set")
    
    if openai_base_url:
        print(f"✅ OPENAI_BASE_URL is set to: {openai_base_url}")
    else:
        print("❌ OPENAI_BASE_URL is not set")
    
    return bool(openai_api_key and openai_base_url)

if __name__ == "__main__":
    print("MemFuse Chatbot Connection Test")
    print("=" * 40)
    
    memfuse_ok = test_memfuse_connection()
    api_keys_ok = test_api_keys()
    
    print("\n" + "=" * 40)
    if memfuse_ok and api_keys_ok:
        print("✅ All checks passed! You can run the chatbot.")
    else:
        print("❌ Some checks failed. Please fix the issues above before running the chatbot.")
        if not memfuse_ok:
            print("   - Start the MemFuse server")
        if not api_keys_ok:
            print("   - Set the required API keys in your .env file") 