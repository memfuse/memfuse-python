#!/usr/bin/env python3
"""
Test to verify non-streaming behavior
"""
import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path so we can import memfuse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memfuse.llm import OpenAI
from memfuse import MemFuse

load_dotenv(override=True)

def test_non_streaming():
    """Test non-streaming functionality directly"""
    
    # Initialize MemFuse and OpenAI client
    memfuse_base_url = os.getenv("MEMFUSE_BASE_URL", "http://127.0.0.1:8000")
    
    try:
        memfuse = MemFuse(base_url=memfuse_base_url)
        memory = memfuse.init(user="non_streaming_test_user")
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            memory=memory
        )
        
        print("Testing non-streaming response...")
        print("=" * 50)
        
        # Test non-streaming
        messages = [{"role": "user", "content": "What is 2+2?"}]
        
        print("Question: What is 2+2?")
        print("Response (non-streaming):")
        print("-" * 30)
        
        response_obj = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=False  # Explicitly disable streaming
        )
        
        if response_obj and response_obj.choices and response_obj.choices[0].message:
            full_response = response_obj.choices[0].message.content
            print(full_response)
            print("\n" + "=" * 50)
            print("✅ Non-streaming response received all at once!")
            print(f"Response length: {len(full_response)} characters")
        else:
            print("❌ No proper response received")
            return False
        
        memfuse.close()
        return True
        
    except Exception as e:
        print(f"❌ Error during non-streaming test: {e}")
        return False

if __name__ == "__main__":
    print("MemFuse Non-Streaming Test")
    print("=" * 40)
    
    success = test_non_streaming()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Non-streaming test completed successfully!")
    else:
        print("❌ Non-streaming test failed.") 