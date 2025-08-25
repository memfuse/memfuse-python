#!/usr/bin/env python3
"""
Simple test to verify streaming is working correctly
"""
import os
import sys
import time
from dotenv import load_dotenv

# Add the src directory to the path so we can import memfuse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memfuse.llm import OpenAI
from memfuse import MemFuse

load_dotenv(override=True)

def test_streaming():
    """Test streaming functionality directly"""
    
    # Initialize MemFuse and OpenAI client
    memfuse_base_url = os.getenv("MEMFUSE_BASE_URL", "http://127.0.0.1:8000")
    
    try:
        memfuse = MemFuse(base_url=memfuse_base_url)
        memory = memfuse.init(user="streaming_test_user")
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            memory=memory
        )
        
        print("Testing streaming response...")
        print("=" * 50)
        
        # Test streaming
        messages = [{"role": "user", "content": "Tell me a short story about a robot"}]
        
        print("Question: Tell me a short story about a robot")
        print("Response (streaming):")
        print("-" * 30)
        
        response_stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        
        full_response = ""
        chunk_count = 0
        
        for chunk in response_stream:
            if (hasattr(chunk, 'choices') and chunk.choices and 
                len(chunk.choices) > 0 and hasattr(chunk.choices[0], 'delta') and 
                chunk.choices[0].delta and hasattr(chunk.choices[0].delta, 'content') and 
                chunk.choices[0].delta.content):
                
                content = chunk.choices[0].delta.content
                full_response += content
                chunk_count += 1
                
                # Print each chunk as it arrives
                print(content, end='', flush=True)
                
                # Small delay to simulate real streaming
                time.sleep(0.02)
        
        print("\n" + "=" * 50)
        print(f"Streaming completed! Received {chunk_count} chunks.")
        print(f"Total response length: {len(full_response)} characters")
        
        memfuse.close()
        return True
        
    except Exception as e:
        print(f"❌ Error during streaming test: {e}")
        return False

if __name__ == "__main__":
    print("MemFuse Streaming Test")
    print("=" * 40)
    
    success = test_streaming()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Streaming test completed successfully!")
    else:
        print("❌ Streaming test failed.") 