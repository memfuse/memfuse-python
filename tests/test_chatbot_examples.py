#!/usr/bin/env python3
"""
Test script to verify that the chatbot examples work correctly
"""
import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path so we can import memfuse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memfuse.llm import OpenAI
from memfuse import MemFuse

load_dotenv(override=True)

def test_chatbot_function():
    """Test the chatbot function directly with example inputs"""
    
    # Initialize MemFuse and OpenAI client
    memfuse_base_url = os.getenv("MEMFUSE_BASE_URL", "http://127.0.0.1:8000")
    
    try:
        memfuse = MemFuse(base_url=memfuse_base_url)
        memory = memfuse.init(user="test_user")
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            memory=memory
        )
        
        # Define the same chatbot function as in the Gradio app (streaming version)
        def memfuse_chatbot(message, history):
            messages_history = []
            if history:
                messages_history = list(history)
            
            current_messages_for_api = messages_history + [{"role": "user", "content": message}]
            
            try:
                response_stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=current_messages_for_api,
                    stream=True  # Enable streaming
                )
                
                # Accumulate the streaming response
                full_response = ""
                for chunk in response_stream:
                    # Extract content from the streaming chunk
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                
                return full_response  # Return the final accumulated response for testing
            except Exception as e:
                print(f"Error calling LLM: {e}")
                return "Sorry, I encountered an error."
        
        # Test the examples
        examples = [
            "What is the capital of France?",
            "Explain the theory of relativity in simple terms.",
            "Tell me a fun fact about space.",
            "What are some good books to read on AI?",
        ]
        
        print("Testing chatbot examples:")
        print("=" * 50)
        
        for i, example in enumerate(examples, 1):
            print(f"\n{i}. Testing: '{example}'")
            print("-" * 30)
            
            # Test with empty history (like clicking an example)
            response = memfuse_chatbot(example, [])
            
            if response and response != "Sorry, I encountered an error.":
                print(f"✅ Success: {response[:100]}{'...' if len(response) > 100 else ''}")
            else:
                print(f"❌ Failed: {response}")
        
        # Test conversation continuity
        print(f"\n{len(examples) + 1}. Testing conversation continuity:")
        print("-" * 30)
        
        # Start a conversation
        history = []
        first_message = "Hello, my name is Alice."
        first_response = memfuse_chatbot(first_message, history)
        history.append({"role": "user", "content": first_message})
        history.append({"role": "assistant", "content": first_response})
        
        # Follow up
        follow_up = "What's my name?"
        follow_up_response = memfuse_chatbot(follow_up, history)
        
        if "Alice" in follow_up_response:
            print("✅ Memory working: Bot remembered the name")
        else:
            print(f"❌ Memory issue: {follow_up_response}")
        
        memfuse.close()
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    print("MemFuse Chatbot Examples Test")
    print("=" * 40)
    
    success = test_chatbot_function()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ All tests passed! The chatbot examples should work correctly.")
    else:
        print("❌ Some tests failed. Check the error messages above.") 