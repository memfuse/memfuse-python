from memfuse.llm import Anthropic
from memfuse import MemFuse
import os
from dotenv import load_dotenv
from memfuse.llm import AsyncAnthropic
from memfuse import AsyncMemFuse

# Load environment variables
load_dotenv(override=True)

# Initialize MemFuse with a user context
memfuse = MemFuse()
memory = memfuse.init(user="alice")

# Create Anthropic client with memory
anthropic_client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    memory=memory
)

# Display memory information
print(f"Using memory for conversation: {memory}")

# Check if API key is available
if not os.getenv("ANTHROPIC_API_KEY"):
    print("ANTHROPIC_API_KEY not found. Please set it in your .env file.")
    exit(1)

try:
    # --- Anthropic Example ---
    print("\n--- Anthropic Example ---")
    anthropic_response = anthropic_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        system="You are a knowledgeable assistant with expertise in astronomy.",
        messages=[
            {
                "role": "user", 
                "content": [{"type": "text", "text": "Tell me a fun fact about the Moon."}]
            }
        ]
    )
    
    # Display response content
    if anthropic_response.content and isinstance(anthropic_response.content, list):
        for content_item in anthropic_response.content:
            if hasattr(content_item, 'text'):
                print(content_item.text)
    else:
        print("Anthropic response content not in expected format or empty.")
        
    # Test follow-up to verify memory is working
    print("\n--- Anthropic Follow-up Question ---")
    followup_response = anthropic_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "How does that celestial body affect Earth's tides?"}]
            }
        ]
    )
    
    # Display follow-up response
    if followup_response.content and isinstance(followup_response.content, list):
        for content_item in followup_response.content:
            if hasattr(content_item, 'text'):
                print(content_item.text)
                
except Exception as e:
    print(f"Error during Anthropic API call: {e}")

# --- Async Anthropic Example ---
async def run_async_anthropic_example():
    print("\n🧠 --- Async Anthropic Example ---")
    # Initialize the async MemFuse client
    memfuse_async = AsyncMemFuse()
    try:
        memory_async = await memfuse_async.init(user="alice_async_anthropic")
        print(f"✓ Async Memory initialized for Anthropic: {memory_async.user} | Session: {memory_async.session}")

        # Ensure you have ANTHROPIC_API_KEY set in your .env file
        if os.getenv("ANTHROPIC_API_KEY"):
            anthropic_async_client = AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                memory=memory_async  # Reusing the same memory object
            )
            
            try:
                # Using the proper Anthropic format with system and messages
                anthropic_response_async = await anthropic_async_client.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=1024,
                    system="You are an async knowledgeable assistant with expertise in astronomy.",
                    messages=[
                        {
                            "role": "user", 
                            "content": [{"type": "text", "text": "Tell me an async fun fact about the Moon."}]
                        }
                    ]
                )
                
                if anthropic_response_async.content and isinstance(anthropic_response_async.content, list):
                    for content_item in anthropic_response_async.content:
                        if hasattr(content_item, 'text'):
                            print("Async Anthropic Response:", content_item.text)
                else:
                    print("Async Anthropic response content not in expected format or empty.")
                    
                # Test follow-up to verify memory is working
                print("\n🔄 --- Async Anthropic Follow-up Question ---")
                followup_response_async = await anthropic_async_client.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "How does that celestial body async affect Earth's tides?"}]
                        }
                    ]
                )
                
                if followup_response_async.content and isinstance(followup_response_async.content, list):
                    for content_item in followup_response_async.content:
                        if hasattr(content_item, 'text'):
                            print("Async Anthropic Follow-up:", content_item.text)
                            
            except Exception as e_async:
                print(f"Error during Async Anthropic API call: {e_async}")
        else:
            print("ANTHROPIC_API_KEY not found. Skipping Async Anthropic example.")
    finally:
        await memfuse_async.close()
        print("\n✅ Async MemFuse client for Anthropic closed")

if __name__ == "__main__":
    # Run the sync example
    # (Keep existing __main__ block if it calls the sync part)
    # For demonstration, assuming the sync part is run by just being in the script.
    # If there's a main function for sync, call it here.
    
    # Run the async example
    import asyncio
    print("\n=== Running Async Anthropic Example ===")
    asyncio.run(run_async_anthropic_example())
    print("\n=== Async Anthropic Example Complete ===") 