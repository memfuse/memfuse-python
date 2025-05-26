from memfuse.llm import GeminiClient
from memfuse import MemFuse
import os
from dotenv import load_dotenv
from memfuse.llm import AsyncGeminiClient
from memfuse import AsyncMemFuse

# Load environment variables
load_dotenv(override=True)

# Initialize MemFuse with a user context
memfuse = MemFuse()
memory = memfuse.init(user="alice")

# Create Gemini client with memory
# For Google AI Studio, use GEMINI_API_KEY
# For Vertex AI, use GOOGLE_APPLICATION_CREDENTIALS
gemini_client = GeminiClient(
    memory=memory,
    api_key=os.getenv("GEMINI_API_KEY")
)

# Display memory information
print(f"Using memory for conversation: {memory}")

# Check if API key is available
if not os.getenv("GEMINI_API_KEY"):
    print("GEMINI_API_KEY not found. Please set it in your .env file.")
    exit(1)

try:
    # --- Gemini Example ---
    print("\n--- Gemini Example ---")
    gemini_response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",  # or "gemini-1.5-pro" for more advanced model
        contents="What's a fascinating fact about Saturn?"
    )
    
    # Extract and display response text
    if gemini_response and gemini_response.candidates:
        response_text = ""
        if gemini_response.candidates[0].content and gemini_response.candidates[0].content.parts:
            for part in gemini_response.candidates[0].content.parts:
                if part.text:
                    response_text += part.text
        
        if response_text:
            print(response_text)
        else:
            print("Gemini response content not in expected format or empty.")
    
    # Test follow-up to verify memory is working
    print("\n--- Gemini Follow-up Question ---")
    followup_response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents="Tell me more about that planet's moons."
    )
    
    # Extract and display follow-up response
    if followup_response and followup_response.candidates:
        followup_text = ""
        if followup_response.candidates[0].content and followup_response.candidates[0].content.parts:
            for part in followup_response.candidates[0].content.parts:
                if part.text:
                    followup_text += part.text
        
        if followup_text:
            print(followup_text)
        
except Exception as e:
    print(f"Error during Gemini API call: {e}")

# --- Async Gemini Example ---
async def run_async_gemini_example():
    print("\n💎 --- Async Gemini Example ---")
    # Initialize the async MemFuse client
    memfuse_async = AsyncMemFuse()
    try:
        memory_async = await memfuse_async.init(user="alice_async_gemini")
        print(f"✓ Async Memory initialized for Gemini: {memory_async.user} | Session: {memory_async.session}")

        # Ensure you have GEMINI_API_KEY set in your .env file for Google AI Studio
        if os.getenv("GEMINI_API_KEY"):
            gemini_async_client = AsyncGeminiClient(
                memory=memory_async,  # Reusing the same memory object
                api_key=os.getenv("GEMINI_API_KEY")  # For Google AI Studio
            )
            
            try:
                gemini_response_async = await gemini_async_client.models.generate_content_async(
                    model="gemini-1.5-flash",
                    contents="What's an async fascinating fact about Saturn?"
                )
                
                if gemini_response_async and gemini_response_async.candidates:
                    response_text_async = ""
                    if gemini_response_async.candidates[0].content and gemini_response_async.candidates[0].content.parts:
                        for part in gemini_response_async.candidates[0].content.parts:
                            if part.text:
                                response_text_async += part.text
                    
                    if response_text_async:
                        print("Async Gemini Response:", response_text_async)
                    else:
                        print("Async Gemini response content not in expected format or empty.")
                
                print("\n🔄 --- Async Gemini Follow-up Question ---")
                followup_response_async = await gemini_async_client.models.generate_content_async(
                    model="gemini-1.5-flash",
                    contents="Tell me more async about that planet's moons."
                )
                
                if followup_response_async and followup_response_async.candidates:
                    followup_text_async = ""
                    if followup_response_async.candidates[0].content and followup_response_async.candidates[0].content.parts:
                        for part in followup_response_async.candidates[0].content.parts:
                            if part.text:
                                followup_text_async += part.text
                    
                    if followup_text_async:
                        print("Async Gemini Follow-up:", followup_text_async)
                        
            except Exception as e_async:
                print(f"Error during Async Gemini API call: {e_async}")
        else:
            print("GEMINI_API_KEY not found. Skipping Async Gemini example.")
    finally:
        await memfuse_async.close()
        print("\n✅ Async MemFuse client for Gemini closed")

if __name__ == "__main__":
    # Run the sync example
    # (Keep existing __main__ block if it calls the sync part)
    # For demonstration, assuming the sync part is run by just being in the script.
    # If there's a main function for sync, call it here.

    # Run the async example
    import asyncio
    print("\n=== Running Async Gemini Example ===")
    asyncio.run(run_async_gemini_example())
    print("\n=== Async Gemini Example Complete ===") 