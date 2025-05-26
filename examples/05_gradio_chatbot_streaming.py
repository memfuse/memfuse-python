import gradio as gr
from memfuse.llm import OpenAI
from memfuse import MemFuse
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Global username variable
USERNAME = "John Doe"

def main():
    # Make MemFuse base URL configurable via environment variable
    memfuse_base_url = os.getenv("MEMFUSE_BASE_URL", "http://127.0.0.1:8000")
    
    try:
        memfuse = MemFuse(base_url=memfuse_base_url)  # Use synchronous client
    except Exception as e:
        print(f"Failed to initialize MemFuse client: {e}")
        print("Please make sure MemFuse server is running and accessible.")
        return
    
    # Initialize memory and client to None for safety in the finally block
    # if an error occurs during their initialization.
    memory = None
    client = None

    try:
        # Initialize MemFuse and OpenAI client (synchronously)
        memory = memfuse.init(user=USERNAME) # Using global username

        client = OpenAI(  # Use synchronous OpenAI
            api_key=os.getenv("OPENAI_API_KEY"), # Make sure OPENAI_API_KEY and OPENAI_BASE_URL are in your .env file
            base_url=os.getenv("OPENAI_BASE_URL"),
            memory=memory # memory is now an instance of synchronous Memory
        )

        # Use the simpler ChatInterface with proper streaming
        def memfuse_chatbot_simple(message, history):
            # Convert gradio history format to messages format
            messages_history = []
            
            # With type="messages", history is already a list of message dicts
            if history:
                messages_history = list(history)  # Copy the existing history
            
            current_messages_for_api = messages_history + [{"role": "user", "content": message}]

            try:
                # Call the synchronous LLM API with streaming enabled
                response_stream = client.chat.completions.create(
                    model="gpt-4o-mini", # Or your preferred model
                    messages=current_messages_for_api, # Pass the combined history and current message
                    stream=True  # Enable streaming
                )
                
                # Accumulate the streaming response
                partial_response = ""
                for chunk in response_stream:
                    # Extract content from the streaming chunk
                    if (hasattr(chunk, 'choices') and chunk.choices and 
                        len(chunk.choices) > 0 and hasattr(chunk.choices[0], 'delta') and 
                        chunk.choices[0].delta and hasattr(chunk.choices[0].delta, 'content') and 
                        chunk.choices[0].delta.content):
                        
                        partial_response += chunk.choices[0].delta.content
                        yield partial_response  # Yield just the partial response
                        
            except Exception as e:
                print(f"Error calling LLM: {e}")
                yield "Sorry, I encountered an error."

        # Create the interface using ChatInterface which handles streaming better
        iface = gr.ChatInterface(
            fn=memfuse_chatbot_simple,
            title=f"🤖 MemFuse Streaming Chatbot - User: {USERNAME}",
            description="Ask any question to see MemFuse in action with real-time streaming responses!",
            theme="ocean",
            examples=[
                "What is the capital of France?",
                "Explain the theory of relativity in simple terms.",
                "Tell me a fun fact about space.",
                "What are some good books to read on AI?",
            ],
            cache_examples=False,
            chatbot=gr.Chatbot(
                height=400,
                type="messages"
            )
        )

        iface.launch(share=False, debug=True)

    except Exception as e:
        print(f"Error during initialization: {e}")
        return
    finally:
        # Ensure the MemFuse client session is closed when the application exits or an error occurs
        if memfuse:
            memfuse.close()

if __name__ == "__main__":
    main()  # No asyncio.run() needed since we're using synchronous client 