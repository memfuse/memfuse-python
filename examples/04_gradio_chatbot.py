import gradio as gr
from memfuse.llm import OpenAI  # Use synchronous OpenAI
from memfuse import MemFuse  # Use synchronous MemFuse
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Global username variable
USERNAME = "Jane Doe"
SYSTEM_MESSAGE = (
    "You are a helpful AI assistant with access to a persistent long-term memory. "
    "You can recall, reference, and use information from previous conversations with the user. "
    "Leverage this memory to provide more relevant, helpful, and context-aware answers. "
    "If you remember something from earlier, feel free to mention it. "
    "If the user mentions something from a previous interaction that you don't remember, please apologize and say you must have forgotten."
)

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

        # memfuse_chatbot definition - traditional non-streaming response
        def memfuse_chatbot(message, history):
            # Debug: Print the history format to understand what we're getting
            print(f"DEBUG: Received history type: {type(history)}")
            print(f"DEBUG: History content: {history}")
            
            # Convert gradio history format to messages format
            messages_history = []
            
            # Handle different possible history formats
            if history:
                for i, item in enumerate(history):
                    print(f"DEBUG: History item {i}: type={type(item)}, content={item}")
                    if isinstance(item, dict):
                        # History is already in message format - extract only role and content
                        if 'role' in item and 'content' in item:
                            messages_history.append({
                                "role": item["role"],
                                "content": item["content"]
                            })
                    elif isinstance(item, (list, tuple)):
                        # Handle tuple/list format - could be (user_msg, assistant_msg) or more items
                        if len(item) >= 2:
                            user_msg, assistant_msg = item[0], item[1]
                            messages_history.append({"role": "user", "content": user_msg})
                            if assistant_msg:  # Assistant message might be None during streaming
                                messages_history.append({"role": "assistant", "content": assistant_msg})
                        elif len(item) == 1:
                            # Only user message
                            messages_history.append({"role": "user", "content": item[0]})
                    else:
                        # Unknown format, skip
                        print(f"Unknown history item format: {type(item)}: {item}")
            
            print(f"DEBUG: Final messages_history: {messages_history}")
            current_messages_for_api = [{"role": "system", "content": SYSTEM_MESSAGE}] + messages_history + [{"role": "user", "content": message}]
            print(f"DEBUG: Sending to API: {current_messages_for_api}")

            try:
                # Call the synchronous LLM API WITHOUT streaming (traditional response)
                response_obj = client.chat.completions.create(
                    model="gpt-4o-mini", # Or your preferred model
                    messages=current_messages_for_api, # Pass the combined history and current message
                    stream=False  # Disable streaming for traditional response
                )
                
                # Return the complete response at once
                if response_obj and response_obj.choices and response_obj.choices[0].message:
                    return response_obj.choices[0].message.content
                else:
                    return "Sorry, I didn't receive a proper response."
                        
            except Exception as e:
                print(f"Error calling LLM: {e}")
                return "Sorry, I encountered an error."

        # Create simple ChatInterface for non-streaming responses
        demo = gr.ChatInterface(
            fn=memfuse_chatbot,
            title=f"MemFuse Chatbot Demo (Non-Streaming) - User: {USERNAME}",
            description="Ask any question to see MemFuse in action with an LLM. Responses appear all at once.",
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

        demo.launch(share=False, debug=True) # Launch with debug for better error messages

    except Exception as e:
        print(f"Error during initialization: {e}")
        return
    finally:
        # Ensure the MemFuse client session is closed when the application exits or an error occurs
        if memfuse:
            memfuse.close()

if __name__ == "__main__":
    main()  # No asyncio.run() needed since we're using synchronous client