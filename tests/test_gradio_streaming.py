#!/usr/bin/env python3
"""
Minimal Gradio streaming test
"""
import gradio as gr
import time

def simple_streaming_fn(message, history):
    """Simple streaming function for testing"""
    words = message.split()
    response = ""
    
    for word in words:
        response += word + " "
        history_copy = history + [(message, response.strip())]
        yield history_copy, ""
        time.sleep(0.2)  # Slow delay to see streaming clearly

def main():
    with gr.Blocks(title="Streaming Test") as demo:
        gr.Markdown("# Streaming Test")
        
        chatbot = gr.Chatbot(height=300)
        msg = gr.Textbox(placeholder="Type something...")
        
        msg.submit(simple_streaming_fn, [msg, chatbot], [chatbot, msg])
        
    demo.launch(debug=True)

if __name__ == "__main__":
    main() 