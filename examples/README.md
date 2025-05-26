# MemFuse Examples

This directory contains example scripts demonstrating how to use the MemFuse Python client library.

## Examples Overview

### 01_quickstart.py

A comprehensive example showing how to use the **synchronous** MemFuse APIs, including:

- Synchronous OpenAI integration
- Basic memory operations (add, query, list messages)
- Error handling

### 02_async_quickstart.py

A comprehensive example showing how to use the **asynchronous** MemFuse APIs, including:

- Async OpenAI integration
- Async memory operations with proper `await` syntax
- Async context managers for automatic resource cleanup
- Concurrent operations and better performance for I/O-bound tasks

### 03_continued_conversation.py

A focused example demonstrating **conversation context persistence**, including:

- Multi-turn conversations with automatic context retention
- How MemFuse maintains conversation history without manual tracking
- Seamless follow-up questions without repeating context
- Testing context switching between different topics

### 04_gradio_chatbot.py

A **web-based chatbot interface** using Gradio with traditional responses, including:

- Interactive web UI for chatting with MemFuse-powered LLMs
- Synchronous OpenAI integration with MemFuse memory
- Non-streaming responses (complete responses appear at once)
- Gradio ChatInterface with custom examples and styling
- Proper error handling and resource cleanup

### 05_gradio_chatbot_streaming.py

A **web-based chatbot interface** with **real-time streaming responses**, including:

- Interactive web UI with live streaming text generation
- Synchronous OpenAI integration with streaming enabled
- Real-time token-by-token response display
- Enhanced user experience with immediate feedback
- Gradio ChatInterface optimized for streaming

### 06_anthropic_example.py

An example demonstrating **synchronous Anthropic integration**, including:

- How to use MemFuse with Anthropic's Claude models
- Synchronous API calls to Anthropic
- Basic memory operations tailored for Anthropic usage
- Asynchronous Anthropic integration with `async/await`

### 07_gemini_example.py

An example demonstrating **synchronous Gemini integration**, including:

- How to use MemFuse with Google's Gemini models
- Synchronous API calls to Gemini
- Basic memory operations tailored for Gemini usage
- Asynchronous Gemini integration with `async/await`
