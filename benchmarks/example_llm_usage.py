#!/usr/bin/env python
"""
Example usage of the unified LLM class.
Demonstrates both synchronous and asynchronous usage.
"""

import os
import asyncio
import logging
from llm import UnifiedLLM, Provider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def example_openai():
    """Example using OpenAI provider (synchronous)."""
    print("\n=== OpenAI Sync Example ===")
    
    try:
        # Initialize OpenAI client
        llm = UnifiedLLM(provider=Provider.OPENAI, model="gpt-4o")
        
        # Generate text
        prompt = "Write a short poem about artificial intelligence."
        response = llm.generate_text(
            prompt=prompt,
            temperature=0.7,
            max_tokens=200
        )
        
        print(f"Model: {response.model}")
        print(f"Response: {response.text}")
        if response.usage:
            print(f"Usage: {response.usage}")
            
    except Exception as e:
        print(f"OpenAI sync example failed: {e}")

async def example_openai_async():
    """Example using OpenAI provider (asynchronous)."""
    print("\n=== OpenAI Async Example ===")
    
    try:
        # Initialize OpenAI client
        llm = UnifiedLLM(provider=Provider.OPENAI, model="gpt-4o")
        
        # Generate text asynchronously
        prompt = "Write a short haiku about technology."
        response = await llm.agenerate_text(
            prompt=prompt,
            temperature=0.8,
            max_tokens=100
        )
        
        print(f"Model: {response.model}")
        print(f"Response: {response.text}")
        if response.usage:
            print(f"Usage: {response.usage}")
            
    except Exception as e:
        print(f"OpenAI async example failed: {e}")

def example_gemini():
    """Example using Gemini provider (synchronous)."""
    print("\n=== Gemini Sync Example ===")
    
    try:
        # Initialize Gemini client
        llm = UnifiedLLM(provider=Provider.GEMINI, model="gemini-2.0-flash-001")
        
        # Generate text
        prompt = "Explain quantum computing in simple terms."
        response = llm.generate_text(
            prompt=prompt,
            temperature=0.5,
            max_tokens=150
        )
        
        print(f"Model: {response.model}")
        print(f"Response: {response.text}")
        if response.usage:
            print(f"Usage: {response.usage}")
            
    except Exception as e:
        print(f"Gemini sync example failed: {e}")

async def example_gemini_async():
    """Example using Gemini provider (asynchronous)."""
    print("\n=== Gemini Async Example ===")
    
    try:
        # Initialize Gemini client
        llm = UnifiedLLM(provider=Provider.GEMINI, model="gemini-2.0-flash-001")
        
        # Generate text asynchronously
        prompt = "What are the benefits of renewable energy?"
        response = await llm.agenerate_text(
            prompt=prompt,
            temperature=0.6,
            max_tokens=120
        )
        
        print(f"Model: {response.model}")
        print(f"Response: {response.text}")
        if response.usage:
            print(f"Usage: {response.usage}")
            
    except Exception as e:
        print(f"Gemini async example failed: {e}")

async def example_concurrent_requests():
    """Example showing concurrent async requests to multiple providers."""
    print("\n=== Concurrent Async Example ===")
    
    tasks = []
    
    # Add OpenAI task if available
    if os.getenv("OPENAI_API_KEY"):
        async def openai_task():
            llm = UnifiedLLM(provider=Provider.OPENAI)
            return await llm.agenerate_text("Tell me a fun fact about space.", max_tokens=100)
        tasks.append(("OpenAI", openai_task()))
    
    # Add Gemini task if available
    if os.getenv("GEMINI_API_KEY"):
        async def gemini_task():
            llm = UnifiedLLM(provider=Provider.GEMINI)
            return await llm.agenerate_text("Tell me a fun fact about the ocean.", max_tokens=100)
        tasks.append(("Gemini", gemini_task()))
    
    if tasks:
        try:
            # Run tasks concurrently
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for i, (provider_name, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    print(f"{provider_name} failed: {result}")
                else:
                    print(f"{provider_name} response: {result.text[:100]}...")
        except Exception as e:
            print(f"Concurrent example failed: {e}")
    else:
        print("No API keys available for concurrent example")

def example_provider_availability():
    """Check which providers are available."""
    print("\n=== Provider Availability ===")
    
    available = UnifiedLLM.list_available_providers()
    for provider, is_available in available.items():
        status = "✓ Available" if is_available else "✗ Not Available"
        print(f"{provider.capitalize()}: {status}")

async def main():
    """Main function to run examples."""
    print("Unified LLM Client Examples")
    print("=" * 40)
    
    # Check provider availability
    example_provider_availability()
    
    # Synchronous examples
    print("\n" + "=" * 20 + " SYNC EXAMPLES " + "=" * 20)
    
    # Try OpenAI sync if available and configured
    if os.getenv("OPENAI_API_KEY"):
        example_openai()
    else:
        print("\n=== OpenAI Sync Example ===")
        print("Skipped: OPENAI_API_KEY not set")
    
    # Try Gemini sync if available and configured
    if os.getenv("GEMINI_API_KEY"):
        example_gemini()
    else:
        print("\n=== Gemini Sync Example ===")
        print("Skipped: GEMINI_API_KEY not set")
    
    # Asynchronous examples
    print("\n" + "=" * 20 + " ASYNC EXAMPLES " + "=" * 20)
    
    # Try OpenAI async if available and configured
    if os.getenv("OPENAI_API_KEY"):
        await example_openai_async()
    else:
        print("\n=== OpenAI Async Example ===")
        print("Skipped: OPENAI_API_KEY not set")
    
    # Try Gemini async if available and configured
    if os.getenv("GEMINI_API_KEY"):
        await example_gemini_async()
    else:
        print("\n=== Gemini Async Example ===")
        print("Skipped: GEMINI_API_KEY not set")
    
    # Concurrent example
    print("\n" + "=" * 18 + " CONCURRENT EXAMPLES " + "=" * 18)
    await example_concurrent_requests()

if __name__ == "__main__":
    asyncio.run(main())