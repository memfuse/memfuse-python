#!/usr/bin/env python
import os
import sys
import logging
from dotenv import load_dotenv
import google.genai as genai
from google.genai.errors import ClientError
from openai import OpenAI
from openai import OpenAIError

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_gemini_api():
    """Check if Gemini API is ready and accessible."""
    
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return False
    
    try:
        # Configure the client
        from google.genai import types
        
        if base_url:
            logger.info(f"Using custom base URL: {base_url}")
            http_options = types.HttpOptions(base_url=base_url)
            client = genai.Client(api_key=api_key, http_options=http_options)
        else:
            client = genai.Client(api_key=api_key)
        
        # Test with a simple model list or generation call
        logger.info("Testing Gemini API connection...")
        
        # Try to make a simple generation request
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Hello, can you respond with just 'OK'?"
        )
        
        if response and response.text:
            logger.info("SUCCESS: Gemini API is ready and responding correctly")
            logger.info(f"Response: {response.text.strip()}")
            return True
        else:
            logger.error("ERROR: Gemini API responded but with empty content")
            return False
            
    except ClientError as e:
        # Check if it's a quota/rate limit error, which means the API is accessible
        if "RESOURCE_EXHAUSTED" in str(e) or "RATE_LIMIT_EXCEEDED" in str(e) or "429" in str(e):
            logger.info("SUCCESS: Gemini API is accessible (quota/rate limit reached)")
            logger.info(f"Rate limit details: {e}")
            return True
        else:
            logger.error(f"ERROR: Gemini API client error: {e}")
            return False
    except Exception as e:
        logger.error(f"ERROR: Unexpected error testing Gemini API: {e}")
        return False

def check_openai_api():
    """Check if OpenAI API is ready and accessible."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    openai_model = os.getenv("OPENAI_COMPATIBLE_MODEL")
    
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return False
    
    try:
        # Configure the client
        if base_url:
            logger.info(f"Using custom base URL: {base_url}")
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        
        # Test with a simple completion request
        logger.info("Testing OpenAI API connection...")
        
        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": "Hello, can you respond with just 'OK'?"}],
            max_tokens=10
        )
        
        if response and response.choices and response.choices[0].message.content:
            logger.info("SUCCESS: OpenAI API is ready and responding correctly")
            logger.info(f"Response: {response.choices[0].message.content.strip()}")
            return True
        else:
            logger.error("ERROR: OpenAI API responded but with empty content")
            return False
            
    except OpenAIError as e:
        # Check if it's a quota/rate limit error, which means the API is accessible
        if "rate_limit_exceeded" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
            logger.info("SUCCESS: OpenAI API is accessible (quota/rate limit reached)")
            logger.info(f"Rate limit details: {e}")
            return True
        else:
            logger.error(f"ERROR: OpenAI API client error: {e}")
            return False
    except Exception as e:
        logger.error(f"ERROR: Unexpected error testing OpenAI API: {e}")
        return False

def main():
    """Main function to check API readiness."""
    logger.info("Checking API readiness...")
    
    gemini_success = check_gemini_api()
    openai_success = check_openai_api()
    
    if gemini_success and openai_success:
        logger.info("All checks passed - Both APIs are ready!")
        sys.exit(0)
    else:
        if not gemini_success:
            logger.error("Gemini API check failed")
        if not openai_success:
            logger.error("OpenAI API check failed")
        sys.exit(1)

if __name__ == "__main__":
    main()