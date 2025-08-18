import os
import json
import logging
import re
from typing import Optional
from openai import OpenAI
from tests.utils.data_models import MultipleChoiceResponse
from tests.utils.prompts import Prompt


def _extract_json_from_markdown(text: str) -> str:
    """Extract JSON content from markdown code blocks."""
    # Try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # If no markdown blocks found, return original text
    return text.strip()


def call_openai_compatible(
        prompt: Prompt,
        model: Optional[str] = None,
        choices_length: Optional[int] = None,
        temperature: float = 0.1,
        logger: Optional[logging.Logger] = None
    ) -> MultipleChoiceResponse:
    """Call OpenAI compatible API with structured output."""

    # Initialize logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)

    # Get model from parameter or environment variable, with fallback to default
    if model is None:
        model = os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini")

    # Get API configuration from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        return MultipleChoiceResponse(
            index=0,
            reasoning="Error: OPENAI_API_KEY not configured",
            description="Configuration error"
        )
    
    # Initialize OpenAI client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = OpenAI(**client_kwargs)
    
    system_prompt, user_prompt = prompt.system, prompt.user
    
    try:
        # Try structured output first
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=temperature
        )
        
        # Parse the response content as JSON
        response_content = response.choices[0].message.content
        if not response_content:
            logger.error("Received empty response from OpenAI compatible API")
            return MultipleChoiceResponse(
                index=0,
                reasoning="Empty response from API",
                description="Error: Empty response"
            )

        logger.debug(f"OpenAI API response: {response_content}")

        # Try to parse as structured response
        try:
            # First, try to extract JSON from markdown code blocks if present
            cleaned_content = _extract_json_from_markdown(response_content)

            # Check if the cleaned content looks like JSON (starts with { and ends with })
            if not (cleaned_content.strip().startswith('{') and cleaned_content.strip().endswith('}')):
                logger.debug(f"Response doesn't look like JSON, falling back to text parsing: {cleaned_content[:100]}...")
                return _parse_text_response(response_content, choices_length, logger)

            parsed_json = json.loads(cleaned_content)

            # Handle different possible JSON structures
            if "index" in parsed_json and "reasoning" in parsed_json:
                # Direct MultipleChoiceResponse format
                parsed_response = MultipleChoiceResponse(**parsed_json)
            elif "choice" in parsed_json:
                # Alternative format with "choice" field
                parsed_response = MultipleChoiceResponse(
                    index=int(parsed_json.get("choice", 0)),
                    reasoning=parsed_json.get("reasoning", parsed_json.get("explanation", "No explanation provided")),
                    description=parsed_json.get("description", "Response from OpenAI compatible API")
                )
            else:
                # Try to extract index from any numeric field
                index = 0
                for key in ["index", "choice", "answer", "selected"]:
                    if key in parsed_json:
                        try:
                            index = int(parsed_json[key])
                            break
                        except (ValueError, TypeError):
                            continue

                reasoning = parsed_json.get("reasoning",
                           parsed_json.get("explanation",
                           parsed_json.get("rationale", "No explanation provided")))

                parsed_response = MultipleChoiceResponse(
                    index=index,
                    reasoning=str(reasoning),
                    description="Response from OpenAI compatible API"
                )

            # Validate choice index
            if choices_length is not None and not (0 <= parsed_response.index < choices_length):
                logger.warning(f"Model returned choice index {parsed_response.index} which is out of range (0-{choices_length-1}). Defaulting to 0.")
                parsed_response.index = 0

            return parsed_response

        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON response: {e}. Falling back to text parsing.")
            # Fall back to text parsing
            return _parse_text_response(response_content, choices_length, logger)
    
    except Exception as e:
        logger.error(f"Error using structured completion with OpenAI compatible API: {e}")
        
        # Fallback to regular completion without structured output
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            
            response_text = response.choices[0].message.content
            if not response_text:
                logger.error("Received empty fallback response from OpenAI compatible API")
                return MultipleChoiceResponse(
                    index=0,
                    reasoning="Empty fallback response from API",
                    description="Error: Empty fallback response"
                )

            logger.info(f"OpenAI compatible API fallback response: {response_text}")

            return _parse_text_response(response_text, choices_length, logger)
            
        except Exception as e2:
            logger.error(f"Error using regular completion with OpenAI compatible API: {e2}")
            return MultipleChoiceResponse(
                index=0,
                reasoning=f"Error calling OpenAI compatible API: {str(e2)}",
                description="Error response"
            )


def _parse_text_response(response_text: str, choices_length: Optional[int] = None, logger: Optional[logging.Logger] = None) -> MultipleChoiceResponse:
    """Parse text response to extract choice and reasoning."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Try to extract JSON from the response text
        import re
        json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed_json = json.loads(json_str)
            
            # Extract index and reasoning
            index = 0
            for key in ["index", "choice", "answer", "selected"]:
                if key in parsed_json:
                    try:
                        index = int(parsed_json[key])
                        break
                    except (ValueError, TypeError):
                        continue
            
            reasoning = parsed_json.get("reasoning", 
                       parsed_json.get("explanation", 
                       parsed_json.get("rationale", "No explanation provided")))
            
            if choices_length is not None and not (0 <= index < choices_length):
                logger.warning(f"Parsed choice index {index} out of range. Defaulting to 0.")
                index = 0
            
            return MultipleChoiceResponse(
                index=index,
                reasoning=str(reasoning),
                description="Parsed from text response"
            )
    
    except Exception as e:
        logger.warning(f"Failed to parse text response: {e}")
    
    # Final fallback - try to find a number in the response
    import re
    numbers = re.findall(r'\b\d+\b', response_text)
    index = 0
    if numbers:
        try:
            index = int(numbers[0])
            if choices_length is not None and not (0 <= index < choices_length):
                index = 0
        except ValueError:
            index = 0
    
    return MultipleChoiceResponse(
        index=index,
        reasoning=response_text[:200] + "..." if len(response_text) > 200 else response_text,
        description="Fallback text parsing"
    )
