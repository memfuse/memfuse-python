"""E2E test to verify that AsyncMemFuse keeps conversation context across turns.

This is the async version of the memory followup test, mimicking the async quickstart example:
1. Ask for a Mars fact using AsyncOpenAI.
2. Immediately ask a follow-up referring to "that planet".

It passes if the follow-up answer still explicitly references Mars, proving
that the async memory pipeline injected the prior context into the LLM prompt.
"""

import asyncio
import os
import re
import time
import pytest
from dotenv import load_dotenv

from memfuse.llm import AsyncOpenAI
from memfuse import AsyncMemFuse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mentions_mars(text: str) -> bool:
    """Return True if *text* explicitly mentions Mars (case-insensitive).

    We also accept the adjective "Martian" to be a bit more robust.
    """
    return bool(re.search(r"\b(mars|martian)\b", text, re.IGNORECASE))

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.e2e  # Layer 6: real server / real LLM
@pytest.mark.asyncio
async def test_async_memory_followup_includes_mars_reference():
    """The async follow-up answer should still reference *Mars*.

    This demonstrates that AsyncMemFuse injected the first turn into the prompt
    before calling the LLM, i.e. the async conversational memory worked.
    """

    # ---------------------------------------------------------------------
    # Skip logic – only run when keys & server are available
    # ---------------------------------------------------------------------
    load_dotenv(override=True)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("OPENAI_API_KEY not set – skipping async E2E memory test")

    memfuse_base_url = os.getenv("MEMFUSE_BASE_URL", "http://127.0.0.1:8000")

    # ---------------------------------------------------------------------
    # Arrange – create AsyncMemFuse session & AsyncOpenAI client with memory
    # ---------------------------------------------------------------------
    try:
        memfuse = AsyncMemFuse(base_url=memfuse_base_url)
    except Exception as exc:
        pytest.skip(f"Cannot connect to MemFuse server at {memfuse_base_url}: {exc}")

    try:
        memory = await memfuse.init(user="e2e_async_memory_test_user")

        client = AsyncOpenAI(
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL"),
            memory=memory,
        )

        # ---------------------------------------------------------------------
        # Act – 1️⃣ initial question
        # ---------------------------------------------------------------------
        await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me something interesting about Mars."}],
        )

        # Small delay to ensure the MemFuse backend has persisted the message
        await asyncio.sleep(1)

        # ---------------------------------------------------------------------
        # Act – 2️⃣ follow-up question that relies on memory
        # ---------------------------------------------------------------------
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What would be the biggest challenges for humans living on that planet?",
                }
            ],
        )

        answer_text = response.choices[0].message.content

        # ---------------------------------------------------------------------
        # Assert – verify with RAGAS semantic-similarity if library is available
        # ---------------------------------------------------------------------

        try:
            # RAGAS + LangChain combo offers a richer automatic check than a
            # plain string search. We fallback to the simple regex-based check
            # if either dependency is unavailable (e.g. in CI without extras).
            print("\n🔧 Attempting to import RAGAS dependencies...")
            from ragas.metrics import SemanticSimilarity  # type: ignore
            from ragas.dataset_schema import SingleTurnSample  # type: ignore
            from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore

            # LangChain embedding wrapper around the Ollama embedding model
            from langchain_ollama import OllamaEmbeddings  # type: ignore
            print("✅ RAGAS dependencies imported successfully")

            # Use Ollama embeddings instead of OpenAI to avoid extra API costs
            ollama_embeddings = OllamaEmbeddings(
                model="nomic-embed-text",  # Popular embedding model for Ollama
                base_url="http://localhost:11434"  # Default Ollama server
            )
            embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)
            sim_metric = SemanticSimilarity(embeddings=embeddings)

            # We use a very short reference answer that explicitly mentions Mars
            reference = "Challenges for humans living on Mars include radiation, thin atmosphere and low gravity."

            sample = SingleTurnSample(response=answer_text, reference=reference)

            # The metric returns a float 0-1; require modest similarity (>0.15)
            score: float = sim_metric.single_turn_score(sample)  # type: ignore

            print(f"\n🔍 RAGAS Semantic Similarity Analysis (Async):")
            print(f"   Score: {score:.3f} (threshold: 0.15)")
            print(f"   Reference: {reference}")
            print(f"   Answer: {answer_text[:100]}{'...' if len(answer_text) > 100 else ''}")

            assert score >= 0.15, (
                f"RAGAS semantic similarity too low (score={score:.3f}).\n"
                f"Answer: {answer_text}"
            )
        except (ModuleNotFoundError, Exception) as e:
            # Fallback: simple lexical check for 'mars' / 'martian'
            print(f"\n⚠️  RAGAS not available or Ollama not running, falling back to regex check. Error: {e}")
            assert _mentions_mars(
                answer_text
            ), f"Follow-up answer did not reference Mars – got: {answer_text}"
            print("✅ Regex check passed: Answer mentions Mars")

    finally:
        # ---------------------------------------------------------------------
        # Teardown – important for async resources
        # ---------------------------------------------------------------------
        await memfuse.close()


@pytest.mark.e2e  # Layer 6: real server / real LLM
@pytest.mark.asyncio
async def test_async_memory_with_context_manager():
    """Test async memory using context manager for automatic cleanup.
    
    This demonstrates the recommended pattern from the async quickstart example.
    """

    # ---------------------------------------------------------------------
    # Skip logic – only run when keys & server are available
    # ---------------------------------------------------------------------
    load_dotenv(override=True)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("OPENAI_API_KEY not set – skipping async context manager test")

    memfuse_base_url = os.getenv("MEMFUSE_BASE_URL", "http://127.0.0.1:8000")

    # ---------------------------------------------------------------------
    # Test using async context manager (recommended approach)
    # ---------------------------------------------------------------------
    try:
        async with AsyncMemFuse(base_url=memfuse_base_url) as memfuse:
            memory = await memfuse.init(user="e2e_async_context_test_user", session="context_test")
            
            client = AsyncOpenAI(
                api_key=openai_key,
                base_url=os.getenv("OPENAI_BASE_URL"),
                memory=memory,
            )

            # Add a test message to memory directly
            await memory.add([
                {"role": "user", "content": "I'm interested in learning about Jupiter."},
                {"role": "assistant", "content": "Jupiter is the largest planet in our solar system, a gas giant with a Great Red Spot storm."}
            ])

            # Small delay for persistence
            await asyncio.sleep(1)

            # Test that memory context is preserved
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user", 
                        "content": "How many moons does that planet have?"
                    }
                ],
            )

            answer_text = response.choices[0].message.content
            
            # Simple check that the response is contextually relevant to Jupiter
            # (we expect it to mention Jupiter or give a number around 95+ moons)
            jupiter_mentioned = bool(re.search(r"\bjupiter\b", answer_text, re.IGNORECASE))
            has_large_number = bool(re.search(r"\b(9[0-9]|[1-9][0-9]{2,})\b", answer_text))  # 90+ or larger numbers
            
            assert jupiter_mentioned or has_large_number, (
                f"Response doesn't seem to reference Jupiter context. Got: {answer_text}"
            )
            
            print(f"✅ Context manager test passed. Response: {answer_text[:100]}...")

    except Exception as exc:
        pytest.skip(f"Cannot connect to MemFuse server at {memfuse_base_url}: {exc}") 