"""E2E test to verify that MemFuse keeps conversation context across turns.

The test mimics the quick-start example:
1. Ask for a Mars fact.
2. Immediately ask a follow-up referring to "that planet".

It passes if the follow-up answer still explicitly references Mars, proving
that the memory pipeline injected the prior context into the LLM prompt.
"""

import os
import re
import time
import pytest
from dotenv import load_dotenv

from memfuse.llm import OpenAI
from memfuse import MemFuse

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
def test_memory_followup_includes_mars_reference():
    """The follow-up answer should still reference *Mars*.

    This demonstrates that MemFuse injected the first turn into the prompt
    before calling the LLM, i.e. the conversational memory worked.
    """

    # ---------------------------------------------------------------------
    # Skip logic – only run when keys & server are available
    # ---------------------------------------------------------------------
    load_dotenv(override=True)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("OPENAI_API_KEY not set – skipping E2E memory test")

    memfuse_base_url = os.getenv("MEMFUSE_BASE_URL", "http://127.0.0.1:8000")

    # ---------------------------------------------------------------------
    # Arrange – create MemFuse session & OpenAI client with memory attached
    # ---------------------------------------------------------------------
    try:
        memfuse = MemFuse(base_url=memfuse_base_url, timeout=30)
    except Exception as exc:
        pytest.skip(f"Cannot connect to MemFuse server at {memfuse_base_url}: {exc}")

    memory = memfuse.init(user="e2e_memory_test_user")

    print(f"🔍 Using OpenAI API key: {openai_key}")
    client = OpenAI(
        api_key=openai_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
        memory=memory,
    )

    # ---------------------------------------------------------------------
    # Act – 1️⃣ initial question
    # ---------------------------------------------------------------------
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me something interesting about Mars."}],
    )

    # Small delay to ensure the MemFuse backend has persisted the message
    time.sleep(1)

    # ---------------------------------------------------------------------
    # Act – 2️⃣ follow-up question that relies on memory
    # ---------------------------------------------------------------------
    response = client.chat.completions.create(
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

        print(f"\n🔍 RAGAS Semantic Similarity Analysis:")
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

    # ---------------------------------------------------------------------
    # Teardown
    # ---------------------------------------------------------------------
    memfuse.close() 