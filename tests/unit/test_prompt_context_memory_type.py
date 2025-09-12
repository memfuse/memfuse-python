import pytest
from memfuse.prompts.prompt_context import PromptContext

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


def test_openai_prompt_uses_memory_type_without_role():
    # Retrieved memories contain only memory_type and content (no role)
    memories = [
        {
            "id": "m1",
            "content": "Some remembered fact",
            "memory_type": "episodic",
            "metadata": {"scope": "cross_session"},
        },
        {
            "id": "m2",
            "content": "Recent in-session detail",
            "memory_type": "message",
            "metadata": {"scope": "in_session"},
        },
    ]

    ctx = PromptContext(
        query_messages=[{"role": "user", "content": "Hi"}],
        retrieved_memories=memories,
        retrieved_chat_history=[],
        max_chat_history=2,
    )

    msgs = ctx.compose_for_openai()
    print("\n=== PROMPT DEBUG: OpenAI compose_for_openai() ===")
    for i, m in enumerate(msgs):
        print(f"[{i}] ROLE={m['role']}\n{m['content']}\n")
    # Expect first system instruction, then one or two system wrappers with memory labels
    assert msgs[0]["role"] == "system"
    # Long-term memory section should include [EPISODIC]
    full_text = "\n".join(m["content"] for m in msgs if m["role"] == "system")
    assert "[EPISODIC]" in full_text
    assert "from USER" not in full_text and "from ASSISTANT" not in full_text


def test_anthropic_prompt_uses_memory_type_without_role():
    memories = [
        {
            "id": "m1",
            "content": "Some remembered fact",
            "memory_type": "semantic",
            "metadata": {"scope": "cross_session"},
        }
    ]

    ctx = PromptContext(
        query_messages=[{"role": "user", "content": "Hi"}],
        retrieved_memories=memories,
        retrieved_chat_history=[],
        max_chat_history=2,
    )

    system_prompt, anth_msgs = ctx.compose_for_anthropic()
    print("\n=== PROMPT DEBUG: Anthropic compose_for_anthropic() ===")
    print("[SYSTEM]\n" + system_prompt + "\n")
    for i, m in enumerate(anth_msgs):
        print(f"[{i}] ROLE={m['role']}\n{m['content']}\n")
    
    # Also visualize Gemini composition for completeness
    gem_msgs = ctx.compose_for_gemini()
    print("\n=== PROMPT DEBUG: Gemini compose_for_gemini() ===")
    for i, m in enumerate(gem_msgs):
        print(f"[{i}] ROLE={m['role']}\n{m['content']}\n")
    assert "[SEMANTIC]" in system_prompt
    assert "from USER" not in system_prompt and "from ASSISTANT" not in system_prompt
