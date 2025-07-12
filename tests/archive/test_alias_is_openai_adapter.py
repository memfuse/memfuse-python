# tests/test_import_alias.py
from memfuse.llm import OpenAI

def test_alias_is_adapter():
    from memfuse.llm.openai_adapter import MemOpenAI
    assert issubclass(OpenAI, MemOpenAI)