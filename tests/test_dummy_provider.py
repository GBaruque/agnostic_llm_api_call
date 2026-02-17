import pytest

from llm import LLMClient
from llm.providers import DummyProvider


def test_dummy_provider_echo():
    """Test that DummyProvider echoes user messages."""
    client = LLMClient(DummyProvider, api_key="test")
    
    # Ask a question (memory is managed automatically)
    response = client.ask("Hello Dummy!")

    assert "Hello Dummy!" in response


def test_conversation_with_memory():
    """Test that conversation memory is maintained across multiple asks."""
    client = LLMClient(
        DummyProvider,
        api_key="test",
        system_prompt="You are a helpful assistant."
    )
    
    # First message
    resp1 = client.ask("What is 2+2?")
    assert "2+2" in resp1
    
    # Second message - memory should be maintained
    resp2 = client.ask("My previous question contains 2+2?")
    ltst_qstn_content = client.memory.as_openai()[-3].get("content") or ""
    assert (("echo" in ltst_qstn_content) and ("2+2" in ltst_qstn_content))  # Check last user message in memory
    
    # Verify memory contains both exchanges
    history = client.get_history()
    assert len(history) == 5  # system + user1 + assistant1 + user2 + assistant2
    assert history[0].role == "system"
    assert history[1].role == "user"
    assert history[2].role == "assistant"
    assert history[3].role == "user"
    assert history[4].role == "assistant"


def test_reset_memory():
    """Test that reset() clears conversation history."""
    client = LLMClient(DummyProvider, api_key="test")
    
    client.ask("First message")
    assert len(client.get_history()) > 0
    
    client.reset()
    assert len(client.get_history()) == 0


def test_multiple_sections():
    """Test that client can manage multiple independent conversation sections."""
    client = LLMClient(DummyProvider, api_key="test")
    
    # Section 1: python-talk
    client.ask("Tell me about Python", section="python-talk")
    resp_py = client.ask("Why is it popular?", section="python-talk")
    assert "Python" in resp_py or "popular" in resp_py
    
    # Section 2: js-talk (independent)
    client.ask("Tell me about JavaScript", section="js-talk")
    resp_js = client.ask("Where is it used?", section="js-talk")
    assert "JavaScript" in resp_js or "used" in resp_js
    
    # Verify sections are independent
    history_py = client.get_history(section="python-talk")
    history_js = client.get_history(section="js-talk")
    
    assert len(history_py) == 4  # 2 user + 2 assistant
    assert len(history_js) == 4  # 2 user + 2 assistant
    assert "Python" in history_py[0].content
    assert "JavaScript" in history_js[0].content


def test_clear_section():
    """Test that clear_section() only clears the specified section."""
    client = LLMClient(DummyProvider, api_key="test")
    
    # Create two sections
    client.ask("Question 1", section="section-a")
    client.ask("Question 2", section="section-b")
    
    assert len(client.get_history(section="section-a")) == 2
    assert len(client.get_history(section="section-b")) == 2
    
    # Clear section-a
    client.clear_section(section="section-a")
    
    assert len(client.get_history(section="section-a")) == 0
    assert len(client.get_history(section="section-b")) == 2


def test_default_section():
    """Test that default section is used when no section specified."""
    client = LLMClient(DummyProvider, api_key="test", system_prompt="System message")
    
    client.ask("Message 1")  # No section specified, uses default
    client.ask("Message 2")  # No section specified, uses default
    
    history_default = client.get_history()  # No section specified, uses default
    history_explicit = client.get_history(section="default")  # Explicitly specify default
    
    assert len(history_default) == len(history_explicit)
    assert len(history_default) == 5  # system + user + assistant + user + assistant



if __name__ == "__main__":
    test_dummy_provider_echo()
    test_conversation_with_memory()
    test_multiple_sections()
    print("All tests passed!")