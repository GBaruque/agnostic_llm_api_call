"""
Example usage of LLMClient with sectioned conversations.

Set environment variables:
- API_KEY: your API key (defaults to "test-key")
- API_URL: optional provider endpoint override
- MODEL: optional model name override

Run: python example.py
"""
import os
from llm import LLMClient
from llm.providers import DummyProvider


def main():
    api_key = os.environ.get("API_KEY", "test-key")
    api_url = os.environ.get("API_URL")
    model = os.environ.get("MODEL")

    # Create client with DummyProvider for offline testing
    # Swap DummyProvider for OpenRouterProvider, GroqProvider, GeminiProvider, etc. when ready
    client = LLMClient(
        provider=DummyProvider,
        api_key=api_key,
        api_url=api_url,
        model=model,
        system_prompt="You are a helpful assistant."
    )

    # Example 1: Single-section conversation (default)
    print("=== Example 1: Simple conversation ===\")
    response1 = client.ask("Hello! Please introduce yourself briefly.")
    print(f"Assistant: {response1}\n")

    print("Follow-up question (memory maintained):")
    response2 = client.ask("What did you just say?")
    print(f"Assistant: {response2}\n")

    # Example 2: Multi-section conversations
    print("\n=== Example 2: Multiple independent sections ===\")
    
    # Section 1: Python discussion
    client.ask("Tell me about Python", section="python-talk")
    resp_py = client.ask("Why is it popular?", section="python-talk")
    print(f"Python section response: {resp_py}\n")
    
    # Section 2: JavaScript discussion (independent)
    client.ask("Tell me about JavaScript", section="js-talk")
    resp_js = client.ask("Where is it used?", section="js-talk")
    print(f"JavaScript section response: {resp_js}\n")

    # Example 3: View conversation history
    print("=== Example 3: Conversation history ===\")
    print("Default section history:")
    for msg in client.get_history():
        print(f"  {msg.role.upper()}: {msg.content}")
    
    print("\nPython section history:")
    for msg in client.get_history(section="python-talk"):
        print(f"  {msg.role.upper()}: {msg.content}")
    
    print("\nJavaScript section history:")
    for msg in client.get_history(section="js-talk"):
        print(f"  {msg.role.upper()}: {msg.content}")

    # Example 4: Clear specific section
    print("\n=== Example 4: Clear section ===\")
    client.clear_section(section="python-talk")
    print(f"Python section after clear: {len(client.get_history(section='python-talk'))} messages")
    print(f"JavaScript section still intact: {len(client.get_history(section='js-talk'))} messages")


if __name__ == "__main__":
    main()
