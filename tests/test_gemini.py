import os
from llm import LLMClient
from llm.providers.gemini import GeminiProvider


def test_gemini_chat(monkeypatch):
    # Set env var with fallback: use real key if set, else use test key
    monkeypatch.setenv("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "fallback-key"))
    
    bot = LLMClient(
        GeminiProvider(),
        system_prompt="You are a concise assistant"
    )

    q = "What is the capital of France?"
    response = bot.ask(q)
    assert response is not None, "Response should not be None"
    assert isinstance(response, str), "Response should be a string"
    assert "Paris" in response, f"Expected 'Paris' in response, got: {response}"


def test_gemini_maintained_chat(monkeypatch):
    # Set env var with fallback: use real key if set, else use test key
    monkeypatch.setenv("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "fallback-key"))
    
    bot = LLMClient(
        GeminiProvider(),
        system_prompt="You are a concise assistant"
    )
    
    countries = ["France", "Germany", "Italy"]
    for country in countries:
        q = f"What is the capital of {country}?"
        response = bot.ask(q)
        assert response is not None, f"Response should not be None for {country}"
        assert isinstance(response, str), f"Response should be a string for {country}"
        assert country in response, f"Expected '{country}' in response, got: {response}"

    response = bot.ask("What were the countries I asked about?")
    assert response is not None, "Response should not be None"
    assert all(country in response for country in countries), \
        f"Expected all countries in response, got: {response}"
