import os
from llm.client import LLMClient
from llm.providers.groq import GroqProvider


def test_groq_chat(monkeypatch):
    # Set env var with fallback: use real key if set, else use test key
    monkeypatch.setenv("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "fallback-key"))
    
    bot = LLMClient(
        GroqProvider(),
        system_prompt="You are a concise assistant"
        )

    q = "What is the capital of France?"
    response = bot.ask(q)
    assert "Paris" in response, f"Expected 'Paris' in response, got: {response}"

def test_groq_maintained_chat(monkeypatch):
    # Set env var with fallback: use real key if set, else use test key
    monkeypatch.setenv("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "fallback-key"))
    
    bot = LLMClient(
        GroqProvider(),
        system_prompt="You are a concise assistant"
        )
    
    countries = ["France", "Germany", "Italy"]
    capitals = ["Paris", "Berlin", "Rome"]
    for country, capital in zip(countries, capitals):
        q = f"What is the capital of {country}?"
        response = bot.ask(q)
        assert response is not None, f"Response should not be None for {country}"
        assert isinstance(response, str), f"Response should be a string for {country}"
        assert capital in response, f"Expected '{capital}' in response, got: {response}"
    response = bot.ask("What were the countries I asked about?")
    assert response is not None, "Response should not be None"
    assert all(country in response for country in countries), \
        f"Expected all countries in response, got: {response}"