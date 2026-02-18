import os
from typing import List, Dict, Optional
from .base import BaseProvider
from openai import OpenAI


class OpenRouterProvider(BaseProvider):
    """Provider adapter for OpenRouter API using the OpenAI-compatible client."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = "https://openrouter.ai/api/v1",
                 model: Optional[str] = "gpt-4o-mini",
                 timeout: int = 30):
        """Initialize OpenRouterProvider with sensible defaults.

        Defaults:
        - api_url: https://openrouter.ai/api/v1
        - model: 'gpt-4o-mini'
        """
        super().__init__(api_key=api_key, api_url=api_url, model=model, timeout=timeout)
        self.api_url = api_url
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required for OpenRouterProvider. " \
                             "Check OPENROUTER_API_KEY env variable or pass api_key argument.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )

    def send_messages(self, messages: List[Dict[str, str]]) -> str:
        """Send messages using OpenRouter API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )

        return response.choices[0].message.content
