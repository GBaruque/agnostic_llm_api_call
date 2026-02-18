import os
from typing import List, Dict, Optional
from .base import BaseProvider
from openai import OpenAI


class GroqProvider(BaseProvider):
    def __init__(self,
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = "https://api.groq.com/openai/v1",
                 model: Optional[str] = "llama-3.3-70b-versatile",
                 timeout: int = 30):
        """Initialize GroqProvider with sensible defaults.

        Defaults:
        - api_url: https://api.groq.ai/v1/complete
        - model: 'default'
        """
        super().__init__(api_key=api_key, api_url=api_url, model=model, timeout=timeout)
        self.api_url = api_url
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required for GroqProvider." \
            "Check GROQ_API_KEY env variable or pass api_key argument.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )

    def send_messages(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )

        return response.choices[0].message.content
