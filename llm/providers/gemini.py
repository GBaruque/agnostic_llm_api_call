import os
from typing import List, Dict, Optional
from google import genai
from llm.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Provider adapter for Google Generative AI (Gemini) using the google-generativeai library.

    Converts OpenAI-style messages to Gemini chat history format and handles
    system prompts by prepending them as the first user message if needed.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = None,
                 model: Optional[str] = "gemini-2.5-flash-lite",
                 timeout: int = 30):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key is required for GeminiProvider. " \
            "Check GEMINI_API_KEY env variable or pass api_key argument.")

        self.api_key = api_key
        self.model_name = model
        self.timeout = timeout
        self.gclient = genai.Client(api_key=self.api_key)
        self.gconfigs = genai.types.GenerateContentConfig(system_instruction=None)

    def _convert_messages_to_gemini_history(self,
                                            messages: List[Dict[str, str]]):
        system_instruction = None
        contents = []

        for m in messages:
            role = m["role"]
            content = m["content"]

            if role == "system":
                system_instruction = content
                continue

            # Gemini só aceita user ou model
            if role == "assistant":
                gemini_role = "model"
            elif role == "user":
                gemini_role = "user"
            else:
                raise ValueError(f"Unsupported role '{role}' in messages. Only 'user', 'assistant', and 'system' are supported.")

            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })
        
        return system_instruction, contents

    def send_messages(self, messages: List[Dict[str, str]]) -> str:
        # Separate system messages from chat messages
        system_instruction, contents = self._convert_messages_to_gemini_history(messages)
        
        if system_instruction:
            self.gconfigs = genai.types.GenerateContentConfig(system_instruction=system_instruction)
        
        response = self.gclient.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self.gconfigs
        )
        
        return response.text

if __name__ == "__main__":
    # Simple test to verify provider works
    provider = GeminiProvider(api_key=os.getenv("GEMINI_API_KEY"))
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    print(provider.send_messages(test_messages))