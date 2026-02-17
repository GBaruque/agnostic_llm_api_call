from typing import List, Dict
from .base import BaseProvider


class OpenRouterProvider(BaseProvider):
    def send_messages(self, messages: List[Dict[str, str]]) -> str:
        payload = {"model": self.model or "gpt-4o-mini", "messages": messages}
        url = self.api_url or "https://api.openrouter.ai/v1/chat/completions"
        r = self.session.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            try:
                return data["choices"][0]["text"]
            except Exception:
                raise RuntimeError("unexpected OpenRouter response: %s" % data)
