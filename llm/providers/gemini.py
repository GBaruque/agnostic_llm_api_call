from typing import List, Dict
from .base import BaseProvider


class GeminiProvider(BaseProvider):
    def send_messages(self, messages: List[Dict[str, str]]) -> str:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        payload = {"model": self.model or "default", "input": prompt}
        url = self.api_url or "https://gemini.api/proxy"
        r = self.session.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            if "output" in data:
                return data["output"]
            if "text" in data:
                return data["text"]
            if "candidates" in data and isinstance(data["candidates"], list):
                return data["candidates"][0].get("content", "")
        return str(data)
