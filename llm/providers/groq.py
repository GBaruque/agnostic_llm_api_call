from typing import List, Dict
from .base import BaseProvider


class GroqProvider(BaseProvider):
    def send_messages(self, messages: List[Dict[str, str]]) -> str:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        payload = {"model": self.model or "default", "input": prompt}
        url = self.api_url or "https://api.groq.ai/v1/complete"
        r = self.session.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # common shapes
        if isinstance(data, dict):
            if "output" in data:
                return data["output"]
            if "text" in data:
                return data["text"]
            if "results" in data and isinstance(data["results"], list):
                return data["results"][0].get("output", "")
        return str(data)
