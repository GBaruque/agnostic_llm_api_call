from typing import List, Dict
from .base import BaseProvider


class DummyProvider(BaseProvider):
    """A local provider useful for testing without network.

    - Returns a deterministic response based on the last user message.
    - No network calls are made.
    """

    def send_messages(self, messages: List[Dict[str, str]]) -> str:
        # Find the last user message
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content")
                break

        if last_user:
            return f"From Dummy reply (echo): {last_user}"
        # fallback reply
        return "From Dummy reply: no user message found."
