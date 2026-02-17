from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import requests


class BaseProvider(ABC):
    """Abstract base provider. Implement `send_messages` in subclasses.

    Subclasses should accept the same constructor signature and implement
    provider-specific request/response handling.
    """

    def __init__(self, api_key: str, api_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

    @abstractmethod
    def send_messages(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat-style `messages` list and return assistant text."""
