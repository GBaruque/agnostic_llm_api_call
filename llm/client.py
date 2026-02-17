from typing import Optional, Type, List
from llm.providers.base import BaseProvider
from .memory import ConversationMemory, Message


class LLMClient:
    """Client that manages conversation with an LLM provider.
    
    Owns both the provider and conversation memory, providing a simple
    interface to chat: ask(text) -> response. Tracks full conversation history.
    
    Parameters:
    - provider: A BaseProvider subclass or instance
    - api_key: API key (required if provider is a class)
    - api_url: Optional provider endpoint override
    - model: Optional model name override
    - system_prompt: Optional initial system message
    - timeout: Request timeout in seconds
    """

    def __init__(
        self, 
        provider: Type[BaseProvider] | BaseProvider, 
        api_key: Optional[str] = None, 
        api_url: Optional[str] = None, 
        model: Optional[str] = None, 
        system_prompt: Optional[str] = None,
        timeout: int = 30
    ):
        # Instantiate provider if a class was passed
        if isinstance(provider, type):
            if api_key is None:
                raise ValueError("api_key is required when passing a provider class")
            self.provider = provider(api_key=api_key, api_url=api_url, model=model, timeout=timeout)
        else:
            self.provider = provider

        if not isinstance(self.provider, BaseProvider):
            raise TypeError("provider must be a BaseProvider subclass or instance")

        self.memory = ConversationMemory()
        
        # Add system prompt if provided
        if system_prompt:
            self.memory.add_system(system_prompt)

    def ask(self, text: str, section: str = "default") -> str:
        """Send a user message to a section and get an assistant response.
        
        Automatically manages conversation memory: adds user message to section,
        sends all messages from section to provider, receives response, and stores it.
        
        Parameters:
            text: User's message
            section: Conversation section (default: "default")
        
        Returns:
            The assistant's response text.
        """
        self.memory.add_user(text, section=section)
        response = self.provider.send_messages(self.memory.as_openai(section=section))
        self.memory.add_assistant(response, section=section)
        return response

    def get_history(self, section: str = "default") -> List[Message]:
        """Get conversation history for a section.
        
        Returns:
            List of Message objects from the section.
        """
        return self.memory.get_section(section)

    def clear_section(self, section: str = "default"):
        """Clear conversation history for a section."""
        self.memory.clear_section(section)

    def reset(self):
        """Clear all conversation history."""
        self.memory.clear()
