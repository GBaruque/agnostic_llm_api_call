from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Message:
    """Represents a single message in conversation."""
    role: str
    content: str


@dataclass
class ConversationMemory:
    """Manages sectioned conversation history with structured messages.
    
    Supports multiple independent conversation sections (e.g., different topics).
    Provides methods to add user, assistant, and system messages,
    and convert to OpenAI-compatible format for API calls.
    
    Example:
        mem = ConversationMemory()
        mem.add_user("Hello", section="greetings")
        mem.add_assistant("Hi there!", section="greetings")
        messages = mem.as_openai("greetings")
    """
    sections: Dict[str, List[Message]] = field(default_factory=dict)

    def _ensure_section(self, section: str = "default"):
        """Ensure section exists."""
        if section not in self.sections:
            self.sections[section] = []

    def add_user(self, text: str, section: str = "default"):
        """Add a user message to a section."""
        self._ensure_section(section)
        self.sections[section].append(Message("user", text))

    def add_assistant(self, text: str, section: str = "default"):
        """Add an assistant message to a section."""
        self._ensure_section(section)
        self.sections[section].append(Message("assistant", text))

    def add_system(self, text: str, section: str = "default"):
        """Add a system message to a section."""
        self._ensure_section(section)
        self.sections[section].append(Message("system", text))

    def get_section(self, section: str = "default") -> List[Message]:
        """Get all messages from a section."""
        return self.sections.get(section, [])

    def clear_section(self, section: str = "default"):
        """Clear all messages from a section."""
        if section in self.sections:
            self.sections[section].clear()

    def clear(self):
        """Clear all messages from all sections."""
        self.sections.clear()

    def as_openai(self, section: str = "default") -> List[Dict]:
        """Convert messages from a section to OpenAI-compatible format."""
        messages = self.get_section(section)
        return [{"role": m.role, "content": m.content} for m in messages]
