# Agnostic LLM API Client

A clean, modular Python client for interacting with multiple LLM providers (Groq, Gemini, OpenRouter) with built-in conversation memory management and sectioned chat history.

## Project Structure

```
llm/
├── __init__.py           # Public API exports
├── client.py             # LLMClient - main facade
├── memory.py             # ConversationMemory, Message dataclasses
└── providers/
    ├── base.py           # BaseProvider abstract class
    ├── groq.py           # GroqProvider implementation
    ├── gemini.py         # GeminiProvider implementation
    ├── openrouter.py     # OpenRouterProvider implementation
    └── dummy.py          # DummyProvider for testing

tests/
├── test_groq.py          # Groq provider tests
├── test_gemini.py        # Gemini provider tests
├── test_openrouter.py    # OpenRouter provider tests
└── conftest.py           # Pytest configuration with .env loading
```

## Features

- **Provider-Agnostic**: Switch providers by passing different provider classes
- **Built-in Memory**: `ConversationMemory` with sectioned conversations (independent chat threads)
- **Message Dataclass**: Structured `Message` objects with `role` and `content`
- **Clean Architecture**: 
  - `BaseProvider` abstract class defines interface
  - Each provider (`GroqProvider`, `GeminiProvider`, `OpenRouterProvider`) inherits and implements
  - `LLMClient` facade manages provider + memory
- **Environment Variables**: Providers read API keys from environment (with fallbacks for testing)
- **Full Test Coverage**: Unit tests for each provider with `monkeypatch` env var handling

## Supported Providers

| Provider | Library | Default Model | Env Var |
|----------|---------|---------------|---------|
| **Groq** | `openai` (OpenAI-compatible) | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **Gemini** | `google-generativeai` | `gemini-1.5-flash` | `GEMINI_API_KEY` |
| **OpenRouter** | `openai` (OpenAI-compatible) | `gpt-4o-mini` | `OPENROUTER_API_KEY` |
| **Dummy** | None (local) | N/A | None |

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your-groq-api-key
GEMINI_API_KEY=your-gemini-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

Or set them directly:

```bash
export GROQ_API_KEY="your-groq-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

## Usage Examples

### Basic Chat with Groq

```python
from llm import LLMClient
from llm.providers import GroqProvider

client = LLMClient(
    GroqProvider(),
    system_prompt="You are a helpful assistant."
)

response = client.ask("What is the capital of France?")
print(response)
```

### Multi-Turn Conversation with Memory

```python
from llm import LLMClient
from llm.providers import GeminiProvider

client = LLMClient(GeminiProvider(), system_prompt="You are a concise assistant.")

# First turn
resp1 = client.ask("What is 2+2?")
print(resp1)

# Second turn - memory is maintained
resp2 = client.ask("What was my previous question?")
print(resp2)

# View conversation history
for msg in client.get_history():
    print(f"{msg.role.upper()}: {msg.content}")
```

### Multiple Independent Conversations (Sections)

```python
from llm import LLMClient
from llm.providers import OpenRouterProvider

client = LLMClient(OpenRouterProvider())

# Section 1: Python discussion
client.ask("Tell me about Python", section="python-chat")
client.ask("Why is it popular?", section="python-chat")

# Section 2: JavaScript discussion (independent)
client.ask("Tell me about JavaScript", section="js-chat")
client.ask("Where is it used?", section="js-chat")

# View history per section
python_history = client.get_history(section="python-chat")
js_history = client.get_history(section="js-chat")

# Clear a specific section
client.clear_section(section="python-chat")

# Or reset everything
client.reset()
```

### Testing with DummyProvider (Offline)

```python
from llm import LLMClient
from llm.providers import DummyProvider

# No API key needed
client = LLMClient(DummyProvider(), api_key="test")
response = client.ask("Hello!")  # Returns: "From Dummy reply (echo): Hello!"
```

## API Reference

### LLMClient

```python
class LLMClient:
    def __init__(
        self,
        provider: Type[BaseProvider] | BaseProvider,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: int = 30
    )
    
    def ask(self, text: str, section: str = "default") -> str
        """Send user message and get response."""
    
    def get_history(self, section: str = "default") -> List[Message]
        """Get conversation history for a section."""
    
    def clear_section(self, section: str = "default")
        """Clear all messages in a section."""
    
    def reset()
        """Clear all conversations."""
```

### ConversationMemory

```python
@dataclass
class Message:
    role: str        # "user", "assistant", "system"
    content: str     # Message text

@dataclass
class ConversationMemory:
    def add_user(text: str, section: str = "default")
    def add_assistant(text: str, section: str = "default")
    def add_system(text: str, section: str = "default")
    def get_section(section: str = "default") -> List[Message]
    def clear_section(section: str = "default")
    def clear()
    def as_openai(section: str = "default") -> List[Dict]
        """Convert section to OpenAI format."""
```

## Running Tests

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Tests for a Specific Provider

```bash
python -m pytest tests/test_groq.py -v
python -m pytest tests/test_gemini.py -v
python -m pytest tests/test_openrouter.py -v
```

### Run with Real API Keys

```bash
GROQ_API_KEY=your-key python -m pytest tests/test_groq.py -v
```

### Run with Verbose Output

```bash
python -m pytest tests/ -vv -s
```

## Architecture Notes

- **BaseProvider**: Abstract base class defining the `send_messages(messages: List[Dict]) -> str` interface
- **Provider Implementations**: Each provider (`GroqProvider`, `GeminiProvider`, `OpenRouterProvider`) inherits `BaseProvider` and:
  - Accepts OpenAI-compatible message format (list of dicts with `role` and `content`)
  - Converts to provider-specific format if needed (e.g., Gemini's custom history format)
  - Returns plain text response
- **LLMClient**: Facade that owns a provider instance and a `ConversationMemory`
  - `ask()` automatically adds user message, sends to provider, stores response
  - Supports sectioned conversations for independent chat threads
- **ConversationMemory**: Manages message history as `Message` dataclasses
  - Sections allow parallel conversations without interference
  - `as_openai()` converts section to OpenAI-compatible format for providers

## Dependencies

- `requests>=2.28` - HTTP requests
- `openai>=2.21.0` - OpenAI-compatible clients (Groq, OpenRouter)
- `google-generativeai` - Google Gemini API
- `python-dotenv>=0.19` - Load `.env` files
- `pytest>=7.0` - Testing framework
