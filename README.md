# Agnostic LLM Client (example)

This small project provides a minimal, provider-agnostic Python client (`llm_client.py`) with sectioned conversation memory.

Quick notes
- Providers supported (identifier): `openrouter`, `groq`, `gemini`.
- The client uses `requests` and sends JSON HTTP requests. Some providers use chat-style `messages`, others expect a single `input` field.
- You should set correct `API_URL` values or use your provider's environment variables if the defaults are not correct for your account.

Setup

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the example (set `API_KEY` and optionally `PROVIDER`, `API_URL`, `MODEL`):

```bash
export API_KEY="your_api_key_here"
export PROVIDER=openrouter  # or groq or gemini
python example.py
```

Usage

- Use `LLMClient(provider, api_key, api_url=None, model=None)` in your code.
- Create a `ConversationMemory` instance and add messages via `add_message(section, role, content)`.
- Send `mem.get_section("section_name")` to `LLMClient.send(...)`.

Notes and limitations
- This is a small, flexible starter. Each provider's true endpoint and request/response schema may differ. If you use a provider's official API, set `api_url` to the exact endpoint and `model` to the exact model name.
- For Gemini (Google Vertex/other wrappers), you may need OAuth or custom headers — the client supports overriding `api_url` and will send the API key in the `Authorization` header by default.

If you'd like, I can:
- Add persistence for memory (JSON file or SQLite)
- Add token-counting / trimming per section
- Add provider-specific adapters for exact endpoint schemas
# agnostic_llm_api_call
Repository for training and testing api llm calls while using goos and maintainable code structure
