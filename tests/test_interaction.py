from llm.client import LLMClient
from llm.memory import ConversationMemory
from llm.providers import DummyProvider

def test_dummy_provider_echo():
    counter = 0
    while True:
            
        mem = ConversationMemory()
        mem.add_system(section="chat",
                    text="You are a friendly assistant.")
        
        mem.add_user(section="chat",
                    text="My input")

        client = LLMClient(DummyProvider, api_key="test")
        resp = client.ask(mem.get_section("chat"))

        assert "From Dummy" in resp
        counter += 1

        if counter >= 5:
            break

if __name__ == "__main__":
    test_dummy_provider_echo()