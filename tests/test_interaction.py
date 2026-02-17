# from llm import Client, ConversationMemory
# from llm.providers import DummyProvider

# def test_dummy_provider_echo():
#     mem = ConversationMemory()
#     mem.add_message("chat", "system", "You are a friendly assistant.")

#     client = Client(DummyProvider, api_key="test")
#     resp = client.send(mem.get_section("chat"))

#     assert "From Dummy" in resp

# if __name__ == "__main__":
#     test_dummy_provider_echo()