import json

from ai_llm_router import LLMClient, LLMSettings

settings = LLMSettings(
    provider="gemini",
    model="gemini-2.5-flash",
    api_key="API_KEY",
    stream=False,
)
client = LLMClient(settings)

prompt = "What are three fun facts about octopuses?"

system_message = (
    "You are a helpful assistant. Answer clearly and briefly."
)
response = client.complete(prompt, system=system_message)
print(f'Response: {response.text}')
print(f'Metadata: {json.dumps(response.metadata, indent=2)}')