from router.wrapped_client import WrappedLangchainClient
from langchain_core.messages import HumanMessage, AIMessage


# ── Create the client once ──────────────────────────────────────────────────
# .from_model() get the model and prepares the router.
# One client instance is reused for all questions.
client = WrappedLangchainClient.from_model("llama3")

print("=" * 60)
print("Agent 7 — Normal Langgraph")
print("=" * 60)

# ── Example 1: Explicit Model Name ─────────────────────────────────────────────
# Basic
response = client.invoke("Write a Python retry decorator")
print(response.content)

# ── Example 1: Explicit Model Name with params─────────────────────────────────────────────
# With custom params
client = WrappedLangchainClient.from_model(
    "codellama:latest",
    temperature=0.2,
    max_tokens=1000
)

response = client.invoke("Write a Python retry decorator")
print(response.content)