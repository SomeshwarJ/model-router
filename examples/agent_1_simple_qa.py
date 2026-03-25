"""
Agent 1 — Simple Q&A Agent
===========================
The most basic usage of WrappedLangchainClient.
A single agent that answers user questions.

What this shows:
  - How to create a client with .auto()
  - How to call .invoke() with hints
  - How to read the explanation after each call
  - Auto-detection when no use_case is given

Run:
    cd llm_router_project
    python examples/agent_1_simple_qa.py
"""

from router.wrapped_client import WrappedLangchainClient
from langchain_core.messages import HumanMessage, SystemMessage

# ── Create the client once ──────────────────────────────────────────────────
# .auto() reads config.json and prepares the router.
# One client instance is reused for all questions.
client = WrappedLangchainClient.auto("config.json")

print("=" * 60)
print("Agent 1 — Simple Q&A")
print("=" * 60)


# ── Example 1: Explicit use_case ─────────────────────────────────────────────
# You know this is a chat task → tell the router directly.
# urgency=high because the user is waiting for a live response.
print("\n[1] General chat question")
response = client.invoke(
    message="What is the difference between RAM and ROM?",
    hints={
        "use_case": "chat",
        "urgency": "high"        # user is waiting → prefer fast model
    }
)
print("Answer:", response.content)
print("Model used:", client.last_recommendation.winner_model_id)
print("Score:", round(client.last_recommendation.winner_score, 3))


# ── Example 2: Auto-detection ────────────────────────────────────────────────
# No use_case hint → phi3-mini reads the message and classifies it.
# Then the router picks the best model for that detected use-case.
print("\n[2] Auto-detected use-case (no hints)")
response = client.invoke(
    message="Summarize the key differences between Python and JavaScript"
)
print("Answer:", response.content)
print("Detected use-case:", client.last_recommendation.use_case_name)
print("Auto-detected?", client.last_recommendation.use_case_auto_detected)
print("Model used:", client.last_recommendation.winner_model_id)


# ── Example 3: Large document context ────────────────────────────────────────
# Simulate passing a large document. input_token_estimate tells the router
# how big the input is → filters out models that can't hold it.
print("\n[3] Large document — context hint forces gemini-flash")
long_doc = "This is a very long document... " * 3000   # ~24k tokens
response = client.invoke(
    message=f"Summarize this: {long_doc[:500]}...",    # truncated for demo
    hints={
        "use_case": "summarization",
        "input_token_estimate": len(long_doc) // 4,    # rough token estimate
        "urgency": "low"
    }
)
print("Answer:", response.content[:200], "...")
print("Model used:", client.last_recommendation.winner_model_id)
print()
print("Why this model?")
print(client.explain_last())


# ── Example 4: With system message ───────────────────────────────────────────
# Passing a list of messages (system + human) directly.
print("\n[4] With system prompt")
messages = [
    SystemMessage(content="You are a concise assistant. Answer in one sentence."),
    HumanMessage(content="What is LangChain?")
]
response = client.invoke(
    message=messages,
    hints={"use_case": "chat", "urgency": "high"}
)
print("Answer:", response.content)
print("Model used:", client.last_recommendation.winner_model_id)
