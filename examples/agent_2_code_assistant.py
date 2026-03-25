"""
Agent 2 — Code Assistant Agent
================================
An agent that helps with coding tasks.
Demonstrates how different code tasks route to different models.

What this shows:
  - code_generation use-case routes to codellama
  - reasoning use-case routes to gpt-oss / llama3 for architecture decisions
  - urgency=low for review tasks (quality matters more than speed)
  - urgency=high for quick syntax questions

Run:
    cd llm_router_project
    python examples/agent_2_code_assistant.py
"""

from router.wrapped_client import WrappedLangchainClient

client = WrappedLangchainClient.auto("config.json")

print("=" * 60)
print("Agent 2 — Code Assistant")
print("=" * 60)


# ── Task 1: Write code ────────────────────────────────────────────────────────
# code_generation → codellama (specialist, highest score for this use-case)
print("\n[1] Write code — routes to codellama")
response = client.invoke(
    message="""
    Write a Python function that:
    - Takes a list of dictionaries
    - Groups them by a given key
    - Returns a dict of lists
    Include type hints and docstring.
    """,
    hints={
        "use_case": "code_generation",
        "urgency": "normal"
    }
)
print(response.content)
print(f"\n→ Model: {client.last_recommendation.winner_model_id} "
      f"(score: {client.last_recommendation.winner_score:.3f})")


# ── Task 2: Debug code ────────────────────────────────────────────────────────
print("\n[2] Debug code — routes to codellama")
buggy_code = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total =+ n        # bug here
    return total / len(numbers)
"""
response = client.invoke(
    message=f"Find and fix the bug in this code:\n{buggy_code}",
    hints={
        "use_case": "code_generation",
        "urgency": "high"     # developer is blocked, needs fix fast
    }
)
print(response.content)
print(f"\n→ Model: {client.last_recommendation.winner_model_id}")


# ── Task 3: Architecture decision ─────────────────────────────────────────────
# This is NOT just code — it's a reasoning task about trade-offs.
# Routes to gpt-oss or llama3 (reasoning use-case, quality dominant).
print("\n[3] Architecture decision — routes to reasoning model")
response = client.invoke(
    message="""
    I'm building a real-time chat application.
    Should I use WebSockets or Server-Sent Events?
    Consider: scalability, browser support, bidirectional needs, complexity.
    """,
    hints={
        "use_case": "reasoning",
        "urgency": "low"      # this is a design decision, take your time
    }
)
print(response.content)
print(f"\n→ Model: {client.last_recommendation.winner_model_id} "
      f"(score: {client.last_recommendation.winner_score:.3f})")


# ── Task 4: Code review of a large file ──────────────────────────────────────
# Large codebase → needs high context window
print("\n[4] Large codebase review — context hint routes to gemini-flash")
large_code = "# some large Python module\n" + "def func():\n    pass\n" * 500
response = client.invoke(
    message=f"Review this code for issues:\n{large_code[:300]}...",
    hints={
        "use_case": "code_generation",
        "input_token_estimate": len(large_code) // 4,
        "urgency": "low"
    }
)
print(response.content[:300], "...")
print(f"\n→ Model: {client.last_recommendation.winner_model_id}")
print("\nFull explanation:")
print(client.explain_last())