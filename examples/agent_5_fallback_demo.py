"""
Agent 5 — Fallback Demonstration
==================================
Shows the fallback mechanism in action.
Manually marks models as "offline" in config to simulate Ollama downtime.

What this shows:
  - What happens when the primary model is offline
  - How the router automatically falls back to next available model
  - How to check which model was actually used
  - The full explanation including eliminated models

Run:
    cd llm_router_project
    python examples/agent_5_fallback_demo.py
"""

import json
import tempfile
import os
from router.wrapped_client import WrappedLangchainClient

print("=" * 60)
print("Agent 5 — Fallback Demonstration")
print("=" * 60)


# ── Helper: create a config with health_check disabled ───────────────────────
# We simulate offline models by temporarily overriding health checks
# and using a modified config. In real usage, Ollama going down is
# detected automatically — this just demonstrates the routing logic.

def load_config_with_health_off():
    """Load config with health checks disabled so we can test scoring only."""
    with open("config.json") as f:
        cfg = json.load(f)
    cfg["recommendation_settings"]["health_check_enabled"] = False
    return cfg


# ── Demo 1: Normal operation ───────────────────────────────────────────────────
print("\n[Demo 1] Normal operation — all models healthy")
client = WrappedLangchainClient.auto("config.json")
response = client.invoke(
    message="What is a neural network?",
    hints={"use_case": "chat", "urgency": "high"}
)
print(f"Response: {response.content[:150]}...")
print(f"Winner: {client.last_recommendation.winner_model_id}")
print(f"Score:  {client.last_recommendation.winner_score:.3f}")
print(f"Fallback used: {client.last_recommendation.fallback_used}")


# ── Demo 2: Show explanation with eliminated models ────────────────────────────
print("\n[Demo 2] Reasoning task — see which models are eliminated")
response = client.invoke(
    message="Explain the CAP theorem in distributed systems",
    hints={"use_case": "reasoning", "urgency": "normal"}
)
print(f"Response: {response.content[:200]}...")
print()
print("Full recommendation explanation:")
print(client.explain_last())


# ── Demo 3: Context filter eliminates most models ────────────────────────────
print("\n[Demo 3] Large input — context filter at work")
response = client.invoke(
    message="Summarize this document",
    hints={
        "use_case": "summarization",
        "input_token_estimate": 90000,    # 90k tokens → only gemini-flash survives
        "urgency": "low"
    }
)
print(f"Response: {response.content[:150]}...")
print()
print("Explanation:")
print(client.explain_last())
print()
print("Eliminated models:")
for model_id, reason in client.last_recommendation.eliminated.items():
    print(f"  ✗ {model_id}: {reason}")


# ── Demo 4: Urgency changes which model wins ──────────────────────────────────
print("\n[Demo 4] Same question, different urgency → different model")

for urgency in ["low", "normal", "high"]:
    response = client.invoke(
        message="Explain gradient descent",
        hints={"use_case": "reasoning", "urgency": urgency}
    )
    print(f"  urgency={urgency:6s} → model: {client.last_recommendation.winner_model_id:20s} "
          f"score: {client.last_recommendation.winner_score:.3f}")


# ── Demo 5: Available use-cases and models ────────────────────────────────────
print("\n[Demo 5] What's available in your config?")
print("Use-cases:", client.available_use_cases())
print("Models:", client.available_models())