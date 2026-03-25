"""
Agent 6 — Batch Document Processor
=====================================
Processes a batch of documents where each document
may need a different model depending on its size and type.

What this shows:
  - Dynamic hints per document (size-aware routing)
  - How the same client handles different document types
  - Mixing use-cases in a loop
  - Reading the feedback log after processing

Run:
    cd llm_router_project
    python examples/agent_6_batch_processor.py
"""

import json
import os
from router.wrapped_client import WrappedLangchainClient

client = WrappedLangchainClient.auto("config.json")

print("=" * 60)
print("Agent 6 — Batch Document Processor")
print("=" * 60)

# ── Simulated document batch ───────────────────────────────────────────────────
# Each doc has: content, type, and approximate size
DOCUMENTS = [
    {
        "id": "doc_001",
        "type": "short_article",
        "content": "Python 3.12 introduces several performance improvements including faster startup times and improved error messages.",
        "tokens": 30
    },
    {
        "id": "doc_002",
        "type": "legal_contract",
        "content": "This agreement is entered into between Party A and Party B. " * 200,
        "tokens": 1200
    },
    {
        "id": "doc_003",
        "type": "large_codebase",
        "content": "# Large Python module\ndef func():\n    pass\n" * 1000,
        "tokens": 45000
    },
    {
        "id": "doc_004",
        "type": "research_paper",
        "content": "Abstract: This paper presents a novel approach to transformer optimization. " * 300,
        "tokens": 8000
    },
    {
        "id": "doc_005",
        "type": "invoice",
        "content": "Invoice #12345. Date: 2024-01-15. Items: Widget A x2 @ $50 = $100. Widget B x1 @ $75 = $75. Total: $175. Tax: $17.50. Grand Total: $192.50.",
        "tokens": 60
    },
]


def process_document(doc: dict) -> dict:
    """
    Process a single document. Picks use-case and urgency
    based on document type and size dynamically.
    """
    doc_type = doc["type"]
    tokens   = doc["tokens"]

    # Map document type to use-case
    use_case_map = {
        "short_article":  "summarization",
        "legal_contract": "data_extraction",
        "large_codebase": "long_context",
        "research_paper": "rag_answer",
        "invoice":        "data_extraction",
    }
    use_case = use_case_map.get(doc_type, "summarization")

    # Urgency based on document size — large docs can wait
    urgency = "high" if tokens < 500 else "low"

    # Build task-specific prompt
    if use_case == "data_extraction":
        prompt = f"Extract all key information as JSON from:\n{doc['content'][:500]}"
    elif use_case == "summarization":
        prompt = f"Summarize in 2 sentences:\n{doc['content'][:500]}"
    elif use_case == "long_context":
        prompt = f"Identify the main patterns in this code:\n{doc['content'][:500]}"
    else:
        prompt = f"Analyze this document:\n{doc['content'][:500]}"

    response = client.invoke(
        message=prompt,
        hints={
            "use_case": use_case,
            "input_token_estimate": tokens,
            "urgency": urgency
        }
    )

    return {
        "doc_id":    doc["id"],
        "doc_type":  doc_type,
        "tokens":    tokens,
        "use_case":  use_case,
        "urgency":   urgency,
        "model":     client.last_recommendation.winner_model_id,
        "score":     round(client.last_recommendation.winner_score, 3),
        "fallback":  client.last_recommendation.fallback_used,
        "result":    response.content[:150] + "..."
    }


# ── Process all documents ──────────────────────────────────────────────────────
results = []
for doc in DOCUMENTS:
    print(f"\nProcessing {doc['id']} ({doc['type']}, ~{doc['tokens']} tokens)...")
    result = process_document(doc)
    results.append(result)
    print(f"  use_case: {result['use_case']}")
    print(f"  model:    {result['model']} (score={result['score']})")
    print(f"  urgency:  {result['urgency']}")
    print(f"  result:   {result['result'][:80]}...")


# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Doc ID':<10} {'Type':<18} {'Tokens':>7} {'Model':<22} {'Score'}")
print("-" * 70)
for r in results:
    print(f"{r['doc_id']:<10} {r['doc_type']:<18} {r['tokens']:>7} "
          f"{r['model']:<22} {r['score']}")


# ── Read feedback log ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Feedback log (last 5 entries):")
log_path = "logs/feedback.jsonl"
if os.path.exists(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    for line in lines[-5:]:
        entry = json.loads(line)
        print(f"  {entry['timestamp']} | {entry['use_case']:<20} → {entry['winner']}")
else:
    print("  No feedback log yet (feedback_logging_enabled may be false)")