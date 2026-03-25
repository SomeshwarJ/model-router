"""
Agent 3 — RAG (Retrieval Augmented Generation) Agent
======================================================
Simulates a document Q&A pipeline.
Shows how context size + quality requirements drive model selection.

What this shows:
  - rag_answer use-case routes to gemini-flash (128k ctx + rag tag)
  - data_extraction use-case routes to llama3
  - How to pass retrieved chunks as context
  - Different models for different stages of the same pipeline

Pipeline:
  User question
      → [Retrieval] fetch relevant chunks  (simulated here)
      → [Answer]    generate answer grounded in chunks  → gemini-flash
      → [Extract]   pull structured data from answer    → llama3

Run:
    cd llm_router_project
    python examples/agent_3_rag_agent.py
"""

from router.wrapped_client import WrappedLangchainClient

client = WrappedLangchainClient.auto("config.json")

print("=" * 60)
print("Agent 3 — RAG Agent")
print("=" * 60)

# Simulated retrieved document chunks (in real RAG, these come from a vector DB)
RETRIEVED_CHUNKS = """
[Chunk 1 - Source: company_policy.pdf, page 3]
Employees are entitled to 20 days of annual leave per calendar year.
Leave must be requested at least 2 weeks in advance via the HR portal.
Unused leave can be carried forward up to a maximum of 5 days.

[Chunk 2 - Source: company_policy.pdf, page 4]
Sick leave is separate from annual leave. Employees receive 10 days
of sick leave per year. A medical certificate is required for absences
exceeding 3 consecutive days.

[Chunk 3 - Source: company_policy.pdf, page 5]
Maternity leave: 16 weeks fully paid. Paternity leave: 2 weeks fully paid.
Shared parental leave is available subject to eligibility criteria.
"""

USER_QUESTION = "How many days of annual leave do I get and can I carry them forward?"


# ── Stage 1: RAG Answer ───────────────────────────────────────────────────────
# Combine retrieved chunks + question into a single prompt
# rag_answer use-case → gemini-flash wins (128k ctx, rag tag)
print(f"\n[1] User question: {USER_QUESTION}")
print("\n[RAG Stage] Answering from retrieved context...")

rag_prompt = f"""
Answer the following question using ONLY the provided context.
If the answer is not in the context, say "I don't know".

Context:
{RETRIEVED_CHUNKS}

Question: {USER_QUESTION}
"""

response = client.invoke(
    message=rag_prompt,
    hints={
        "use_case": "rag_answer",
        "input_token_estimate": len(rag_prompt) // 4,
        "urgency": "normal"
    }
)
rag_answer = response.content
print("Answer:", rag_answer)
print(f"\n→ Model: {client.last_recommendation.winner_model_id} "
      f"(score: {client.last_recommendation.winner_score:.3f})")


# ── Stage 2: Extract structured data from the answer ─────────────────────────
# Now extract specific fields from the RAG answer as structured JSON
# data_extraction → llama3 (instruction-following, quality=0.85)
print("\n[Extract Stage] Pulling structured data from answer...")

extract_prompt = f"""
Extract the following fields from this text as JSON:
- annual_leave_days (integer)
- carry_forward_days (integer)
- advance_notice_required (string)

Text: {rag_answer}

Reply with only valid JSON, no explanation.
"""

response = client.invoke(
    message=extract_prompt,
    hints={
        "use_case": "data_extraction",
        "urgency": "normal"
    }
)
print("Extracted JSON:", response.content)
print(f"\n→ Model: {client.last_recommendation.winner_model_id} "
      f"(score: {client.last_recommendation.winner_score:.3f})")


# ── Stage 3: Summarize the policy section ────────────────────────────────────
# Quick summary for a Slack notification — fast + cheap
print("\n[Summarize Stage] Generating a short Slack-friendly summary...")

response = client.invoke(
    message=f"Summarize this in one sentence for a Slack message: {rag_answer}",
    hints={
        "use_case": "summarization",
        "urgency": "high"     # Slack notification should be instant
    }
)
print("Slack summary:", response.content)
print(f"\n→ Model: {client.last_recommendation.winner_model_id} "
      f"(fastest model wins for summarization+high urgency)")


# ── Full pipeline summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Pipeline model usage:")
print("  Stage 1 (rag_answer)     → gemini-flash  [128k ctx + rag tag]")
print("  Stage 2 (data_extraction)→ llama3         [best instruction-following]")
print("  Stage 3 (summarization)  → llama3.2-3b   [fastest for quick summary]")
print("  Each stage uses the RIGHT model for its task automatically.")