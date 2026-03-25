"""
Agent 4 — LangGraph Multi-Node Pipeline
=========================================
A full LangGraph agent with 4 nodes, each using a different model
automatically selected by the router based on the node's use-case.

Pipeline:
  User input
      → [intent_node]     classify intent          → gemma2-2b  (fastest)
      → [router_node]     decide next step         → phi3-mini  (lightweight)
      → [answer_node]     generate main response   → llama3     (reasoning)
      → [format_node]     clean up output          → llama3.2-3b (fast)

What this shows:
  - @recommend_model decorator pattern
  - RouterNode base class pattern
  - get_client() manual pattern
  - All 3 patterns working inside the same LangGraph graph
  - Different models per node, all automatic

Run:
    cd llm_router_project
    python examples/agent_4_langgraph_pipeline.py
"""

from typing import TypedDict, Optional
from router.wrapped_client import WrappedLangchainClient
from router.langgraph_integration import RouterNode, recommend_model, get_client
from langchain_core.messages import HumanMessage

# ── State definition ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_input: str
    intent: Optional[str]
    routing_decision: Optional[str]
    answer: Optional[str]
    final_output: Optional[str]
    models_used: list


# ── Node 1: Intent Classifier ─────────────────────────────────────────────────
# Pattern: @recommend_model decorator
# routing_decision use-case → gemma2-2b (fastest, 0.994 score)
# This runs on every single request — must be near-instant
@recommend_model(
    use_case="routing_decision",
    urgency="high",
    config_path="config.json"
)
def intent_node(state: AgentState, client) -> AgentState:
    print("\n[intent_node] Classifying intent...")
    response = client.invoke(
        message=f"""
        Classify this user message into ONE category:
        categories: question, code_request, summary_request, analysis_request

        Message: "{state['user_input']}"

        Reply with only the category name.
        """,
        hints={"use_case": "routing_decision", "urgency": "high"}
    )
    intent = response.content.strip().lower()
    print(f"  Intent detected: {intent}")
    print(f"  Model used: {client.last_recommendation.winner_model_id}")
    return {
        **state,
        "intent": intent,
        "models_used": state.get("models_used", []) + [
            f"intent_node → {client.last_recommendation.winner_model_id}"
        ]
    }


# ── Node 2: Router Decision ────────────────────────────────────────────────────
# Pattern: RouterNode base class
# routing_decision use-case → phi3-mini or gemma2-2b
class RouterDecisionNode(RouterNode):
    use_case = "routing_decision"
    urgency  = "high"
    config_path = "config.json"

    def run(self, state: AgentState) -> AgentState:
        print("\n[router_node] Deciding processing strategy...")
        response = self.invoke(
            message=f"""
            Given intent="{state['intent']}", decide the processing strategy:
            - "direct_answer" if it's a simple question
            - "deep_analysis" if it needs reasoning
            - "code_task" if it involves coding

            Reply with only the strategy name.
            """
        )
        decision = response.content.strip().lower()
        print(f"  Decision: {decision}")
        print(f"  Model used: {self.client.last_recommendation.winner_model_id}")
        return {
            **state,
            "routing_decision": decision,
            "models_used": state["models_used"] + [
                f"router_node → {self.client.last_recommendation.winner_model_id}"
            ]
        }


# ── Node 3: Main Answer Generator ─────────────────────────────────────────────
# Pattern: Manual get_client() — most flexible, full control over hints per call
def answer_node(state: AgentState) -> AgentState:
    print("\n[answer_node] Generating answer...")

    # Dynamically pick use_case based on routing decision
    decision = state.get("routing_decision", "direct_answer")
    use_case_map = {
        "direct_answer": "chat",
        "deep_analysis": "reasoning",
        "code_task":     "code_generation",
    }
    use_case = use_case_map.get(decision, "chat")

    # urgency based on intent — code tasks can wait, chat should be fast
    urgency = "high" if use_case == "chat" else "normal"

    client = get_client(
        hints={"use_case": use_case, "urgency": urgency},
        config_path="config.json"
    )

    response = client.invoke(
        message=state["user_input"],
        hints={"use_case": use_case, "urgency": urgency}
    )

    print(f"  Use-case selected: {use_case}")
    print(f"  Model used: {client.last_recommendation.winner_model_id}")
    print(f"  Score: {client.last_recommendation.winner_score:.3f}")

    return {
        **state,
        "answer": response.content,
        "models_used": state["models_used"] + [
            f"answer_node({use_case}) → {client.last_recommendation.winner_model_id}"
        ]
    }


# ── Node 4: Output Formatter ──────────────────────────────────────────────────
# Pattern: Direct WrappedLangchainClient usage
# summarization + high urgency → llama3.2-3b (fast, lightweight)
def format_node(state: AgentState) -> AgentState:
    print("\n[format_node] Formatting output...")

    client = WrappedLangchainClient.auto("config.json")
    response = client.invoke(
        message=f"""
        Clean up and format this response for the user.
        Make it clear, concise, and well-structured.
        Keep all important information.

        Response to format:
        {state['answer']}
        """,
        hints={
            "use_case": "summarization",
            "urgency": "high"
        }
    )

    print(f"  Model used: {client.last_recommendation.winner_model_id}")

    return {
        **state,
        "final_output": response.content,
        "models_used": state["models_used"] + [
            f"format_node → {client.last_recommendation.winner_model_id}"
        ]
    }


# ── Build and run the graph ───────────────────────────────────────────────────
def build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        print("langgraph not installed. Run: pip install langgraph")
        return None

    graph = StateGraph(AgentState)

    graph.add_node("intent",   intent_node)
    graph.add_node("router",   RouterDecisionNode())
    graph.add_node("answer",   answer_node)
    graph.add_node("format",   format_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent",  "router")
    graph.add_edge("router",  "answer")
    graph.add_edge("answer",  "format")
    graph.add_edge("format",  END)

    return graph.compile()


def run_agent(question: str):
    print("=" * 60)
    print(f"Input: {question}")
    print("=" * 60)

    app = build_graph()
    if app is None:
        return

    result = app.invoke({
        "user_input": question,
        "intent": None,
        "routing_decision": None,
        "answer": None,
        "final_output": None,
        "models_used": []
    })

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print(result["final_output"])
    print("\nMODELS USED PER NODE:")
    for m in result["models_used"]:
        print(f"  {m}")
    print("=" * 60)


if __name__ == "__main__":
    # Test 1 — General question
    run_agent("What are the main differences between SQL and NoSQL databases?")

    print("\n\n")

    # Test 2 — Code request
    run_agent("Write a Python function to validate an email address using regex")

    print("\n\n")

    # Test 3 — Analysis
    run_agent("Analyze the trade-offs of using microservices vs monolithic architecture for a startup")