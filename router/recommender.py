"""
Module 6 — recommender.py
Orchestrates Modules 1-5. Picks the winner, builds explanation,
handles fallback, writes feedback log.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .config_loader import RouterConfig, UseCaseConfig, ModelConfig
from .health_checker import check_health
from .filter_engine import apply_filters, FilterResult
from .urgency_adjuster import adjust_weights
from .scorer import score_models, ScoredModel


@dataclass
class RecommendationResult:
    winner_model_id: str
    winner_model: ModelConfig
    winner_score: float
    use_case_name: str
    urgency: str
    fallback_used: bool
    fallback_group: Optional[str]
    all_scores: List[ScoredModel]
    eliminated: Dict[str, str]
    explanation: str
    hints_applied: Dict = field(default_factory=dict)
    use_case_auto_detected: bool = False


VALID_USE_CASES = ["summarization", "reasoning", "code_generation", "routing_decision", "rag_answer", "chat", "data_extraction", "long_context"]

def _auto_detect_use_case(message: str, config:RouterConfig) -> str:
    print("[recommender] Auto-detecting use-case via phi3-mini...")

    uc_names = list(config.use_cases.keys())
    uc_list = ", ".join(uc_names)

    prompt = (
        f"Classify this user message into exactly ONE of these use-cases: {uc_list}\n\n"
        f"User message: \"{message}\"\n\n"
        f"Reply with only the use-case name, nothing else."
    )

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage

        classifier_model_id = "phi3-mini"
        if classifier_model_id in config.models:
            classifier_model_id = list(config.models.keys())[0]

        classifier_cfg = config.models[classifier_model_id]
        llm = ChatOllama(
            model=classifier_cfg.model_name,
            temperature=0.0,
            num_predict=20
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        detected = response.content.strip().lower().replace("", "_")

        if detected not in config.use_cases:
            print(f"[recommender] Auto-detected use-case: '{detected}'")
            return detected
        else:
            print(
                f"[recommender] Auto-detection returned unknown use-case '{detected}', falling back to 'chat'"
            )
            return "chat"

    except Exception as e:
        print(f"[recommender] Auto-detection failed ({e}), falling back to 'chat'")
        return "chat"


def _build_explanation(
        winner: ScoredModel,
        use_case_name: str,
        urgency: str,
        eliminated: Dict[str, str],
        fallback_used: bool,
        fallback_group: Optional[str],
        all_scores: List[ScoredModel],
) -> str:
    lines = []

    if fallback_used:
        lines.append(f"⚠ No model passed all filters for use-case '{use_case_name}'. Fell back to group '{fallback_group}'.")

    tag_note = ""
    if winner.tag_bonus_applied:
        tag_note = f" (tag bonus +{winner.final_score - winner.base_score:.2f} for {winner.matched_tags})"

    lines.append(
        f"✓ Selected: {winner.model_id} "
        f"(score={winner.final_score:.3f}{tag_note}) "
        f"for use-case='{use_case_name}', urgency='{urgency}'"
    )

    if len(all_scores) > 1:
        runner_up = all_scores[1]
        lines.append(f"  Runner-up: {runner_up.model_id} (score={runner_up.final_score:.3f})")

    if eliminated:
        lines.append("  Eliminated:")
        for model_id, reason in eliminated.items():
            lines.append(f"    ✗ {model_id}: {reason}")

    return "\n".join(lines)


def _log_feedback(
        result: RecommendationResult,
        config: RouterConfig,
        start_time: float
):
    if not config.settings.feedback_logging_enabled:
        return

    log_path = config.settings.feedback_log_path

    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "use_case": result.use_case_name,
        "use_case_auto_detected": result.use_case_auto_detected,
        "urgency": result.urgency,
        "hints_applied": result.hints_applied,
        "winner": result.winner_model_id,
        "winner_score": result.winner_score,
        "fallback_used": result.fallback_used,
        "fallback_group": result.fallback_group,
        "all_scores": [
            {"model_id": s.model_id, "score": s.final_score}
            for s in result.all_scores
        ],
        "eliminated": result.eliminated,
        "recommendation_time_ms": round((time.time() - start_time) * 1000, 1)
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"[recommender] Feedback logged to {log_path}")


def recommend(
        message: str,
        hints: Dict,
        config: RouterConfig
) -> RecommendationResult:

    start_time = time.time()

    auto_detected = False
    use_case_name = hints["use_case"]

    if use_case_name:
        if use_case_name not in config.use_cases:
            raise ValueError(
                f"hints['use_case'] = '{use_case_name}' is not defined in config. Available: {list(config.use_cases.keys())}"
            )
        print(f"[recommender] Use-case from hints: '{use_case_name}'")

    else:
        use_case_name = _auto_detect_use_case(message, config)
        auto_detected = True

    use_case: UseCaseConfig = config.use_cases[use_case_name]
    urgency = hints.get("urgency", "normal")

    all_models = list(config.models.values())
    health_status = check_health(all_models, config.settings)

    filter_result: FilterResult = apply_filters(
        all_models = all_models,
        use_case = use_case,
        health_status = health_status,
        hints = hints,
    )

    fallback_used = False
    fallback_group = None
    all_scores: List[ScoredModel] = []

    if filter_result.has_survivors:
        adjusted_weights = adjust_weights(use_case.weights, urgency)
        all_scores = score_models(
            survivors=filter_result.survivors,
            use_case=use_case,
            adjusted_weights=adjusted_weights,
            settings=config.settings
        )

    else:
        fallback_used = True
        fallback_group = use_case.fallback_group
        fallback_cfg = config.groups[fallback_group]

        print(
            f"[recommender] Activating fallback group '{fallback_group}' "
            f"with models: {fallback_cfg.models}"
        )

        fallback_models = [
            config.models[mid]
            for mid in fallback_cfg.models
            if mid in config.models and health_status.get(mid, False)
        ]

        if not fallback_models:
            raise RuntimeError(
                f"All models failed — including all models in fallback group "
                f"'{fallback_group}'. Models tried: "
                f"{fallback_cfg.models}. "
                f"Check that Ollama is running."
            )

        adjusted_weights = adjust_weights(use_case.weights, urgency)
        all_scores = score_models(
            survivors=fallback_models,
            adjusted_weights=adjusted_weights,
            use_case=use_case,
            settings=config.settings
        )

    winner: ScoredModel = all_scores[0]

    explanation = _build_explanation(
        winner=winner,
        use_case_name=use_case_name,
        urgency=urgency,
        eliminated=filter_result.eliminated,
        fallback_used=fallback_used,
        fallback_group=fallback_group,
        all_scores=all_scores
    )

    print(f"[recommender]\n{explanation}")

    result = RecommendationResult(
        winner_model_id=winner.model_id,
        winner_model=winner.model,
        winner_score=winner.final_score,
        use_case_name=use_case_name,
        urgency=urgency,
        fallback_used=fallback_used,
        fallback_group=fallback_group,
        all_scores=all_scores,
        eliminated=filter_result.eliminated,
        explanation=explanation,
        hints_applied=hints,
        use_case_auto_detected=auto_detected
    )

    # --- Step 8: Log ---
    _log_feedback(result, config, start_time)

    return result