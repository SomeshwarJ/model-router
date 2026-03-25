"""
Module 3 — filter_engine.py
Eliminates models that cannot handle the request BEFORE scoring.
Runs hard filters: health, context_length, quality_score, prefer_cost.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from.config_loader import ModelConfig, UseCaseConfig

@dataclass
class FilterResult:
    survivors: List[ModelConfig]
    eliminated: Dict[str, str] = field(default_factory=dict)

    @property
    def has_survivors(self) -> bool:
        return len(self.survivors) > 0

    def summary(self):
        lines = [f"Filter result: {len(self.survivors)} survivors, {len(self.eliminated)} eliminated"]
        for model_id, reason in self.eliminated.items():
            lines.append(f"  ✗ {model_id}: {reason}")
        for m in self.survivors:
            lines.append(f"  ✓ {m.id}")
        return "\n".join(lines)


def _apply_health_filter(
        models: List[ModelConfig],
        health_status: Dict[str, bool],
        eliminated: Dict[str, str],
) -> List[ModelConfig]:
    survivors = []
    for model in models:
        if not health_status.get(model.id, True):
            eliminated[model.id] = f"health check failed (offline)"
        else:
            survivors.append(model)
    return survivors


def _apply_quality_filter(
        models: List[ModelConfig],
        min_quality: float,
        eliminated: Dict[str, str]
) -> List[ModelConfig]:
    if min_quality <= 0.0:
        return models
    survivors = []
    for model in models:
        if model.metadata.quality_score < min_quality:
            eliminated[model.id] = f"quality_score {model.metadata.quality_score} below minimum {min_quality}"
        else:
            survivors.append(model)
    return survivors


def _apply_context_filer(
        models: List[ModelConfig],
        requires_context: int,
        eliminated: Dict[str, str],
) -> List[ModelConfig]:
    if requires_context <= 0:
        return models
    survivors = []
    for model in models:
        if model.metadata.context_length < requires_context:
            eliminated[model.id] = f"context_length {model.metadata.context_length} below minimum {requires_context}"
        else:
            survivors.append(model)
    return survivors


def _apply_cost_filter(
        models: List[ModelConfig],
        prefer_cost: Optional[str],
        eliminated: Dict[str, str],
) -> List[ModelConfig]:
    if prefer_cost != "free":
        return models
    survivors = []
    for model in models:
        if model.metadata.cost_score < 1.0:
            eliminated[model.id] = f"prefer_cost='free' but cost_score={model.metadata.cost_score}"
        else:
            survivors.append(model)
    return survivors


def apply_filters(
        all_models: List[ModelConfig],
        use_case: UseCaseConfig,
        health_status: Dict[str, bool],
        hints: Dict
) -> FilterResult:

    eliminated: Dict[str, str] = {}
    models = list(all_models)

    print(f"[filter_engine] Starting with {len(models)} candidates for use-case '{use_case.name}'")

    models = _apply_health_filter(models, health_status, eliminated)
    print(f"[filter_engine] After health filter: {len(models)} remaining")

    min_quality = use_case.minimum_requirements.quality_score
    models = _apply_quality_filter(models, min_quality, eliminated)
    print(f"[filter_engine] After quality filter (min={min_quality}): {len(models)} remaining")

    uc_min_context = use_case.minimum_requirements.context_length
    hint_context = int(hints.get("input_token_estimate", 0))
    required_context = max(uc_min_context, hint_context)

    models = _apply_context_filer(models, required_context, eliminated)
    print(f"[filter_engine] After context filter (min={required_context}): {len(models)} remaining")

    prefer_cost = hints.get("prefer_cost")
    models =_apply_cost_filter(models, prefer_cost, eliminated)
    print(f"[filter_engine] After cost filter (prefer={prefer_cost}): {len(models)} remaining")

    result = FilterResult(survivors=models, eliminated=eliminated)

    if not result.has_survivors:
        print(
            f"[filter_engine] WARNING: All models eliminated for use-case '{use_case.name}'. Fallback group '{use_case.fallback_group}' will be used."
        )

    return result
