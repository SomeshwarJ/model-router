"""
Module 1 — config_loader.py
Reads config.json and converts it into clean Python dataclasses.
Every other module depends on the output of this one.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class ModelMetadata:
    quality_score: float
    latency_score: float
    cost_score: float
    context_length: int
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        for attr in ("quality_score", "latency_score", "cost_score"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"ModelMetadata.{attr} must be between 0.0 and 1.0, got {val}")

            if self.context_length <= 0:
                raise ValueError(f"ModelMetadata.context_length must be > 0, got {self.context_length}")


@dataclass
class ModelConfig:
    id: str
    model_name: str
    provider: str
    parameters: Dict[str, Any]
    metadata: ModelMetadata
    api_key: Optional[str] = None


@dataclass
class GroupConfig:
    name: str
    models: List[str]
    routing_strategy: str

    def __post_init__(self):
        valid_strategies = {"round-robin", "priority"}
        if self.routing_strategy not in valid_strategies:
            raise ValueError(f"Group '{self.name}': routing_strategy must be one of {valid_strategies}, got '{self.routing_strategy}'")

@dataclass
class UseCaseWeights:
    quality: float = 0.0
    latency: float = 0.0
    cost: float = 0.0

    def __post_init__(self):
        total = round(self.quality + self.latency + self.cost, 6)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"UseCaseWeights must sum to 1.0, got {total} (quality={self.quality}, latency={self.latency}, cost={self.cost})")

@dataclass
class MinimumRequirements:
    quality_score: float = 0.0
    context_length: int = 0

@dataclass
class UseCaseConfig:
    name: str
    description: str
    weights: UseCaseWeights
    minimum_requirements: MinimumRequirements
    preferred_tags: List[str]
    fallback_group: str

@dataclass
class RecommendationSettings:
    health_check_enabled: bool = True
    health_check_timeout_seconds: int = 3
    score_tag_bonus: float = 0.05
    minimum_composite_score: float = 0.50
    feedback_logging_enabled: bool = True
    feedback_log_path: str = "logs/feedback.jsonl"

@dataclass
class RouterConfig:
    models: Dict[str, ModelConfig]
    groups: Dict[str, GroupConfig]
    use_cases: Dict[str, UseCaseConfig]
    settings: RecommendationSettings


def _resolve_env(value: Optional[str]) -> Optional[str]:
    if value and isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        resolved = os.environ.get(env_var)
        if resolved is None:
            print(f"[config_loader] WARNING: env var '{env_var}' is not set")
        return resolved
    return value


def _parse_models(raw_models: list) -> Dict[str, ModelConfig]:
    models = {}
    for i, m in enumerate(raw_models):
        # Check required fields exist
        for required in ("id", "model_name", "provider", "metadata"):
            if required not in m:
                raise ValueError(f"models[{i}]: missing required field '{required}'")

        # Validate metadata has required fields
        meta_raw = m["metadata"]
        for required_meta in ("quality_score", "latency_score", "cost_score", "context_length"):
            if required_meta not in meta_raw:
                raise ValueError(f"models[{i}] (id='{m.get('id', '?')}') metadata: missing required field '{required_meta}'")

        # Create ModelMetadata object (runs __post_init__ validation)
        metadata = ModelMetadata(
            quality_score=float(meta_raw["quality_score"]),
            latency_score=float(meta_raw["latency_score"]),
            cost_score=float(meta_raw["cost_score"]),
            context_length=int(meta_raw["context_length"]),
            tags=meta_raw.get("tags", [])
        )

        # Create ModelConfig object
        models[m.get("id")] = ModelConfig(
            id=m.get("id"),
            model_name=m.get("model_name"),
            provider=m.get("provider", "ollama"),
            parameters=m.get("parameters", {}),
            metadata=metadata,
            api_key=_resolve_env(m.get("api_key")),
        )

    return models


def _parse_groups(raw_groups: dict, known_model_ids: List[str]) -> Dict[str, GroupConfig]:
    groups = {}
    for group_name, group_data in raw_groups.items():
        # Check required fields
        for required in ("models", "routing_strategy"):
            if required not in group_data:
                raise ValueError(f"groups.{group_name}: missing required field '{required}'")

        # Validate all models exist in models section
        for model_id in group_data["models"]:
            if model_id not in known_model_ids:
                raise ValueError(f"groups.{group_name}: model '{model_id}' is not defined under 'models'. Known models: {known_model_ids}")

        # Validate routing_strategy (done in __post_init__)
        groups[group_name] = GroupConfig(
            name=group_name,
            models=group_data["models"],
            routing_strategy=group_data["routing_strategy"]
        )
    return groups


def _parse_use_cases(raw_use_cases: dict, known_group_names: List[str]) -> Dict[str, UseCaseConfig]:
    use_cases = {}
    for uc_name, uc_data in raw_use_cases.items():
        # Check all required fields exist
        for required in ("description", "weights", "minimum_requirements",
                         "preferred_tags", "fallback_group"):
            if required not in uc_data:
                raise ValueError(
                    f"use_cases.{uc_name}: missing required field '{required}'"
                )

        # Validate fallback_group exists
        if uc_data["fallback_group"] not in known_group_names:
            raise ValueError(
                f"use_cases.{uc_name}: fallback_group '{uc_data['fallback_group']}' "
                f"not defined under 'groups'. Known groups: {known_group_names}"
            )

        # Parse weights (must sum to 1.0)
        weights_raw = uc_data["weights"]
        weights = UseCaseWeights(
            quality=float(weights_raw.get("quality", 0.0)),
            latency=float(weights_raw.get("latency", 0.0)),
            cost=float(weights_raw.get("cost", 0.0))
        )

        # Parse minimum requirements
        min_req_raw = uc_data["minimum_requirements"]
        min_req = MinimumRequirements(
            quality_score=float(min_req_raw.get("quality_score", 0.0)),
            context_length=int(min_req_raw.get("context_length", 0))
        )

        # Create UseCaseConfig
        use_cases[uc_name] = UseCaseConfig(
            name=uc_name,
            description=uc_data["description"],
            weights=weights,
            minimum_requirements=min_req,
            preferred_tags=uc_data.get("preferred_tags", []),
            fallback_group=uc_data["fallback_group"]
        )

    return use_cases


def _parse_settings(raw_settings: Optional[dict]) -> RecommendationSettings:
    if not raw_settings:
        return RecommendationSettings()
    return RecommendationSettings(
        health_check_enabled=raw_settings.get("health_check_enabled", True),
        health_check_timeout_seconds=int(raw_settings.get("health_check_timeout_seconds", 3)),
        score_tag_bonus=float(raw_settings.get("score_tag_bonus", 0.05)),
        minimum_composite_score=float(raw_settings.get("minimum_composite_score", 0.50)),
        feedback_logging_enabled=raw_settings.get("feedback_logging_enabled", True),
        feedback_log_path=raw_settings.get("feedback_log_path", "logs/feedback.jsonl")
    )


def load_config(config_path: str) -> RouterConfig:
    if not os.path.isfile(config_path):
        raise ValueError(f"Config file not found: '{config_path}'")

    with open(config_path, "r") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"config.json is not valid JSON: {e.msg}", e.doc, e.pos)

    for required_key in ("models", "use_cases", "settings", "groups"):
        if required_key not in raw:
            raise ValueError(f"config.json: missing required top-level key '{required_key}'")

    if not isinstance(raw["models"], list) or len(raw["models"]) == 0:
        raise ValueError("config.json: 'models' must be a non-empty list")

    if not isinstance(raw["groups"], dict) or len(raw["groups"]) == 0:
        raise ValueError("config.json: 'groups' must be a non-empty object")

    if not isinstance(raw["use_cases"], dict) or len(raw["use_cases"]) == 0:
        raise ValueError("config.json: 'use_cases' must be a non-empty object")

    models = _parse_models(raw["models"])
    groups = _parse_groups(raw["groups"], known_model_ids=list(models.keys()))
    use_cases = _parse_use_cases(raw["use_cases"], known_group_names=list(groups.keys()))
    settings = _parse_settings(raw.get("recommendation_settings"))

    config = RouterConfig(
        models=models,
        groups=groups,
        use_cases=use_cases,
        settings=settings,
    )
    print(
        f"[config_loader] Loaded: {len(models)} models | "
        f"{len(groups)} groups | {len(use_cases)} use-cases"
    )
    return config