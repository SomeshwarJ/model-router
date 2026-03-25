from dataclasses import dataclass
from typing import List
from .config_loader import ModelConfig, UseCaseConfig, RecommendationSettings
from .urgency_adjuster import AdjustedWeights


@dataclass
class ScoredModel:
    model_id: str
    model: ModelConfig
    base_score: float
    tag_bonus_applied: bool
    matched_tags: List[str]
    final_score: float

    def summary(self) -> str:
        tag_info = f" +tag_bonus({self.matched_tags})" if self.tag_bonus_applied else ""
        return (
            f"{self.model_id}: "
            f"base={self.base_score:.4f}{tag_info} "
            f"→ final={self.final_score:.4f}"
        )

def score_models(
        survivors: List[ModelConfig],
        adjusted_weights: AdjustedWeights,
        use_case: UseCaseConfig,
        settings: RecommendationSettings,
) -> List[ScoredModel]:
    if not survivors:
        print("[scorer] No survivors to score — returning empty list")
        return []

    w = adjusted_weights
    scored: List[ScoredModel] = []

    for model in survivors:
        meta = model.metadata

        base_score = (
            (meta.quality_score * w.quality) +
            (meta.latency_score * w.latency) +
            (meta.cost_score * w.cost)
        )
        base_score = round(base_score, 6)

        matched_tags = [t for t in meta.tags if t in use_case.preferred_tags]
        tag_bonus_applied = len(matched_tags) > 0
        bonus = settings.score_tag_bonus if tag_bonus_applied else 0.0

        final_score = round(min(1.0, base_score + bonus), 6)

        if final_score < settings.minimum_composite_score:
            print(f"[scorer] WARNING: {model.id} final_score={final_score:.4f} is below minimum_composite_score={settings.minimum_composite_score}")

        scored.append(ScoredModel(
            model_id=model.id,
            model=model,
            base_score=base_score,
            tag_bonus_applied=tag_bonus_applied,
            matched_tags=matched_tags,
            final_score=final_score
        ))

    scored.sort(key=lambda s: s.final_score, reverse=True)

    print(f"[scorer] Scores for use-case '{use_case.name}' (urgency='{w.urgency_applied}'):")
    for s in scored:
        print(f"  {s.summary()}")

    return scored

