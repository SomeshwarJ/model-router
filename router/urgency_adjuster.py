from dataclasses import dataclass
from .config_loader import UseCaseWeights

URGENCY_SHIFT = {
    "high":   {"latency": +0.25, "quality": -0.25},  # speed critical
    "normal": {"latency":  0.00, "quality":  0.00},  # no change
    "low":    {"latency": -0.20, "quality": +0.20},  # quality over speed
}

VALID_URGENCY = set(URGENCY_SHIFT.keys())

@dataclass
class AdjustedWeights:
    quality: float
    latency: float
    cost: float
    urgency_applied: str

    def as_dict(self) -> dict:
        return {
            "quality": self.quality,
            "latency": self.latency,
            "cost": self.cost
        }

    def summary(self) -> str:
        return f"Weights after urgency='{self.urgency_applied}': quality={self.quality:.2f}, latency={self.latency:.2f}, cost={self.cost:.2f}"

def adjust_weights(base_weights: UseCaseWeights, urgency: str) -> AdjustedWeights:
    if urgency not in VALID_URGENCY:
        print(
            f"[urgency_adjuster] Unknown urgency '{urgency}', falling back to 'normal'. Valid values: {VALID_URGENCY}"
        )

        urgency = "normal"

    shift = URGENCY_SHIFT[urgency]

    raw_quality = base_weights.quality + shift["quality"]
    raw_latency = base_weights.latency + shift["latency"]
    raw_cost = base_weights.cost

    clamped_quality = max(0.0, min(1.0, raw_quality))
    clamped_latency = max(0.0, min(1.0, raw_latency))
    clamped_cost = max(0.0, min(1.0, raw_cost))

    total = clamped_quality + clamped_latency + clamped_cost
    if total == 0.0:
        clamped_quality = clamped_latency = clamped_cost = round(1.0 / 3, 4)
        total = 1.0

    final_quality = round(clamped_quality / total, 6)
    final_latency = round(clamped_latency / total, 6)
    final_cost = round(clamped_cost / total, 6)

    diff = round(1.0 - (final_quality + final_latency + final_cost), 6)
    final_quality += diff  # absorb rounding error in quality

    adjusted = AdjustedWeights(
        quality=final_quality,
        latency=final_latency,
        cost=final_cost,
        urgency_applied=urgency
    )

    print(f"[urgency_adjuster] Base: quality={base_weights.quality}, "
          f"latency={base_weights.latency}, cost={base_weights.cost}")
    print(f"[urgency_adjuster] {adjusted.summary()}")

    return adjusted
