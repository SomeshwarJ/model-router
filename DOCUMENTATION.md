# Model Recommendation - Comprehensive Documentation 📚

**Version:** 1.0  
**Date:** March 26, 2026  
**Purpose:** Intelligent LLM Model Recommendation System

---

## Table of Contents

1. [Overview](#overview)
2. [What is Dataclasses?](#what-is-dataclasses)
3. [Module 1: config_loader.py](#module-1-config_loaderpy)
4. [Module 2: health_checker.py](#module-2-health_checkerpy)
5. [Module 3: filter_engine.py](#module-3-filter_enginepy)
6. [Module 4: urgency_adjuster.py](#module-4-urgency_adjusterpy)
7. [Module 5: scorer.py](#module-5-scorerpy)
8. [Module 6: recommender.py](#module-6-recommenderpy)
9. [Module 7: wrapped_client.py](#module-7-wrapped_clientpy)
10. [Complete Pipeline Flow](#complete-pipeline-flow)
11. [Example Workflows](#example-workflows)

---

# Overview

## What is Model Recommendation?

The **Model Recommendation** system is an intelligent system that automatically selects the best language model for each task based on:

- **Quality** - Accuracy and correctness
- **Speed** - Response latency
- **Cost** - Resource consumption
- **Context** - Maximum token handling
- **Availability** - Model health status
- **Specialization** - Task-specific tags

Instead of using one model for everything, the router analyzes each request and picks the optimal model from a pool of available models.

## Why Model Recommendation?

✅ **Cost Optimization** - Use cheaper models for simple tasks  
✅ **Performance** - Fast response for time-sensitive queries  
✅ **Quality** - Best model for complex reasoning  
✅ **Reliability** - Automatic fallback if primary model fails  
✅ **Scalability** - Distribute load across multiple models  

---

# What is Dataclasses?

## Definition

**Dataclasses** are a Python feature (3.7+) that automatically generates special methods for classes, reducing boilerplate code.

### What They Generate Automatically:

- `__init__()` - Constructor with all fields
- `__repr__()` - String representation
- `__eq__()` - Equality comparison
- Type hints - Built-in type checking

### Example:

```python
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ModelMetadata:
    quality_score: float
    latency_score: float
    cost_score: float
    context_length: int
    tags: List[str] = field(default_factory=list)  # Default empty list
    
    def __post_init__(self):  # Custom validation after __init__
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError(f"quality_score must be 0-1, got {self.quality_score}")
```

### Without Dataclass (Verbose):

```python
class ModelMetadata:
    def __init__(self, quality_score, latency_score, cost_score, context_length, tags=None):
        self.quality_score = quality_score
        self.latency_score = latency_score
        self.cost_score = cost_score
        self.context_length = context_length
        self.tags = tags or []
    
    def __repr__(self):
        return f"ModelMetadata(quality_score={self.quality_score}, ...)"
    
    def __eq__(self, other):
        return (self.quality_score == other.quality_score and 
                self.latency_score == other.latency_score and ...)
```

### With Dataclass (Clean):

```python
@dataclass
class ModelMetadata:
    quality_score: float
    latency_score: float
    cost_score: float
    context_length: int
    tags: List[str] = field(default_factory=list)
```

**Benefits:**
- ✅ Less code to write
- ✅ Self-documenting (types are clear)
- ✅ Automatic `__repr__` for debugging
- ✅ Easy serialization to JSON
- ✅ Built-in validation via `__post_init__`

---

# Module 1: config_loader.py

## Purpose

**config_loader.py** is the **foundation** of the entire system. It:

1. ✅ Loads and parses `config.json`
2. ✅ Validates all data against strict rules
3. ✅ Converts JSON into Python dataclass objects
4. ✅ Ensures config integrity before recommendation engine runs
5. ✅ Provides helpful error messages

## Architecture

### Dataclasses Defined

All dataclasses validate their data in `__post_init__()`:

```
RouterConfig (Main)
├── ModelConfig[]
│   ├── id: str
│   ├── model_name: str
│   ├── provider: str
│   ├── metadata: ModelMetadata
│   │   ├── quality_score: 0.0-1.0
│   │   ├── latency_score: 0.0-1.0
│   │   ├── cost_score: 0.0-1.0
│   │   ├── context_length: int > 0
│   │   └── tags: List[str]
│   └── parameters: Dict
│
├── GroupConfig[]
│   ├── name: str
│   ├── models: List[str] (must exist in models)
│   └── routing_strategy: "priority" | "round-robin"
│
├── UseCaseConfig[]
│   ├── name: str
│   ├── weights: UseCaseWeights (sums to 1.0)
│   │   ├── quality: 0.0-1.0
│   │   ├── latency: 0.0-1.0
│   │   └── cost: 0.0-1.0
│   ├── minimum_requirements: MinimumRequirements
│   │   ├── quality_score: 0.0-1.0
│   │   └── context_length: int ≥ 0
│   ├── preferred_tags: List[str]
│   └── fallback_group: str (must exist in groups)
│
└── RecommendationSettings
    ├── health_check_enabled: bool
    ├── health_check_timeout_seconds: int
    ├── score_tag_bonus: float
    ├── minimum_composite_score: float
    ├── feedback_logging_enabled: bool
    └── feedback_log_path: str
```

## Key Validation Rules

### Rule 1: Score Values Must Be 0.0-1.0

```python
@dataclass
class ModelMetadata:
    quality_score: float
    latency_score: float
    cost_score: float
    
    def __post_init__(self):
        for attr in ("quality_score", "latency_score", "cost_score"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be between 0.0 and 1.0, got {val}")
```

**From your config.json:**
```json
{
  "models": [
    {
      "id": "codellama",
      "metadata": {
        "quality_score": 0.88,      ✓ Valid (0.0-1.0)
        "latency_score": 0.55,      ✓ Valid (0.0-1.0)
        "cost_score": 1.0           ✓ Valid (0.0-1.0)
      }
    }
  ]
}
```

### Rule 2: Context Length Must Be > 0

```python
if self.context_length <= 0:
    raise ValueError(f"context_length must be > 0, got {self.context_length}")
```

**From your config.json:**
```json
{
  "metadata": {
    "context_length": 16384   ✓ Valid (> 0)
  }
}
```

### Rule 3: Weights Must Sum to 1.0

```python
@dataclass
class UseCaseWeights:
    quality: float
    latency: float
    cost: float
    
    def __post_init__(self):
        total = round(self.quality + self.latency + self.cost, 6)
        if abs(total - 1.0) > 0.01:  # Allow 1% rounding error
            raise ValueError(f"Weights must sum to 1.0, got {total}")
```

**From your config.json:**
```json
{
  "use_cases": {
    "code_generation": {
      "weights": {
        "quality": 0.60,    ✓
        "latency": 0.25,    ✓
        "cost": 0.15        ✓ Sum = 1.0 ✓
      }
    },
    "summarization": {
      "weights": {
        "latency": 0.40,    ✓
        "cost": 0.35,       ✓
        "quality": 0.25     ✓ Sum = 1.0 ✓
      }
    }
  }
}
```

**Invalid example (would fail):**
```json
{
  "weights": {
    "quality": 0.60,
    "latency": 0.25,
    "cost": 0.10    // Sum = 0.95, fails validation!
  }
}
```

### Rule 4: Routing Strategy Must Be Valid

```python
@dataclass
class GroupConfig:
    routing_strategy: str
    
    def __post_init__(self):
        valid_strategies = {"round-robin", "priority"}
        if self.routing_strategy not in valid_strategies:
            raise ValueError(f"routing_strategy must be one of {valid_strategies}")
```

**From your config.json:**
```json
{
  "groups": {
    "default": {
      "routing_strategy": "round-robin"  ✓ Valid
    },
    "fastest": {
      "routing_strategy": "priority"     ✓ Valid
    }
  }
}
```

### Rule 5: Cross-References Must Exist

```python
# In _parse_groups():
for model_id in group_data["models"]:
    if model_id not in known_model_ids:
        raise ValueError(
            f"groups.{group_name}: model '{model_id}' not defined"
        )

# In _parse_use_cases():
if uc_data["fallback_group"] not in known_group_names:
    raise ValueError(
        f"use_cases.{uc_name}: fallback_group '{fallback}' not defined"
    )
```

**From your config.json:**
```json
{
  "groups": {
    "code": {
      "models": ["codellama", "llama3", "gpt-oss"]  ✓ All exist in models
    }
  },
  "use_cases": {
    "code_generation": {
      "fallback_group": "code"  ✓ Exists in groups
    }
  }
}
```

## Parsing Process

### Step 1: Load and Parse JSON

```python
def load_config(config_path: str) -> RouterConfig:
    with open(config_path, "r") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"config.json is not valid JSON: {e.msg}")
```

**Checks:**
- ✅ File exists
- ✅ Valid JSON syntax

### Step 2: Check Required Top-Level Keys

```python
for required_key in ("models", "use_cases", "settings", "groups"):
    if required_key not in raw:
        raise ValueError(f"config.json: missing required top-level key '{required_key}'")
```

**Your config.json has all 4 keys:** ✓ models, ✓ groups, ✓ use_cases, ✓ settings

### Step 3: Parse Models

```python
def _parse_models(raw_models: list) -> Dict[str, ModelConfig]:
    models = {}
    for i, m in enumerate(raw_models):
        # 1. Check required fields
        for required in ("id", "model_name", "provider", "metadata"):
            if required not in m:
                raise ValueError(f"models[{i}]: missing required field '{required}'")
        
        # 2. Parse metadata (creates ModelMetadata with validation)
        metadata = ModelMetadata(
            quality_score=float(m["metadata"]["quality_score"]),
            latency_score=float(m["metadata"]["latency_score"]),
            cost_score=float(m["metadata"]["cost_score"]),
            context_length=int(m["metadata"]["context_length"]),
            tags=m["metadata"].get("tags", [])
        )
        
        # 3. Create ModelConfig
        models[m["id"]] = ModelConfig(
            id=m["id"],
            model_name=m["model_name"],
            provider=m.get("provider", "ollama"),
            parameters=m.get("parameters", {}),
            metadata=metadata
        )
    
    return models
```

**Your 9 models parsed:**
```
✓ phi3-mini (2048 ctx, 0.65 quality)
✓ gemma2-2b (2048 ctx, 0.60 quality)
✓ llama3.2-3b (4096 ctx, 0.68 quality)
✓ gemma2 (8192 ctx, 0.78 quality)
✓ mistral (8192 ctx, 0.80 quality)
✓ gemini-flash (128000 ctx, 0.82 quality)
✓ codellama (16384 ctx, 0.88 quality)
✓ llama3 (8192 ctx, 0.85 quality)
✓ gpt-oss (16384 ctx, 0.92 quality)
```

### Step 4: Parse Groups

```python
def _parse_groups(raw_groups: dict, known_model_ids: List[str]):
    groups = {}
    for group_name, group_data in raw_groups.items():
        # 1. Check required fields
        for required in ("models", "routing_strategy"):
            if required not in group_data:
                raise ValueError(f"groups.{group_name}: missing '{required}'")
        
        # 2. Validate all models exist
        for model_id in group_data["models"]:
            if model_id not in known_model_ids:
                raise ValueError(
                    f"groups.{group_name}: model '{model_id}' not defined"
                )
        
        # 3. Create GroupConfig (validates routing_strategy)
        groups[group_name] = GroupConfig(
            name=group_name,
            models=group_data["models"],
            routing_strategy=group_data["routing_strategy"]
        )
    
    return groups
```

**Your 6 groups parsed:**
```
✓ default: [llama3, mistral, gemma2] → round-robin
✓ fastest: [phi3-mini, gemma2-2b, llama3.2-3b] → priority
✓ fast: [gemini-flash, gemma2, mistral, llama3.2-3b] → priority
✓ reasoning: [llama3, codellama, gpt-oss] → priority
✓ code: [codellama, llama3, gpt-oss] → priority
✓ long_context: [gemini-flash, codellama, gpt-oss] → priority
```

### Step 5: Parse Use Cases

```python
def _parse_use_cases(raw_use_cases: dict, known_group_names: List[str]):
    use_cases = {}
    for uc_name, uc_data in raw_use_cases.items():
        # 1. Check required fields
        for required in ("description", "weights", "minimum_requirements", 
                         "preferred_tags", "fallback_group"):
            if required not in uc_data:
                raise ValueError(f"use_cases.{uc_name}: missing '{required}'")
        
        # 2. Validate fallback_group exists
        if uc_data["fallback_group"] not in known_group_names:
            raise ValueError(
                f"use_cases.{uc_name}: fallback_group "
                f"'{uc_data['fallback_group']}' not defined"
            )
        
        # 3. Parse weights (validates sum to 1.0)
        weights = UseCaseWeights(
            quality=float(uc_data["weights"].get("quality", 0.0)),
            latency=float(uc_data["weights"].get("latency", 0.0)),
            cost=float(uc_data["weights"].get("cost", 0.0))
        )
        
        # 4. Parse minimum requirements
        min_req = MinimumRequirements(
            quality_score=float(uc_data["minimum_requirements"].get("quality_score", 0.0)),
            context_length=int(uc_data["minimum_requirements"].get("context_length", 0))
        )
        
        # 5. Create UseCaseConfig
        use_cases[uc_name] = UseCaseConfig(
            name=uc_name,
            description=uc_data["description"],
            weights=weights,
            minimum_requirements=min_req,
            preferred_tags=uc_data.get("preferred_tags", []),
            fallback_group=uc_data["fallback_group"]
        )
    
    return use_cases
```

**Your 8 use cases parsed:**

| Use Case | Quality Weight | Latency Weight | Cost Weight | Min Quality | Min Context | Fallback Group |
|----------|---|---|---|---|---|---|
| summarization | 0.25 | 0.40 | 0.35 | 0.60 | 2K | fast |
| reasoning | 0.65 | 0.20 | 0.15 | 0.80 | 4K | reasoning |
| code_generation | 0.60 | 0.25 | 0.15 | 0.75 | 4K | code |
| routing_decision | 0.10 | 0.55 | 0.35 | 0.58 | 1K | fastest |
| rag_answer | 0.55 | 0.25 | 0.20 | 0.75 | 8K | reasoning |
| chat | 0.35 | 0.45 | 0.20 | 0.60 | 2K | default |
| data_extraction | 0.65 | 0.20 | 0.15 | 0.75 | 4K | reasoning |
| long_context | 0.50 | 0.30 | 0.20 | 0.75 | 32K | long_context |

### Step 6: Parse Settings

```python
def _parse_settings(raw_settings: Optional[dict]) -> RecommendationSettings:
    if not raw_settings:
        return RecommendationSettings()  # Use defaults
    
    return RecommendationSettings(
        health_check_enabled=raw_settings.get("health_check_enabled", True),
        health_check_timeout_seconds=int(raw_settings.get("health_check_timeout_seconds", 3)),
        score_tag_bonus=float(raw_settings.get("score_tag_bonus", 0.05)),
        minimum_composite_score=float(raw_settings.get("minimum_composite_score", 0.50)),
        feedback_logging_enabled=raw_settings.get("feedback_logging_enabled", True),
        feedback_log_path=raw_settings.get("feedback_log_path", "logs/feedback.jsonl")
    )
```

**Your settings:**
```
✓ health_check_enabled: true
✓ health_check_timeout_seconds: 3
✓ score_tag_bonus: 0.05
✓ minimum_composite_score: 0.50
✓ feedback_logging_enabled: true
✓ feedback_log_path: "logs/feedback.jsonl"
```

### Step 7: Create Final RouterConfig

```python
config = RouterConfig(
    models=models,                # Dict[id → ModelConfig]
    groups=groups,               # Dict[name → GroupConfig]
    use_cases=use_cases,         # Dict[name → UseCaseConfig]
    settings=settings            # RecommendationSettings
)
```

**Result:**
```
RouterConfig loaded:
  - 9 models
  - 6 groups
  - 8 use-cases
```

## Error Examples

### Error 1: Invalid JSON

```python
# config.json: {"models": [, ...  ← Missing opening brace

# Error:
ValueError: config.json is not valid JSON: Expecting value: line 1 column 15
```

### Error 2: Missing Required Key

```python
# config.json missing "settings" key

# Error:
ValueError: config.json: missing required top-level key 'settings'
```

### Error 3: Score Out of Range

```python
# "quality_score": 1.5  ← Must be 0.0-1.0

# Error:
ValueError: ModelMetadata.quality_score must be between 0.0 and 1.0, got 1.5
```

### Error 4: Weights Don't Sum to 1.0

```python
# "weights": {"quality": 0.60, "latency": 0.25, "cost": 0.10}  ← Sum = 0.95

# Error:
ValueError: UseCaseWeights must sum to 1.0, got 0.95
```

### Error 5: Model Reference Doesn't Exist

```python
# "groups": {
#   "code": {
#     "models": ["codellama", "unknown_model"]  ← Doesn't exist
#   }
# }

# Error:
ValueError: groups.code: model 'unknown_model' not defined in models
```

## Summary

**config_loader.py:**
- ✅ Loads `config.json` with strict validation
- ✅ Catches all config errors before router runs
- ✅ Converts JSON to Python dataclasses
- ✅ Ensures data integrity and consistency
- ✅ Provides clear error messages for troubleshooting

---

# Module 2: health_checker.py

## Purpose

**health_checker.py** verifies that all configured models are:

1. ✅ Online and accessible
2. ✅ Actually available in Ollama
3. ✅ Responsive within timeout

**Key Feature:** Runs health checks **concurrently** (all models pinged simultaneously for speed).

## How It Works

### Step 1: Check If Health Checks Enabled

```python
def check_health(models: List[ModelConfig], settings: RecommendationSettings):
    if not settings.health_check_enabled:  # ← From config.json
        print("[health_checker] Health checks disabled — marking all models healthy")
        return {model.id: True for model in models}  # All True, no pinging!
```

**From your config.json:**
```json
{
  "settings": {
    "health_check_enabled": true  // ← Enabled, so pinging will occur
  }
}
```

If disabled, all models instantly marked as healthy (useful for testing).

### Step 2: Ping Each Model Concurrently

```python
async def _ping_model(model: ModelConfig, timeout: int, client) -> tuple[str, bool]:
    try:
        # Make request to Ollama API
        response = await client.get(
            "http://localhost:11434/api/tags",  # ← Ollama's standard endpoint
            timeout=timeout  # ← From config.json settings
        )
        
        # Check HTTP status
        if response.status_code != 200:
            return (model.id, False)  # ❌ Not healthy
        
        # Parse available models
        data = response.json()
        available_names = [m.get("name", "") for m in data.get("models", [])]
        
        # Check if this model is available
        is_available = any(
            name == model.model_name or 
            name.startswith(model.model_name + ":")
            for name in available_names
        )
        
        return (model.id, is_available)  # ✓ or ❌
    
    except Exception as e:
        print(f"[health_checker] {model.id}: ping failed — {e}")
        return (model.id, False)  # ❌ Failed to ping
```

**Example:**

Your config.json model:
```json
{
  "id": "codellama",
  "model_name": "codellama:latest"
}
```

Ollama's `/api/tags` response:
```json
{
  "models": [
    {"name": "codellama:latest"},
    {"name": "llama3:latest"},
    {"name": "phi3:mini"}
  ]
}
```

Matching:
```
Check: "codellama:latest" == "codellama:latest"  ✓ Match!
Result: ("codellama", True)
```

### Step 3: Run All Pings in Parallel

```python
async def _check_all_async(models: List[ModelConfig], timeout: int):
    async with httpx.AsyncClient() as client:
        # Create one task per model
        tasks = [_ping_model(model, timeout, client) for model in models]
        
        # Run all tasks simultaneously (not sequentially!)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile into dictionary
        health_map: Dict[str, bool] = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            model_id, is_healthy = result
            health_map[model_id] = is_healthy
        
        return health_map
```

**Speed Comparison:**

```
Sequential (one at a time):
  Model 1 ping: 1s
  Model 2 ping: 1s
  Model 3 ping: 1s
  Total: 3s

Concurrent (all at same time):
  Model 1 ping: 1s  \
  Model 2 ping: 1s  } All happen in parallel
  Model 3 ping: 1s  /
  Total: 1s
```

For your 9 models: **~3 seconds total** instead of ~27 seconds!

### Step 4: Print Results

```python
elapsed = time.time() - start
healthy_count = sum(1 for v in health_map.values() if v)

print(f"[health_checker] Done in {elapsed}s — {healthy_count}/{len(models)} models healthy")
for model_id, is_healthy in health_map.items():
    status = "✓ online" if is_healthy else "✗ offline"
    print(f"  {status}  {model_id}")

return health_map
```

## Example Execution

### Scenario: Your 9 Models

**Setup from config.json:**
```json
{
  "settings": {
    "health_check_enabled": true,
    "health_check_timeout_seconds": 3
  },
  "models": [
    {"id": "phi3-mini", "model_name": "phi3:mini"},
    {"id": "gemma2-2b", "model_name": "gemma2:2b"},
    ...
  ]
}
```

**Assuming Ollama has these models:**
- ✓ phi3:mini
- ✓ gemma2:2b
- ✓ llama3.2:3b
- ✓ gemma2:latest
- ✓ mistral:latest
- ✓ gemini-3-flash-preview:latest
- ✓ codellama:latest
- ✗ llama3:latest (not installed)
- ✗ gpt-oss:latest (not installed)

**Output:**
```
[health_checker] Pinging 9 models (timeout=3s)...
[health_checker] Done in 1.23s — 7/9 models healthy
  ✓ online  phi3-mini
  ✓ online  gemma2-2b
  ✓ online  llama3.2-3b
  ✓ online  gemma2
  ✓ online  mistral
  ✓ online  gemini-flash
  ✓ online  codellama
  ✗ offline  llama3
  ✗ offline  gpt-oss
```

**Result dictionary:**
```python
{
    "phi3-mini": True,
    "gemma2-2b": True,
    "llama3.2-3b": True,
    "gemma2": True,
    "mistral": True,
    "gemini-flash": True,
    "codellama": True,
    "llama3": False,     # ❌ Offline
    "gpt-oss": False     # ❌ Offline
}
```

This dictionary is passed to **filter_engine.py** to eliminate offline models!

## Timeout Behavior

**From config.json:**
```json
{
  "settings": {
    "health_check_timeout_seconds": 3  // ← Wait max 3 seconds per model
  }
}
```

**If a model doesn't respond within 3 seconds:**
```python
# After 3 seconds of waiting...
response = await client.get(..., timeout=3)  # Timeout!
return (model.id, False)  # ❌ Mark as offline
```

This prevents health checks from hanging indefinitely.

## Summary

**health_checker.py:**
- ✅ Pings all models concurrently to Ollama API
- ✅ Returns health status for each model
- ✅ Fast (1-3 seconds for all models vs sequential minutes)
- ✅ Respects timeout from config.json settings
- ✅ Can be disabled via config for testing
- ✅ Output used by filter_engine to eliminate offline models

---

# Module 3: filter_engine.py

## Purpose

**filter_engine.py** progressively eliminates unsuitable models using hard filters.

**Why filters?** Fast rejection before expensive scoring stage.

## Filter Pipeline

```
All models (9)
    ↓
[1] Health Filter
    ├─ Remove: offline models
    └─ Result: 7 online
    ↓
[2] Quality Filter
    ├─ Remove: quality < minimum_requirements.quality_score
    └─ Result: 5 models with sufficient quality
    ↓
[3] Context Filter
    ├─ Remove: context_length < minimum_requirements.context_length
    └─ Result: 4 models with sufficient context
    ↓
[4] Cost Filter
    ├─ Remove: cost_score < 1.0 (if prefer_cost="free")
    └─ Result: 3 free models
    ↓
Survivors (pass to scorer.py)
```

## Filter 1: Health Filter

```python
def _apply_health_filter(models: List[ModelConfig], health_status: Dict[str, bool]):
    survivors = []
    for model in models:
        if not health_status.get(model.id, True):  # False = offline
            eliminated[model.id] = "health check failed (offline)"
        else:
            survivors.append(model)
    return survivors
```

**From health_checker.py output:**
```python
health_status = {
    "phi3-mini": True,      ✓
    "llama3": False,        ❌ Offline
    "codellama": True,      ✓
    ...
}
```

**Result:**
```
Input:  [phi3-mini, llama3, codellama, ...]
Output: [phi3-mini, codellama, ...]
Eliminated: {"llama3": "health check failed (offline)"}
```

## Filter 2: Quality Filter

```python
def _apply_quality_filter(models: List[ModelConfig], min_quality: float):
    if min_quality <= 0.0:
        return models  # No requirement
    
    survivors = []
    for model in models:
        if model.metadata.quality_score < min_quality:  # ← From config.json
            eliminated[model.id] = f"quality_score {model.metadata.quality_score} below minimum {min_quality}"
        else:
            survivors.append(model)
    return survivors
```

**Example: code_generation use case**

From config.json:
```json
{
  "use_cases": {
    "code_generation": {
      "minimum_requirements": {
        "quality_score": 0.75
      }
    }
  }
}
```

Models:
```
phi3-mini:    quality=0.65  ← 0.65 < 0.75  ❌ ELIMINATED
gemma2:       quality=0.78  ← 0.78 ≥ 0.75  ✓
codellama:    quality=0.88  ← 0.88 ≥ 0.75  ✓
llama3:       quality=0.85  ← 0.85 ≥ 0.75  ✓
```

**Result:**
```
Input:  [phi3-mini, gemma2, codellama, llama3]
Output: [gemma2, codellama, llama3]
Eliminated: {"phi3-mini": "quality_score 0.65 below minimum 0.75"}
```

## Filter 3: Context Filter

```python
def _apply_context_filter(models: List[ModelConfig], requires_context: int):
    if requires_context <= 0:
        return models  # No requirement
    
    survivors = []
    for model in models:
        if model.metadata.context_length < requires_context:  # ← From config.json
            eliminated[model.id] = f"context_length {model.metadata.context_length} below minimum {requires_context}"
        else:
            survivors.append(model)
    return survivors
```

**How context requirement is calculated:**

```python
uc_min_context = use_case.minimum_requirements.context_length  # From config.json
hint_context = int(hints.get("input_token_estimate", 0))       # User runtime input
required_context = max(uc_min_context, hint_context)            # Use greater value
```

**Example 1: No user input**

From config.json:
```json
{
  "use_cases": {
    "code_generation": {
      "minimum_requirements": {
        "context_length": 4096
      }
    }
  }
}
```

Runtime:
```python
uc_min_context = 4096
hint_context = 0  # User didn't specify
required_context = max(4096, 0) = 4096
```

Models:
```
gemma2-2b:     context=2048   ← 2048 < 4096  ❌ ELIMINATED
llama3.2-3b:   context=4096   ← 4096 ≥ 4096  ✓
gemma2:        context=8192   ← 8192 ≥ 4096  ✓
codellama:     context=16384  ← 16384 ≥ 4096 ✓
```

**Example 2: User provides large document (50,000 tokens)**

```python
hints = {"input_token_estimate": 50000}

uc_min_context = 4096
hint_context = 50000
required_context = max(4096, 50000) = 50000  ← Boosted!
```

Models:
```
gemini-flash:  context=128000  ← 128000 ≥ 50000  ✓ WINNER
codellama:     context=16384   ← 16384 < 50000   ❌ ELIMINATED
llama3:        context=8192    ← 8192 < 50000    ❌ ELIMINATED
```

## Filter 4: Cost Filter

```python
def _apply_cost_filter(models: List[ModelConfig], prefer_cost: Optional[str]):
    if prefer_cost != "free":  # Only filter if explicitly requested
        return models  # No filtering
    
    survivors = []
    for model in models:
        if model.metadata.cost_score < 1.0:  # ← From config.json
            eliminated[model.id] = f"prefer_cost='free' but cost_score={model.metadata.cost_score}"
        else:
            survivors.append(model)
    return survivors
```

**Understanding cost_score from config.json:**

```json
{
  "models": [
    {
      "id": "phi3-mini",
      "metadata": {
        "cost_score": 1.0  // Free (local Ollama)
      }
    },
    {
      "id": "gemini-flash",
      "metadata": {
        "cost_score": 0.7  // Affordable (Google API)
      }
    },
    {
      "id": "gpt-oss",
      "metadata": {
        "cost_score": 0.2  // Expensive (OpenAI API)
      }
    }
  ]
}
```

**Scenario 1: User doesn't care about cost**

```python
prefer_cost = None  # Not specified
if prefer_cost != "free":  # This is True
    return models  # ✅ Return all (no filtering)
```

All models pass through.

**Scenario 2: User wants only free models**

```python
prefer_cost = "free"  # User wants cheap!
if prefer_cost != "free":  # This is False
    # Skip this block

# Apply cost filter
Input:  [phi3-mini(1.0), gemini-flash(0.7), gpt-oss(0.2)]
Output: [phi3-mini(1.0)]  # Only 1.0 (free)
Eliminated: {
    "gemini-flash": "prefer_cost='free' but cost_score=0.7",
    "gpt-oss": "prefer_cost='free' but cost_score=0.2"
}
```

## Complete Example: code_generation Use Case

### Setup from config.json

**Use case:**
```json
{
  "use_cases": {
    "code_generation": {
      "minimum_requirements": {
        "quality_score": 0.75,
        "context_length": 4096
      }
    }
  }
}
```

**Models:**
```json
{
  "models": [
    {"id": "phi3-mini", "metadata": {"quality_score": 0.65, "context_length": 2048}},
    {"id": "gemma2-2b", "metadata": {"quality_score": 0.60, "context_length": 2048}},
    {"id": "llama3.2-3b", "metadata": {"quality_score": 0.68, "context_length": 4096}},
    {"id": "codellama", "metadata": {"quality_score": 0.88, "context_length": 16384}},
    {"id": "llama3", "metadata": {"quality_score": 0.85, "context_length": 8192}}
  ]
}
```

**Health status:**
```
All online ✓
```

**User hints:**
```python
hints = {
    "input_token_estimate": 0,  # No special context requirement
    "prefer_cost": None         # Doesn't care about cost
}
```

### Filter Execution

**Step 1: Health Filter (all pass)**
```
Input:  [phi3-mini, gemma2-2b, llama3.2-3b, codellama, llama3]
Output: [phi3-mini, gemma2-2b, llama3.2-3b, codellama, llama3]
```

**Step 2: Quality Filter (min=0.75)**
```
phi3-mini(0.65):      0.65 < 0.75  ❌
gemma2-2b(0.60):      0.60 < 0.75  ❌
llama3.2-3b(0.68):    0.68 < 0.75  ❌
codellama(0.88):      0.88 ≥ 0.75  ✓
llama3(0.85):         0.85 ≥ 0.75  ✓

Output: [codellama, llama3]
Eliminated: {
    "phi3-mini": "quality_score 0.65 below minimum 0.75",
    "gemma2-2b": "quality_score 0.60 below minimum 0.75",
    "llama3.2-3b": "quality_score 0.68 below minimum 0.75"
}
```

**Step 3: Context Filter (required=4096)**
```
codellama(16384):     16384 ≥ 4096  ✓
llama3(8192):         8192 ≥ 4096   ✓

Output: [codellama, llama3]
Eliminated: (no change)
```

**Step 4: Cost Filter (prefer=None)**
```
prefer_cost != "free"  → True, skip filtering

Output: [codellama, llama3]
Eliminated: (no change)
```

### Final Result

**Survivors (pass to scorer):**
```
✓ codellama (quality=0.88, context=16384)
✓ llama3 (quality=0.85, context=8192)
```

**Eliminated:**
```
✗ phi3-mini: quality_score 0.65 below minimum 0.75
✗ gemma2-2b: quality_score 0.60 below minimum 0.75
✗ llama3.2-3b: quality_score 0.68 below minimum 0.75
```

**Console output:**
```
[filter_engine] Starting with 5 candidates for use-case 'code_generation'
[filter_engine] After health filter: 5 remaining
[filter_engine] After quality filter (min=0.75): 2 remaining
[filter_engine] After context filter (min=4096): 2 remaining
[filter_engine] After cost filter (prefer=None): 2 remaining

Filter result: 2 survivors, 3 eliminated
  ✓ codellama
  ✓ llama3
  ✗ phi3-mini: quality_score 0.65 below minimum 0.75
  ✗ gemma2-2b: quality_score 0.60 below minimum 0.75
  ✗ llama3.2-3b: quality_score 0.68 below minimum 0.75
```

## Summary

**filter_engine.py:**
- ✅ Applies 4 progressive filters: health → quality → context → cost
- ✅ Eliminates unsuitable models quickly
- ✅ Uses requirements from config.json use_cases
- ✅ Uses health status from health_checker.py
- ✅ Respects runtime hints (input_token_estimate, prefer_cost)
- ✅ Returns survivors and elimination reasons
- ✅ Fast rejection before expensive scoring

---

# Module 4: urgency_adjuster.py

## Purpose

**urgency_adjuster.py** dynamically adjusts scoring weights based on **how urgently the user needs a response**.

**Key insight:** Quality and latency are trade-offs. Cost never changes.

## Urgency Levels

### Level 1: HIGH URGENCY ⚡ (Fast response critical)

**Use case:** User says "Answer NOW, I don't care if it's perfect"

**Shift rule:**
```python
quality:  -0.25  (less important)
latency:  +0.25  (more important)
cost:      0.00  (unchanged)
```

### Level 2: NORMAL URGENCY 🎯 (Balanced)

**Use case:** Standard request, no urgency specified

**Shift rule:**
```python
quality:   0.00  (use as-is from config.json)
latency:   0.00  (use as-is from config.json)
cost:      0.00  (unchanged)
```

### Level 3: LOW URGENCY 📚 (Quality over speed)

**Use case:** User says "Take your time, give the best answer"

**Shift rule:**
```python
quality:  +0.20  (more important)
latency:  -0.20  (less important)
cost:      0.00  (unchanged)
```

## Adjustment Process

### Step 1: Start with Base Weights

**From config.json:**
```json
{
  "use_cases": {
    "code_generation": {
      "weights": {
        "quality": 0.60,
        "latency": 0.25,
        "cost": 0.15
      }
    }
  }
}
```

Base: `quality=0.60, latency=0.25, cost=0.15`

### Step 2: Apply Urgency Shift

**HIGH urgency:**
```
raw_quality = 0.60 + (-0.25) = 0.35
raw_latency = 0.25 + (+0.25) = 0.50
raw_cost = 0.15 + 0.00 = 0.15
Total = 1.0 ✓
```

**NORMAL urgency:**
```
raw_quality = 0.60 + 0.00 = 0.60
raw_latency = 0.25 + 0.00 = 0.25
raw_cost = 0.15 + 0.00 = 0.15
Total = 1.0 ✓
```

**LOW urgency:**
```
raw_quality = 0.60 + 0.20 = 0.80
raw_latency = 0.25 + (-0.20) = 0.05
raw_cost = 0.15 + 0.00 = 0.15
Total = 1.0 ✓
```

### Step 3: Clamp to Valid Range

Ensure no weight goes negative or exceeds 1.0:

```python
clamped_quality = max(0.0, min(1.0, raw_quality))
clamped_latency = max(0.0, min(1.0, raw_latency))
clamped_cost = max(0.0, min(1.0, raw_cost))
```

### Step 4: Renormalize

Divide by total to ensure sum = 1.0:

```python
total = clamped_quality + clamped_latency + clamped_cost

final_quality = round(clamped_quality / total, 6)
final_latency = round(clamped_latency / total, 6)
final_cost = round(clamped_cost / total, 6)
```

(In practice, since we only shift quality/latency equally, total already = 1.0)

## Complete Example

### Code Generation, Three Urgencies

**Base weights from config.json:**
```
quality=0.60, latency=0.25, cost=0.15
```

**Models:**
```
codellama:    quality=0.88, latency=0.55 (fast)
llama3:       quality=0.85, latency=0.60 (faster)
phi3-mini:    quality=0.65, latency=0.95 (fastest)
```

---

### Scenario 1: HIGH URGENCY ⚡

**Adjusted weights:**
```
quality=0.35, latency=0.50, cost=0.15
```

**Scoring:**
```
codellama:
  score = (0.88 × 0.35) + (0.55 × 0.50) + (1.0 × 0.15)
        = 0.308 + 0.275 + 0.15
        = 0.733

llama3:
  score = (0.85 × 0.35) + (0.60 × 0.50) + (1.0 × 0.15)
        = 0.2975 + 0.30 + 0.15
        = 0.7475

phi3-mini:
  score = (0.65 × 0.35) + (0.95 × 0.50) + (1.0 × 0.15)
        = 0.2275 + 0.475 + 0.15
        = 0.8525  ← WINNER! Fastest model
```

**Winner: phi3-mini** (prioritized speed)

---

### Scenario 2: NORMAL URGENCY 🎯

**Adjusted weights:**
```
quality=0.60, latency=0.25, cost=0.15  (unchanged)
```

**Scoring:**
```
codellama:
  score = (0.88 × 0.60) + (0.55 × 0.25) + (1.0 × 0.15)
        = 0.528 + 0.1375 + 0.15
        = 0.8155  ← WINNER! Best quality

llama3:
  score = (0.85 × 0.60) + (0.60 × 0.25) + (1.0 × 0.15)
        = 0.51 + 0.15 + 0.15
        = 0.81

phi3-mini:
  score = (0.65 × 0.60) + (0.95 × 0.25) + (1.0 × 0.15)
        = 0.39 + 0.2375 + 0.15
        = 0.7775
```

**Winner: codellama** (balanced approach)

---

### Scenario 3: LOW URGENCY 📚

**Adjusted weights:**
```
quality=0.80, latency=0.05, cost=0.15
```

**Scoring:**
```
codellama:
  score = (0.88 × 0.80) + (0.55 × 0.05) + (1.0 × 0.15)
        = 0.704 + 0.0275 + 0.15
        = 0.8815  ← WINNER! Best quality

llama3:
  score = (0.85 × 0.80) + (0.60 × 0.05) + (1.0 × 0.15)
        = 0.68 + 0.03 + 0.15
        = 0.86

phi3-mini:
  score = (0.65 × 0.80) + (0.95 × 0.05) + (1.0 × 0.15)
        = 0.52 + 0.0475 + 0.15
        = 0.7175
```

**Winner: codellama** (even more dominant)

---

### Comparison

```
┌──────────┬─────────┬─────────┬──────────┐
│ Urgency  │ Quality │ Latency │ Winner   │
├──────────┼─────────┼─────────┼──────────┤
│ HIGH ⚡  │  0.35   │  0.50   │ phi3     │
│ NORMAL 🎯│  0.60   │  0.25   │ codellama│
│ LOW 📚   │  0.80   │  0.05   │ codellama│
└──────────┴─────────┴─────────┴──────────┘
```

**Key insight:** Same models, different urgencies = different winners!

## Implementation

```python
def adjust_weights(base_weights: UseCaseWeights, urgency: str) -> AdjustedWeights:
    # 1. Validate urgency
    if urgency not in {"high", "normal", "low"}:
        urgency = "normal"
    
    # 2. Get shift for this urgency
    shift = URGENCY_SHIFT[urgency]
    
    # 3. Apply shift
    raw_quality = base_weights.quality + shift["quality"]
    raw_latency = base_weights.latency + shift["latency"]
    raw_cost = base_weights.cost  # Never changed
    
    # 4. Clamp to 0-1 range
    clamped_quality = max(0.0, min(1.0, raw_quality))
    clamped_latency = max(0.0, min(1.0, raw_latency))
    clamped_cost = max(0.0, min(1.0, raw_cost))
    
    # 5. Renormalize
    total = clamped_quality + clamped_latency + clamped_cost
    final_quality = round(clamped_quality / total, 6) if total > 0 else 0.333
    final_latency = round(clamped_latency / total, 6) if total > 0 else 0.333
    final_cost = round(clamped_cost / total, 6) if total > 0 else 0.334
    
    # 6. Fix rounding errors
    diff = round(1.0 - (final_quality + final_latency + final_cost), 6)
    final_quality += diff
    
    return AdjustedWeights(
        quality=final_quality,
        latency=final_latency,
        cost=final_cost,
        urgency_applied=urgency
    )
```

## Summary

**urgency_adjuster.py:**
- ✅ Adjusts weights based on urgency (high/normal/low)
- ✅ Quality and latency are trade-offs
- ✅ Cost weight never changes
- ✅ Weights renormalized to sum to 1.0
- ✅ Different urgencies can select different models for same task
- ✅ Transparent adjustment (base weights from config.json unmodified)

---

# Module 5: scorer.py

## Purpose

**scorer.py** calculates composite scores for each surviving model and ranks them.

**Goal:** Determine which model best fits the use case.

## Scoring Formula

```
base_score = (quality_score × weight_quality)
           + (latency_score × weight_latency)
           + (cost_score × weight_cost)

matched_tags = model.tags ∩ use_case.preferred_tags

if matched_tags:
    tag_bonus = settings.score_tag_bonus  (from config.json)
else:
    tag_bonus = 0.0

final_score = min(1.0, base_score + tag_bonus)
```

**All values from config.json!**

## Components

### Component 1: Base Score

```
base_score = (model.metadata.quality_score × adjusted_weights.quality)
           + (model.metadata.latency_score × adjusted_weights.latency)
           + (model.metadata.cost_score × adjusted_weights.cost)
```

**Values from config.json:**

```json
{
  "models": [{
    "id": "codellama",
    "metadata": {
      "quality_score": 0.88,        // ← Model's quality
      "latency_score": 0.55,        // ← Model's speed
      "cost_score": 1.0             // ← Model's cost
    }
  }]
}
```

**Weights from urgency_adjuster.py (derived from config.json):**
```
quality_weight = 0.60
latency_weight = 0.25
cost_weight = 0.15
```

**Calculation:**
```
base_score = (0.88 × 0.60) + (0.55 × 0.25) + (1.0 × 0.15)
           = 0.528 + 0.1375 + 0.15
           = 0.8155
```

### Component 2: Tag Matching

```python
matched_tags = [t for t in model.metadata.tags 
                if t in use_case.preferred_tags]
```

**Example:**

From config.json:
```json
{
  "models": [{
    "id": "codellama",
    "metadata": {
      "tags": ["code", "reasoning", "long-context"]  // Model's tags
    }
  }],
  "use_cases": {
    "code_generation": {
      "preferred_tags": ["code", "reasoning"]  // Preferred tags
    }
  }
}
```

**Matching:**
```
model.tags: ["code", "reasoning", "long-context"]
preferred: ["code", "reasoning"]

Intersection: ["code", "reasoning"]  ✓ Match!
```

### Component 3: Tag Bonus

```python
from config.json settings:
  score_tag_bonus = 0.05  // 5% bonus for ANY matched tags

if matched_tags:
    bonus = 0.05  // One bonus (not per tag!)
else:
    bonus = 0.0   // No bonus
```

**Important:** Bonus is flat (all-or-nothing), not cumulative per tag!

```
1 matched tag = 0.05 bonus
2 matched tags = 0.05 bonus (same)
3 matched tags = 0.05 bonus (same)

It's: "has matched tags" → bonus, or "no matched tags" → no bonus
```

### Component 4: Final Score

```python
final_score = min(1.0, base_score + bonus)
```

**Why min(1.0, ...)?** Scores can't exceed 1.0 (impossible score).

**Example:**
```
base_score = 0.8155
bonus = 0.05
final = min(1.0, 0.8155 + 0.05)
      = min(1.0, 0.8655)
      = 0.8655  ✓
```

**If extreme bonus (hypothetically):**
```
base_score = 0.95
bonus = 0.10
final = min(1.0, 0.95 + 0.10)
      = min(1.0, 1.05)
      = 1.0  ✓ Capped
```

## Complete Example

### Setup from config.json

**Models:**
```json
{
  "models": [
    {
      "id": "codellama",
      "metadata": {
        "quality_score": 0.88,
        "latency_score": 0.55,
        "cost_score": 1.0,
        "tags": ["code", "reasoning"]
      }
    },
    {
      "id": "llama3",
      "metadata": {
        "quality_score": 0.85,
        "latency_score": 0.60,
        "cost_score": 1.0,
        "tags": ["reasoning", "chat"]
      }
    },
    {
      "id": "phi3-mini",
      "metadata": {
        "quality_score": 0.65,
        "latency_score": 0.95,
        "cost_score": 1.0,
        "tags": ["fastest", "lightweight"]
      }
    }
  ]
}
```

**Use case:**
```json
{
  "use_cases": {
    "code_generation": {
      "preferred_tags": ["code", "reasoning"]
    }
  }
}
```

**Settings:**
```json
{
  "settings": {
    "score_tag_bonus": 0.05
  }
}
```

**Adjusted weights (normal urgency):**
```
quality=0.60, latency=0.25, cost=0.15
```

---

### Scoring Process

**Model 1: codellama**

```
Base score:
  = (0.88 × 0.60) + (0.55 × 0.25) + (1.0 × 0.15)
  = 0.528 + 0.1375 + 0.15
  = 0.8155

Tags:
  model: ["code", "reasoning"]
  preferred: ["code", "reasoning"]
  matched: ["code", "reasoning"]  ✓ Match!

Bonus: 0.05

Final score:
  = min(1.0, 0.8155 + 0.05)
  = 0.8655
```

**Model 2: llama3**

```
Base score:
  = (0.85 × 0.60) + (0.60 × 0.25) + (1.0 × 0.15)
  = 0.51 + 0.15 + 0.15
  = 0.81

Tags:
  model: ["reasoning", "chat"]
  preferred: ["code", "reasoning"]
  matched: ["reasoning"]  ✓ Partial match

Bonus: 0.05

Final score:
  = min(1.0, 0.81 + 0.05)
  = 0.86
```

**Model 3: phi3-mini**

```
Base score:
  = (0.65 × 0.60) + (0.95 × 0.25) + (1.0 × 0.15)
  = 0.39 + 0.2375 + 0.15
  = 0.7775

Tags:
  model: ["fastest", "lightweight"]
  preferred: ["code", "reasoning"]
  matched: []  ❌ No match

Bonus: 0.0

Final score:
  = min(1.0, 0.7775 + 0.0)
  = 0.7775
```

---

### Sorted Ranking

```
1. codellama:    0.8655  ← WINNER 🏆
2. llama3:       0.86
3. phi3-mini:    0.7775
```

**Console output:**
```
[scorer] Scores for use-case 'code_generation' (urgency='normal'):
  codellama: base=0.8155 +tag_bonus(['code', 'reasoning']) → final=0.8655
  llama3: base=0.8100 +tag_bonus(['reasoning']) → final=0.86
  phi3-mini: base=0.7775 → final=0.7775
```

## ScoredModel Dataclass

```python
@dataclass
class ScoredModel:
    model_id: str              # From model.id
    model: ModelConfig         # Full model config
    base_score: float          # Before tag bonus
    tag_bonus_applied: bool    # Was bonus applied?
    matched_tags: List[str]    # Which tags matched
    final_score: float         # After tag bonus
    
    def summary(self) -> str:
        tag_note = ""
        if self.tag_bonus_applied:
            tag_note = f" +tag_bonus({self.matched_tags})"
        return f"{self.model_id}: base={self.base_score:.4f}{tag_note} → final={self.final_score:.4f}"
```

**Example:**
```python
s = ScoredModel(
    model_id="codellama",
    model=ModelConfig(...),
    base_score=0.8155,
    tag_bonus_applied=True,
    matched_tags=["code", "reasoning"],
    final_score=0.8655
)

print(s.summary())
# Output: codellama: base=0.8155 +tag_bonus(['code', 'reasoning']) → final=0.8655
```

## Minimum Score Threshold

```python
if final_score < settings.minimum_composite_score:
    print(f"WARNING: {model_id} final_score={final_score:.4f} "
          f"is below minimum_composite_score={settings.minimum_composite_score}")
```

**From config.json:**
```json
{
  "settings": {
    "minimum_composite_score": 0.50
  }
}
```

**This is a WARNING, not a filter!** Models below threshold still included.

**Use case:** Alert when recommended model is borderline poor quality (all models filtered but fallback selected).

## Summary

**scorer.py:**
- ✅ Calculates composite score for each model
- ✅ Uses quality, latency, cost scores from config.json models
- ✅ Uses weights from urgency_adjuster (derived from config.json)
- ✅ Applies tag bonus if preferred tags match
- ✅ Ranks models by final score (highest first)
- ✅ Returns sorted list of ScoredModel objects
- ✅ Warns if score below minimum threshold from config.json

---

# Module 6: recommender.py

## Purpose

**recommender.py** is the **main orchestrator** that coordinates all previous modules to produce a final model recommendation.

It's called `recommend()` and returns `RecommendationResult` with:
- Winner model
- Explanation
- Score
- All eliminated models
- Fallback info
- Logging

## Complete Pipeline

```
User request:
  {message, hints}
         ↓
recommend(message, hints, config)
         ↓
1. Determine Use Case
   ├─ From hints OR auto-detect via phi3-mini
   └─ Validate exists in config.json
         ↓
2. Load and Validate config.json
   ├─ config_loader.py
   └─ RouterConfig ready
         ↓
3. Check Health
   ├─ health_checker.py
   ├─ Ping all models (concurrent)
   └─ Dict[model_id: bool]
         ↓
4. Apply Filters
   ├─ filter_engine.py
   ├─ Health → Quality → Context → Cost
   └─ List[survivors], Dict[eliminated]
         ↓
5. Check Survivors
   ├─ if survivors exist:
   │    Proceed to scoring
   │
   └─ else (all eliminated):
       Activate fallback from config.json
         ↓
6. Adjust Weights by Urgency
   ├─ urgency_adjuster.py
   ├─ From hints or default "normal"
   └─ AdjustedWeights
         ↓
7. Score Models
   ├─ scorer.py
   ├─ Calculate composite score
   ├─ Apply tag bonus
   └─ List[ScoredModel] sorted
         ↓
8. Select Winner
   ├─ scored[0] (highest score)
   └─ ScoredModel
         ↓
9. Build Explanation
   ├─ Why was this model selected?
   ├─ What were runners-up?
   ├─ What was eliminated and why?
   └─ Human-readable string
         ↓
10. Log Feedback
    ├─ To feedback_log_path from config.json
    ├─ JSON with all details
    └─ For analysis and audit trail
         ↓
11. Return RecommendationResult
    ├─ winner_model_id
    ├─ winner_model
    ├─ winner_score
    ├─ use_case_name
    ├─ urgency
    ├─ fallback_used
    ├─ fallback_group
    ├─ all_scores
    ├─ eliminated
    ├─ explanation
    ├─ hints_applied
    └─ use_case_auto_detected
```

## Step 1: Determine Use Case

```python
use_case_name = hints.get("use_case")

if use_case_name:
    # Explicitly provided
    if use_case_name not in config.use_cases:  # ← Validate against config.json
        raise ValueError(
            f"hints['use_case'] = '{use_case_name}' is not defined in config. "
            f"Available: {list(config.use_cases.keys())}"
        )
    print(f"[recommender] Use-case from hints: '{use_case_name}'")
    auto_detected = False

else:
    # Auto-detect using LLM
    print("[recommender] Auto-detecting use-case via phi3-mini...")
    
    uc_names = list(config.use_cases.keys())  # ← From config.json
    uc_list = ", ".join(uc_names)
    
    prompt = (
        f"Classify into exactly ONE: {uc_list}\n\n"
        f"Message: \"{message}\"\n\n"
        f"Reply with only the use-case name."
    )
    
    detected = _auto_detect_via_phi3(prompt)
    use_case_name = detected if detected in config.use_cases else "chat"
    auto_detected = True
```

**From your config.json:**
```json
{
  "use_cases": {
    "summarization": {...},
    "reasoning": {...},
    "code_generation": {...},
    "routing_decision": {...},
    "rag_answer": {...},
    "chat": {...},
    "data_extraction": {...},
    "long_context": {...}
  }
}
```

## Step 2: Get Use Case Config

```python
use_case: UseCaseConfig = config.use_cases[use_case_name]
```

Extracts full configuration:
```python
UseCaseConfig(
    name="code_generation",
    description="Write, debug, review, or explain code",
    weights=UseCaseWeights(quality=0.60, latency=0.25, cost=0.15),
    minimum_requirements=MinimumRequirements(quality_score=0.75, context_length=4096),
    preferred_tags=["code", "reasoning"],
    fallback_group="code"
)
```

## Step 3: Check Health

```python
all_models = list(config.models.values())  # All 9 models

health_status = check_health(all_models, config.settings)
# Returns: Dict[model_id: bool]
```

## Step 4: Apply Filters

```python
filter_result: FilterResult = apply_filters(
    all_models=all_models,
    use_case=use_case,
    health_status=health_status,
    hints=hints
)
```

Returns:
```python
FilterResult(
    survivors=[...],        # Models that passed all filters
    eliminated={...}        # Models removed + reasons
)
```

## Step 5: Check if Survivors Exist

```python
fallback_used = False

if filter_result.has_survivors:
    # Normal path: score the survivors
    adjusted_weights = adjust_weights(use_case.weights, urgency)
    all_scores = score_models(
        survivors=filter_result.survivors,
        use_case=use_case,
        adjusted_weights=adjusted_weights,
        settings=config.settings
    )

else:
    # Fallback path: all models eliminated!
    fallback_used = True
    fallback_group = use_case.fallback_group  # ← From config.json
    
    print(f"[recommender] Activating fallback group '{fallback_group}'")
    
    fallback_cfg = config.groups[fallback_group]
    fallback_models = [
        config.models[mid]
        for mid in fallback_cfg.models
        if config.models[mid] and health_status.get(mid, False)
    ]
    
    if not fallback_models:
        raise RuntimeError(f"All models failed including fallback group '{fallback_group}'")
    
    adjusted_weights = adjust_weights(use_case.weights, urgency)
    all_scores = score_models(
        survivors=fallback_models,
        use_case=use_case,
        adjusted_weights=adjusted_weights,
        settings=config.settings
    )
```

## Step 6: Select Winner

```python
winner: ScoredModel = all_scores[0]  # Highest score
```

## Step 7: Build Explanation

```python
explanation = _build_explanation(
    winner=winner,
    use_case_name=use_case_name,
    urgency=urgency,
    eliminated=filter_result.eliminated,
    fallback_used=fallback_used,
    fallback_group=fallback_group,
    all_scores=all_scores
)
```

**Example:**
```
✓ Selected: codellama (score=0.8655) for use-case='code_generation', urgency='normal'
  Runner-up: llama3 (score=0.86)
  Eliminated:
    ✗ phi3-mini: quality_score 0.65 below minimum 0.75
    ✗ gemma2-2b: quality_score 0.60 below minimum 0.75
    ✗ llama3.2-3b: quality_score 0.68 below minimum 0.75
```

## Step 8: Log Feedback

```python
_log_feedback(result, config, start_time)
```

**To feedback_log_path from config.json:**
```json
{
  "timestamp": "2026-03-26T14:32:15.123456",
  "use_case": "code_generation",
  "urgency": "normal",
  "winner": "codellama",
  "winner_score": 0.8655,
  "fallback_used": false,
  "all_scores": [
    {"model_id": "codellama", "score": 0.8655},
    {"model_id": "llama3", "score": 0.86},
    ...
  ],
  "eliminated": {...},
  "recommendation_time_ms": 245.3
}
```

## Complete Example

### Request

```python
message = "Write a quicksort function in Python"
hints = {
    "use_case": "code_generation",
    "urgency": "normal"
}

result = recommend(message, hints, config)
```

### Execution

**Step 1: Use case**
```
✓ code_generation from hints
```

**Step 2: Health check**
```
[health_checker] Pinging 9 models...
[health_checker] Done in 1.2s — 7/9 models healthy
```

**Step 3: Filter**
```
[filter_engine] Starting with 9 candidates
[filter_engine] After health filter: 7 remaining
[filter_engine] After quality filter (min=0.75): 4 remaining
[filter_engine] After context filter (min=4096): 4 remaining
[filter_engine] After cost filter: 4 remaining
```

**Step 4: Adjust weights**
```
[urgency_adjuster] Base: quality=0.60, latency=0.25, cost=0.15
[urgency_adjuster] Adjusted: quality=0.60, latency=0.25, cost=0.15 (normal)
```

**Step 5: Score**
```
[scorer] Scores for code_generation:
  codellama: base=0.8155 +tag_bonus(['code']) → final=0.8655
  llama3: base=0.8100 +tag_bonus(['reasoning']) → final=0.8600
  gemini-flash: base=0.7975 → final=0.7975
  mistral: base=0.7875 → final=0.7875
```

**Step 6: Winner**
```
Winner: codellama (score=0.8655)
```

**Step 7: Explanation**
```
✓ Selected: codellama (score=0.8655) for use-case='code_generation', urgency='normal'
  Runner-up: llama3 (score=0.86)
  Eliminated:
    ✗ phi3-mini: quality_score 0.65 below minimum 0.75
    ✗ gemma2-2b: quality_score 0.60 below minimum 0.75
    ✗ llama3.2-3b: quality_score 0.68 below minimum 0.75
    ✗ llama3: health check failed (offline)
    ✗ gpt-oss: health check failed (offline)
```

**Step 8: Log**
```
[recommender] Feedback logged to logs/feedback.jsonl
```

**Step 9: Return**
```python
RecommendationResult(
    winner_model_id="codellama",
    winner_model=ModelConfig(...),
    winner_score=0.8655,
    use_case_name="code_generation",
    urgency="normal",
    fallback_used=False,
    fallback_group=None,
    all_scores=[...],
    eliminated={...},
    explanation="✓ Selected: codellama...",
    hints_applied={"use_case": "code_generation", "urgency": "normal"},
    use_case_auto_detected=False
)
```

## RecommendationResult Dataclass

```python
@dataclass
class RecommendationResult:
    winner_model_id: str              # Winning model's ID
    winner_model: ModelConfig         # Full model config
    winner_score: float               # Final composite score
    use_case_name: str                # Selected use case
    urgency: str                      # "high", "normal", "low"
    fallback_used: bool               # Was fallback activated?
    fallback_group: Optional[str]     # Which fallback group?
    all_scores: List[ScoredModel]     # All scored models
    eliminated: Dict[str, str]        # Models removed + reasons
    explanation: str                  # Human-readable summary
    hints_applied: Dict = field(default_factory=dict)
    use_case_auto_detected: bool = False
```

## Fallback Scenario

**When all models fail filtering:**

```python
[filter_engine] Starting with 9 candidates for 'long_context'
[filter_engine] After quality filter (min=0.95): 0 remaining  ← ALL ELIMINATED!

[recommender] Activating fallback group 'long_context'
[recommender] Fallback models: ['gemini-flash', 'codellama', 'gpt-oss']

[scorer] Scores for long_context (fallback):
  gemini-flash: base=0.8175 → final=0.8175
  codellama: base=0.7975 → final=0.7975
  gpt-oss: base=0.7275 → final=0.7275

✓ Selected: gemini-flash (score=0.8175) for use-case='long_context', urgency='normal'
⚠ Fell back to group 'long_context' (all models failed primary filters)
```

## Summary

**recommender.py:**
- ✅ Main orchestrator coordinating all modules
- ✅ Determines use case (from hints or auto-detect)
- ✅ Runs complete pipeline: health → filter → score
- ✅ Handles fallback when all models eliminated
- ✅ Returns detailed RecommendationResult
- ✅ Logs to feedback file for analytics
- ✅ Provides clear explanation for debugging

---

# Module 7: wrapped_client.py

## Purpose

**wrapped_client.py** is the **public-facing API** that developers import and use.

It wraps all the complexity of the model recommendation engine behind a simple `.invoke()` method.

## Three Operating Modes

### Mode 1: MODEL_ONLY 🚀 (Direct invocation)

```python
client = WrappedLangchainClient.from_model(
    model_name="codellama:34b"
    # config_path=None (no config!)
)

response = client.invoke("Write code")
```

**What happens:**
- ❌ No config.json loaded
- ✅ Model invoked directly
- ❌ NO fallback
- ⚡ Fastest (skip recommender overhead)

**Use case:** Developer knows exactly which model they want

---

### Mode 2: CONFIG_ONLY 🧠 (Full recommendation engine)

```python
client = WrappedLangchainClient.auto("config.json")

response = client.invoke(
    "Write code",
    hints={"use_case": "code_generation"}
)
```

**What happens:**
- ✅ Config.json fully loaded
- ✅ Recommendation engine runs (health → filter → score)
- ✅ Best model selected
- ✅ Fallback enabled
- 🐢 Slower but most intelligent

**Uses all of config.json**

---

### Mode 3: MODEL_CONFIG 🎯 (Try model first, then fallback)

```python
client = WrappedLangchainClient.from_model(
    model_name="codellama:34b",
    config_path="config.json"  # ← Enables fallback
)

response = client.invoke("Write code")
```

**What happens:**
1. Try specified model first
2. If fails, activate recommendation engine
3. Fallback via fallback_group from config.json

**Use case:** "Use my preferred model, but intelligently pick alternative if needed"

---

## Factory Methods

### Method: `.auto(config_path)`

```python
@classmethod
def auto(cls, config_path: str) -> "WrappedLangchainClient":
    resolved = _resolve_config_path(config_path)
    config = load_config(resolved)  # ← config_loader.py
    print(f"[wrapped_client] Mode: config-only (recommendation engine active)")
    return cls(config=config, mode=_Mode.CONFIG_ONLY)
```

**Returns CONFIG_ONLY mode client**

---

### Method: `.from_model(model_name, config_path)`

```python
@classmethod
def from_model(
    cls,
    model_name: str,
    config_path: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500
) -> "WrappedLangchainClient":
    if config_path:
        # MODEL_CONFIG mode (try model first, then fallback)
        config = load_config(_resolve_config_path(config_path))
        mode = _Mode.MODEL_CONFIG
        print(f"[wrapped_client] Mode: model+config (primary='{model_name}', fallback)")
    else:
        # MODEL_ONLY mode (direct invocation)
        config = None
        mode = _Mode.MODEL_ONLY
        print(f"[wrapped_client] Mode: model-only ('{model_name}')")
    
    return cls(
        config=config,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        mode=mode
    )
```

---

## Invoke Method

```python
def invoke(self, message: Any, hints: Optional[Dict] = None) -> AIMessage:
    hints = hints or {}
    messages = _to_messages(message)
    
    if self._mode == _Mode.MODEL_ONLY:
        return self._invoke_model_only(messages)
    
    if self._mode == _Mode.CONFIG_ONLY:
        return self._invoke_config_only(messages, hints)
    
    if self._mode == _Mode.MODEL_CONFIG:
        return self._invoke_model_config(messages, hints)
```

---

### Handler: MODEL_ONLY

```python
def _invoke_model_only(self, messages: List[BaseMessage]) -> AIMessage:
    print(f"[wrapped_client] Direct invoke: {self._model_name}")
    return _invoke_direct(messages, self._model_name, ...)
```

**Direct LLM call, no recommender**

---

### Handler: CONFIG_ONLY

```python
def _invoke_config_only(self, messages, hints) -> AIMessage:
    # 1. Call recommender with config.json
    recommendation = recommend(
        message=_to_raw_str(messages),
        hints=hints,              # ← User hints
        config=self._config       # ← Entire config.json
    )
    self._last_recommendation = recommendation
    
    # 2. Get fallback_group from config.json
    fallback_group = recommendation.fallback_group or \
                     self._config.use_cases[recommendation.use_case_name].fallback_group
    
    # 3. Invoke with fallback support
    return _invoke_with_fallback(
        messages=messages,
        winner_model=recommendation.winner_model,
        fallback_group=self._config.groups.get(fallback_group),
        config=self._config
    )
```

**Full recommendation engine + fallback**

---

### Handler: MODEL_CONFIG

```python
def _invoke_model_config(self, messages, hints) -> AIMessage:
    try:
        # Try specified model first
        print(f"[wrapped_client] Trying named model: '{self._model_name}'")
        result = _invoke_direct(messages, self._model_name, ...)
        print(f"[wrapped_client] Named model succeeded: '{self._model_name}'")
        return result
    
    except Exception as e:
        # Model failed, activate recommender
        print(f"[wrapped_client] Named model failed: {e}")
        print(f"[wrapped_client] Falling back to recommendation engine...")
        return self._invoke_config_only(messages, hints)
```

**Try model first, then CONFIG_ONLY fallback**

---

## Fallback Logic

```python
def _invoke_with_fallback(messages, winner_model, fallback_group, config):
    # 1. Try winner
    try:
        print(f"[wrapped_client] Invoking: {winner_model.id}")
        return _build_and_invoke(winner_model, messages)
    except Exception as e:
        print(f"[wrapped_client] '{winner_model.id}' failed: {e}")
    
    # 2. Check fallback_group
    if fallback_group is None:
        raise RuntimeError(f"Model '{winner_model.id}' failed, no fallback configured")
    
    # 3. Try fallback models (priority or round-robin strategy)
    strategy = _build_strategy(fallback_group)  # ← From config.json routing_strategy
    
    for model_id in strategy.get_order():
        if model_id == winner_model.id:
            continue  # Already tried
        
        try:
            print(f"[wrapped_client] Fallback → {model_id}")
            return _build_and_invoke(config.models[model_id], messages)
        except Exception as e:
            print(f"[wrapped_client] Fallback '{model_id}' failed: {e}")
    
    # 4. All failed
    raise RuntimeError(
        f"All models failed — winner '{winner_model.id}' "
        f"+ fallback group '{fallback_group.name}'"
    )
```

---

## Routing Strategies

### Strategy 1: PRIORITY (Sequential)

**From config.json:**
```json
{
  "groups": {
    "code": {
      "models": ["codellama", "llama3", "gpt-oss"],
      "routing_strategy": "priority"
    }
  }
}
```

**Behavior:**
```
Try: codellama (1st)
  ✓ Success → return
  ✗ Failed → try next

Try: llama3 (2nd)
  ✓ Success → return
  ✗ Failed → try next

Try: gpt-oss (3rd)
  ✓ Success → return
  ✗ Failed → error
```

---

### Strategy 2: ROUND-ROBIN (Load-balanced)

**From config.json:**
```json
{
  "groups": {
    "default": {
      "models": ["llama3", "mistral", "gemma2"],
      "routing_strategy": "round-robin"
    }
  }
}
```

**Behavior (rotates each call):**
```
Call 1: [llama3, mistral, gemma2]
Call 2: [mistral, gemma2, llama3]
Call 3: [gemma2, llama3, mistral]
Call 4: [llama3, mistral, gemma2]  (wraps)
```

**Why?** Distribute load across multiple models

---

## Example Usage Scenarios

### Scenario 1: Developer knows exact model

```python
client = WrappedLangchainClient.from_model("codellama:34b")

response = client.invoke("Write quicksort")
# Direct to codellama, no recommender, fast
```

---

### Scenario 2: Developer wants intelligent routing

```python
client = WrappedLangchainClient.auto("config.json")

response = client.invoke(
    "Write quicksort",
    hints={"use_case": "code_generation"}
)
# Uses recommender to select best model
```

---

### Scenario 3: Developer wants primary model with fallback

```python
client = WrappedLangchainClient.from_model(
    "codellama:34b",
    config_path="config.json"
)

response = client.invoke("Write quicksort")
# Try codellama first
# If fails, activate recommender + fallback_group
```

---

## Helper Methods

```python
# What use cases are available?
client.available_use_cases()
# ['summarization', 'reasoning', 'code_generation', ...]

# What models are available?
client.available_models()
# ['phi3-mini', 'codellama', 'llama3', ...]

# Explain last recommendation
client.explain_last()
# "✓ Selected: codellama (score=0.8655) for..."

# Get last recommendation details
client.last_recommendation
# RecommendationResult(...)

# What mode is client in?
client.mode
# "config_only", "model_only", or "model_config"
```

---

## Summary

**wrapped_client.py:**
- ✅ Simple public API (`.auto()` or `.from_model()`)
- ✅ Three modes: MODEL_ONLY, CONFIG_ONLY, MODEL_CONFIG
- ✅ Single method: `.invoke(message, hints)`
- ✅ Automatic fallback when model fails
- ✅ Respects routing_strategy from config.json (priority/round-robin)
- ✅ Helper methods for discovery and debugging
- ✅ Thread-safe for concurrent requests

---

# Complete Pipeline Flow

## End-to-End Request Flow

```
┌─────────────────────────────────────────┐
│ User (Flask UI or Python code)          │
│ Sends request with message + hints      │
└──────────────────────┬──────────────────┘
                       ↓
┌─────────────────────────────────────────┐
│ wrapped_client.py                       │
│ .invoke(message, hints)                 │
│ Mode: CONFIG_ONLY or MODEL_CONFIG       │
└──────────────────────┬──────────────────┘
                       ↓
┌─────────────────────────────────────────┐
│ recommender.py                          │
│ recommend(message, hints, config)       │
└──────────────────────┬──────────────────┘
                       ├─ Uses config.json
                       ├─ Validates use_case
                       └─ Calls all modules:
                       │
           ┌───────────┼───────────┐
           │           │           │
           ↓           ↓           ↓
┌──────────────────┐  │  ┌──────────────────┐
│ health_checker   │  │  │  filter_engine   │
│ Check if models  │  │  │ Quality/Context  │
│ are online       │  │  │ Cost requirements│
└────────┬─────────┘  │  └────────┬─────────┘
         │            │          │
         └────────────┼──────────┘
                      ↓
┌──────────────────────────────────────────┐
│ urgency_adjuster.py                      │
│ Adjust weights based on urgency          │
└──────────────────────┬───────────────────┘
                       ↓
┌──────────────────────────────────────────┐
│ scorer.py                                │
│ Calculate composite scores               │
│ Apply tag bonuses                        │
│ Sort by final score                      │
└──────────────────────┬───────────────────┘
                       ↓
┌──────────────────────────────────────────┐
│ Select Winner (highest score)            │
└──────────────────────┬───────────────────┘
                       ↓
┌──────────────────────────────────────────┐
│ recommender.py                           │
│ Build explanation                        │
│ Log feedback to JSON file                │
│ Return RecommendationResult              │
└──────────────────────┬───────────────────┘
                       ↓
┌──────────────────────────────────────────┐
│ wrapped_client.py                        │
│ Invoke winner model (or fallback)        │
│ Get LLM response                         │
└──────────────────────┬───────────────────┘
                       ↓
┌─────────────────────────────────────────┐
│ Return AIMessage to caller               │
│ With model's response                    │
└─────────────────────────────────────────┘
```

## Data Passing Between Modules

```
config.json (input)
    ↓
config_loader.py
    ↓ RouterConfig(models, groups, use_cases, settings)
    │
    ├─ recommender.py
    │   ├─ health_checker.py
    │   │   ↓ Dict[model_id: bool]
    │   │
    │   ├─ filter_engine.py
    │   │   ├─ health_status ← health_checker
    │   │   ↓ FilterResult(survivors, eliminated)
    │   │
    │   ├─ urgency_adjuster.py
    │   │   ├─ base_weights ← use_case.weights
    │   │   ↓ AdjustedWeights
    │   │
    │   ├─ scorer.py
    │   │   ├─ survivors ← filter_engine
    │   │   ├─ adjusted_weights ← urgency_adjuster
    │   │   ↓ List[ScoredModel]
    │   │
    │   ├─ Select winner
    │   ├─ Build explanation
    │   ├─ Log feedback → logs/feedback.jsonl
    │   ↓ RecommendationResult
    │
    └─ wrapped_client.py
        ├─ Get winner_model
        ├─ Invoke LLM
        └─ Return response
```

---

# Example Workflows

## Workflow 1: Simple Code Generation

**User:** "Write a Python function for binary search"

**Flow:**

1. **wrapped_client.auto("config.json")**
   - Load config.json
   - CONFIG_ONLY mode

2. **recommend(message, hints, config)**
   - Use case: auto-detect "code_generation"
   - Health check: 9 models, 7 online
   - Filter: quality 0.75 → 4 survivors
   - Context 4096 → 4 survivors
   - Adjust weights: normal urgency → 0.60, 0.25, 0.15
   - Score: codellama=0.8655, llama3=0.86, gemini-flash=0.7975
   - Winner: codellama (0.8655)

3. **wrapped_client.invoke**
   - Try codellama
   - Success ✓
   - Return response

4. **Log to feedback.jsonl**
   ```json
   {
     "timestamp": "2026-03-26T...",
     "use_case": "code_generation",
     "winner": "codellama",
     "winner_score": 0.8655,
     "recommendation_time_ms": 245.3
   }
   ```

---

## Workflow 2: High-Urgency Summarization

**User:** "Summarize NOW!" + urgency="high"

**Flow:**

1. **wrapped_client.auto("config.json")**
   - CONFIG_ONLY mode

2. **recommend(message, hints={"urgency": "high"}, config)**
   - Use case: auto-detect "summarization"
   - Health check: 7 online
   - Filter: quality 0.6 → 6 survivors
   - Context 2048 → 6 survivors
   - **Adjust weights (HIGH urgency): quality=0.10, latency=0.65, cost=0.25**
   - Score (latency boosted): phi3-mini=0.76, gemma2-2b=0.77, llama3.2-3b=0.72
   - Winner: gemma2-2b (0.77) - fastest!

3. **invoke**
   - Try gemma2-2b (fast)
   - Success ✓
   - Return response

**Key difference:** HIGH urgency selected fastest model (gemma2-2b) instead of highest quality (gemma2)

---

## Workflow 3: Large Document with Fallback

**User:** "Process my 50KB document" + use_case="long_context"

**Flow:**

1. **wrapped_client.from_model("codellama:34b", config_path="config.json")**
   - MODEL_CONFIG mode (try model first)

2. **invoke(message)**
   - Try codellama:34b
   - ❌ Server down (connection refused)
   - Fallback to recommender

3. **recommend (fallback)**
   - Use case: long_context
   - minimum_requirements: context_length=32768
   - Health check: gemini-flash offline
   - Filter: context ≥ 32768 → gemini-flash(128K), codellama(16K), gpt-oss(16K)
   - ❌ Only gemini-flash survives but offline
   - **Activate fallback_group "long_context"**
   - Models: gemini-flash (offline), codellama, gpt-oss
   - Score: codellama=0.81, gpt-oss=0.79
   - Winner: codellama

4. **invoke with fallback**
   - Try: codellama ✓ Success
   - Return response

5. **Log**
   ```json
   {
     "fallback_used": true,
     "fallback_group": "long_context",
     "winner": "codellama"
   }
   ```

---

## Workflow 4: All Models Fail

**Scenario:** Ollama server crashed, all models offline

**Flow:**

1. **health_checker**
   - Ping all 9 models
   - All timeout ❌
   - health_status: all False

2. **filter_engine**
   - Health filter: all eliminated
   - FilterResult.has_survivors = False
   - **Activate fallback_group**

3. **fallback_group "code"**
   - Models: codellama, llama3, gpt-oss
   - Health check: all still offline ❌
   - fallback_models = []

4. **Error**
   ```
   RuntimeError: All models failed — including all models in fallback group 'code'.
   Check that Ollama is running.
   ```

**Fix:** Restart Ollama server or reconfigure fallback groups

---

## Workflow 5: Cost-Conscious User

**User:** "I want only free models" + prefer_cost="free"

**Flow:**

1. **filter_engine**
   - Cost filter: prefer_cost="free"
   - Remove all models where cost_score < 1.0
   - From your config: all models have cost_score=1.0 ✓
   - All pass

2. **Score and select as normal**

**Note:** In your current config.json, all models are free (cost_score=1.0), so cost filter doesn't eliminate anything. If you added paid models (e.g., OpenAI with cost_score=0.5), this would filter them out.

---

# Summary of Complete Architecture

## All Modules at a Glance

| Module | Input | Process | Output |
|--------|-------|---------|--------|
| **config_loader** | config.json | Parse JSON, validate, convert to dataclasses | RouterConfig |
| **health_checker** | RouterConfig | Ping all models concurrently | Dict[model_id: bool] |
| **filter_engine** | RouterConfig + health_status | Apply 4 filters (health, quality, context, cost) | FilterResult (survivors, eliminated) |
| **urgency_adjuster** | weights + urgency | Adjust quality/latency by urgency | AdjustedWeights |
| **scorer** | survivors + adjusted_weights | Calculate composite score, apply tag bonus, sort | List[ScoredModel] |
| **recommender** | message + hints + RouterConfig | Orchestrate all modules | RecommendationResult |
| **wrapped_client** | message + hints | Call recommender or direct invoke, handle fallback | AIMessage response |

## Config.json Usage by Module

```
config.json
├─ models[]
│  ├─ config_loader: Parse + validate
│  ├─ health_checker: model_name for pinging
│  ├─ filter_engine: metadata (quality, latency, cost, context_length)
│  ├─ scorer: metadata.tags for bonus matching
│  └─ wrapped_client: Select and invoke
│
├─ groups{}
│  ├─ recommender: For fallback_group lookup
│  └─ wrapped_client: routing_strategy (priority/round-robin)
│
├─ use_cases{}
│  ├─ recommender: weights, minimum_requirements, preferred_tags
│  ├─ filter_engine: minimum_requirements for filtering
│  ├─ scorer: preferred_tags for bonus matching
│  └─ urgency_adjuster: base weights
│
└─ settings
   ├─ health_checker: health_check_enabled, health_check_timeout_seconds
   ├─ filter_engine: (uses use_case requirements)
   ├─ scorer: score_tag_bonus, minimum_composite_score
   └─ recommender: feedback_logging_enabled, feedback_log_path
```

---

## Key Validations

**From config_loader.py:**

✓ Scores must be 0.0-1.0  
✓ Context length must be > 0  
✓ Weights must sum to 1.0  
✓ Routing strategy must be valid  
✓ Cross-references must exist (models in groups, groups in use_cases)  
✓ All required fields present  

**From filter_engine.py:**

✓ Quality score ≥ minimum_requirements.quality_score  
✓ Context length ≥ maximum of (use_case minimum + user hint)  
✓ Cost score = 1.0 if prefer_cost="free"  

**From scorer.py:**

✓ Final score ≤ 1.0 (capped)  
✓ Final score ≥ minimum_composite_score (warning only)  

---

## Performance Characteristics

```
Module                      Time        Bottleneck
────────────────────────────────────────────────────
config_loader.py           ~50ms       File I/O
health_checker.py          ~1-3s       Network (concurrent)
filter_engine.py           ~10ms       Fast filtering
urgency_adjuster.py        ~1ms        Simple math
scorer.py                  ~20ms       Per-model scoring
recommender.py (total)     ~200-300ms  health_checker + scorer
wrapped_client.invoke()    ~1-5s       LLM response time
────────────────────────────────────────────────────
Total per request          ~1-5s       LLM is bottleneck
```

---

**End of Documentation**

This comprehensive guide explains all modules with respect to your config.json. Share with your team! 🚀







