"""
Module 2 — health_checker.py
Pings each Ollama model concurrently to check availability.
Returns a simple {model_id: bool} map before routing begins.
"""

import asyncio
import time
from typing import Dict, List

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .config_loader import ModelConfig, RecommendationSettings

OLLAMA_BASE_URL = "http://localhost:11434"

async def _ping_model(
    model: ModelConfig,
    timeout: int,
    client: "httpx.AsyncClient"
) -> tuple[str, bool]:
    try:
        response = await client.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=timeout
        )
        if response.status_code != 200:
            print(f"[health_checker] {model.id}: Ollama responded {response.status_code}")
            return model.id, False

        data = response.json()
        available_names = [m.get("name", "") for m in data.get("models", [])]

        # Match by model_name prefix (e.g. "llama3" matches "llama3:latest")
        is_available = any(
            name == model.model_name or name.startswith(model.model_name + ":")
            for name in available_names
        )

        if not is_available:
            print(
                f"[health_checker] {model.id} ('{model.model_name}'): "
                f"not found in Ollama. Available: {available_names}"
            )

        return (model.id, is_available)

    except Exception as e:
        print(f"[health_checker] {model.id}: ping failed — {type(e).__name__}: {e}")
        return (model.id, False)


async def _check_all_async(
    models: List[ModelConfig],
    timeout: int
) -> Dict[str, bool]:
    if not HTTPX_AVAILABLE:
        raise ImportError("httpx is required for health checks. Install it with: pip install httpx")

    async with httpx.AsyncClient() as client:
        tasks = [_ping_model(model, timeout, client) for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    health_map: Dict[str, bool] = {}
    for result in results:
        if isinstance(result, Exception):
            print(f"[health_checker] Unexpected error during ping: {result}")
            continue
        model_id, is_healthy = result
        health_map[model_id] = is_healthy

    return health_map


def check_health(
    models: List[ModelConfig],
    settings: RecommendationSettings
) -> Dict[str, bool]:
    if not settings.health_check_enabled:
        print("[health_checker] Health checks disabled — marking all models healthy")
        return {model.id: True for model in models}

    print(f"[health_checker] Pinging {len(models)} models (timeout={settings.health_check_timeout_seconds}s)...")
    start = time.time()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    _check_all_async(models, settings.health_check_timeout_seconds)
                )
                health_map = future.result()
        else:
            health_map = loop.run_until_complete(
                _check_all_async(models, settings.health_check_timeout_seconds)
            )
    except RuntimeError:
        health_map = asyncio.run(
            _check_all_async(models, settings.health_check_timeout_seconds)
        )

    elapsed = round(time.time() - start, 2)
    healthy_count = sum(1 for v in health_map.values() if v)

    print(
        f"[health_checker] Done in {elapsed}s — "
        f"{healthy_count}/{len(models)} models healthy"
    )

    for model_id, is_healthy in health_map.items():
        status = "✓ online" if is_healthy else "✗ offline"
        print(f"  {status}  {model_id}")

    return health_map
