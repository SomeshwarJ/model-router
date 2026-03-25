"""
Module 8 — langgraph_integration.py
Makes WrappedLangchainClient native inside LangGraph.
Provides decorator, base class, and manual helper.
"""

import functools
from typing import Any, Callable, Dict, Optional

from .config_loader import load_config
from .wrapped_client import WrappedLangchainClient

_DEFAULT_CONFIG_PATH = "config.json"

def recommend_model(
    use_case: str,
    urgency: str = "normal",
    input_token_estimate: int = 0,
    prefer_cost: Optional[str] = None,
    config_path: str = _DEFAULT_CONFIG_PATH
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: Dict, *args, **kwargs):
            hints = {"use_case": use_case, "urgency": urgency}
            if input_token_estimate > 0:
                hints["input_token_estimate"] = input_token_estimate
            if prefer_cost:
                hints["prefer_cost"] = prefer_cost

            client = WrappedLangchainClient.auto(config_path)
            return func(state, client, *args, **kwargs)
        return wrapper
    return decorator

def use_model(
    model_name: str,
    config_path: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: Dict, *args, **kwargs):
            client = WrappedLangchainClient.from_model(
                model_name=model_name,
                config_path=config_path,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return func(state, client, *args, **kwargs)
        return wrapper
    return decorator


class RouterNode:

    use_case:             str           = "chat"
    urgency:              str           = "normal"
    input_token_estimate: int           = 0
    prefer_cost:          Optional[str] = None

    model_name:   Optional[str] = None
    temperature:  float         = 0.7
    max_tokens:   int           = 500

    config_path: Optional[str] = _DEFAULT_CONFIG_PATH

    def __init__(self):
        self._hints: Dict[str, Any] = {
            "use_case": self.use_case,
            "urgency":  self.urgency,
        }
        if self.input_token_estimate > 0:
            self._hints["input_token_estimate"] = self.input_token_estimate
        if self.prefer_cost:
            self._hints["prefer_cost"] = self.prefer_cost

        if self.model_name:
            # Mode B or C — model name is set
            self.client = WrappedLangchainClient.from_model(
                model_name=self.model_name,
                config_path=self.config_path,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        else:
            if not self.config_path:
                raise ValueError(
                    f"{self.__class__.__name__}: must set either "
                    f"'model_name' or 'config_path'"
                )
            self.client = WrappedLangchainClient.auto(self.config_path)

    def __call__(self, state: Dict) -> Dict:
        return self.run(state)

    def run(self, state: Dict) -> Dict:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run(self, state) -> dict"
        )

    def invoke(self, message: Any, hints: Optional[Dict] = None) -> Any:
        merged_hints = {**self._hints, **(hints or {})}
        return self.client.invoke(message, hints=merged_hints)


def get_client(
    hints: Dict,
    config_path: str = _DEFAULT_CONFIG_PATH
) -> WrappedLangchainClient:

    return WrappedLangchainClient.auto(config_path)

def get_model_client(
    model_name: str,
    config_path: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500
) -> WrappedLangchainClient:
    return WrappedLangchainClient.from_model(
        model_name=model_name,
        config_path=config_path,
        temperature=temperature,
        max_tokens=max_tokens
    )