import threading
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from .config_loader import load_config, RouterConfig, ModelConfig, GroupConfig
from .recommender import recommend, RecommendationResult


class _RoundRobinStrategy:
    def __init__(self, model_ids: List[str]):
        self._ids   = model_ids
        self._index = 0
        self._lock  = threading.Lock()

    def get_order(self) -> List[str]:
        with self._lock:
            ordered = self._ids[self._index:] + self._ids[:self._index]
            self._index = (self._index + 1) % len(self._ids)
        return ordered


class _PriorityStrategy:
    def __init__(self, model_ids: List[str]):
        self._ids = model_ids

    def get_order(self) -> List[str]:
        return list(self._ids)


def _build_strategy(group: GroupConfig):
    if group.routing_strategy == "round-robin":
        return _RoundRobinStrategy(group.models)
    return _PriorityStrategy(group.models)


def _build_ollama_model(
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 500
) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_predict=max_tokens
    )


def _build_ollama_model_from_config(model: ModelConfig) -> ChatOllama:
    params = model.parameters
    return _build_ollama_model(
        model_name=model.model_name,
        temperature=params.get("temperature", 0.7),
        max_tokens=params.get("max_tokens", 500)
    )


def _to_messages(message: Any) -> List[BaseMessage]:
    if isinstance(message, str):
        return [HumanMessage(content=message)]
    if isinstance(message, BaseMessage):
        return [message]
    return list(message)  # already a list


def _to_raw_str(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, BaseMessage):
        return message.content
    if isinstance(message, list) and message:
        first = message[-1]
        return first.content if isinstance(first, BaseMessage) else str(first)
    return str(message)


def _invoke_direct(
    messages: List[BaseMessage],
    model_name: str,
    temperature: float,
    max_tokens: int
) -> AIMessage:
    llm = _build_ollama_model(model_name, temperature, max_tokens)
    return llm.invoke(messages)


async def _ainvoke_direct(
    messages: List[BaseMessage],
    model_name: str,
    temperature: float,
    max_tokens: int
) -> AIMessage:
    llm = _build_ollama_model(model_name, temperature, max_tokens)
    return await llm.ainvoke(messages)


def _invoke_with_fallback(
    messages: List[BaseMessage],
    winner_model: ModelConfig,
    fallback_group: Optional[GroupConfig],
    config: RouterConfig
) -> AIMessage:
    try:
        print(f"[wrapped_client] Invoking: {winner_model.id} ({winner_model.model_name})")
        llm = _build_ollama_model_from_config(winner_model)
        return llm.invoke(messages)
    except Exception as e:
        print(f"[wrapped_client] '{winner_model.id}' failed: {e}")

    if fallback_group is None:
        raise RuntimeError(f"Model '{winner_model.id}' failed and no fallback group configured.")

    strategy = _build_strategy(fallback_group)
    for model_id in strategy.get_order():
        if model_id == winner_model.id:
            continue
        if model_id not in config.models:
            continue
        try:
            print(f"[wrapped_client] Fallback → {model_id}")
            llm = _build_ollama_model_from_config(config.models[model_id])
            result = llm.invoke(messages)
            print(f"[wrapped_client] Fallback succeeded: {model_id}")
            return result
        except Exception as e:
            print(f"[wrapped_client] Fallback '{model_id}' failed: {e}")

    raise RuntimeError(
        f"All models failed — winner '{winner_model.id}' "
        f"+ fallback group '{fallback_group.name}': {strategy.get_order()}"
    )


async def _ainvoke_with_fallback(
    messages: List[BaseMessage],
    winner_model: ModelConfig,
    fallback_group: Optional[GroupConfig],
    config: RouterConfig
) -> AIMessage:
    try:
        print(f"[wrapped_client] [async] Invoking: {winner_model.id}")
        llm = _build_ollama_model_from_config(winner_model)
        return await llm.ainvoke(messages)
    except Exception as e:
        print(f"[wrapped_client] [async] '{winner_model.id}' failed: {e}")

    if fallback_group is None:
        raise RuntimeError(
            f"Model '{winner_model.id}' failed and no fallback group configured."
        )

    strategy = _build_strategy(fallback_group)
    for model_id in strategy.get_order():
        if model_id == winner_model.id:
            continue
        if model_id not in config.models:
            continue
        try:
            print(f"[wrapped_client] [async] Fallback → {model_id}")
            llm = _build_ollama_model_from_config(config.models[model_id])
            result = await llm.ainvoke(messages)
            print(f"[wrapped_client] [async] Fallback succeeded: {model_id}")
            return result
        except Exception as e:
            print(f"[wrapped_client] [async] Fallback '{model_id}' failed: {e}")

    raise RuntimeError(
        f"All models failed (async) — winner + fallback group '{fallback_group.name}'"
    )


def _resolve_config_path(config_path: str) -> str:
    import os

    if os.path.isabs(config_path) or os.path.exists(config_path):
        return config_path

    cwd_path = os.path.join(os.getcwd(), config_path)
    if os.path.exists(cwd_path):
        return cwd_path

    current = os.getcwd()
    for _ in range(5):
        candidate = os.path.join(current, config_path)
        if os.path.exists(candidate):
            print(f"[wrapped_client] Found config at: {candidate}")
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return config_path


class _Mode:
    MODEL_ONLY   = "model_only"
    CONFIG_ONLY  = "config_only"
    MODEL_CONFIG = "model_config"


class WrappedLangchainClient:
    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        mode: str = _Mode.CONFIG_ONLY
    ):
        self._config      = config
        self._model_name  = model_name
        self._temperature = temperature
        self._max_tokens  = max_tokens
        self._mode        = mode
        self._last_recommendation: Optional[RecommendationResult] = None

    @classmethod
    def from_model(
        cls,
        model_name: str,
        config_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> "WrappedLangchainClient":
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")

        config = None
        mode   = _Mode.MODEL_ONLY

        if config_path:
            resolved = _resolve_config_path(config_path)
            config   = load_config(resolved)
            mode     = _Mode.MODEL_CONFIG
            print(
                f"[wrapped_client] Mode: model+config "
                f"(primary='{model_name}', fallback via config)"
            )
        else:
            print(f"[wrapped_client] Mode: model-only ('{model_name}')")

        return cls(
            config=config,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            mode=mode
        )

    @classmethod
    def auto(cls, config_path: str) -> "WrappedLangchainClient":
        resolved = _resolve_config_path(config_path)
        config   = load_config(resolved)
        print(f"[wrapped_client] Mode: config-only (recommendation engine active)")
        return cls(config=config, mode=_Mode.CONFIG_ONLY)

    def invoke(self, message: Any, hints: Optional[Dict] = None) -> AIMessage:
        hints    = hints or {}
        messages = _to_messages(message)

        if self._mode == _Mode.MODEL_ONLY:
            return self._invoke_model_only(messages)

        if self._mode == _Mode.CONFIG_ONLY:
            return self._invoke_config_only(messages, hints)

        if self._mode == _Mode.MODEL_CONFIG:
            return self._invoke_model_config(messages, hints)

        raise RuntimeError(f"Unknown mode: {self._mode}")

    async def ainvoke(self, message: Any, hints: Optional[Dict] = None) -> AIMessage:
        hints    = hints or {}
        messages = _to_messages(message)

        if self._mode == _Mode.MODEL_ONLY:
            return await self._ainvoke_model_only(messages)

        if self._mode == _Mode.CONFIG_ONLY:
            return await self._ainvoke_config_only(messages, hints)

        if self._mode == _Mode.MODEL_CONFIG:
            return await self._ainvoke_model_config(messages, hints)

        raise RuntimeError(f"Unknown mode: {self._mode}")

    def _invoke_model_only(self, messages: List[BaseMessage]) -> AIMessage:
        print(f"[wrapped_client] Direct invoke: {self._model_name}")
        return _invoke_direct(
            messages, self._model_name,
            self._temperature, self._max_tokens
        )

    async def _ainvoke_model_only(self, messages: List[BaseMessage]) -> AIMessage:
        print(f"[wrapped_client] [async] Direct invoke: {self._model_name}")
        return await _ainvoke_direct(
            messages, self._model_name,
            self._temperature, self._max_tokens
        )


    def _invoke_config_only(
        self,
        messages: List[BaseMessage],
        hints: Dict
    ) -> AIMessage:
        raw_message  = _to_raw_str(messages)
        recommendation = recommend(
            message=raw_message,
            hints=hints,
            config=self._config
        )
        self._last_recommendation = recommendation

        use_case_name       = recommendation.use_case_name
        fallback_group_name = (
            recommendation.fallback_group
            or self._config.use_cases[use_case_name].fallback_group
        )
        fallback_group_cfg = self._config.groups.get(fallback_group_name)

        return _invoke_with_fallback(
            messages=messages,
            winner_model=recommendation.winner_model,
            fallback_group=fallback_group_cfg,
            config=self._config
        )

    async def _ainvoke_config_only(
        self,
        messages: List[BaseMessage],
        hints: Dict
    ) -> AIMessage:
        raw_message  = _to_raw_str(messages)
        recommendation = recommend(
            message=raw_message,
            hints=hints,
            config=self._config
        )
        self._last_recommendation = recommendation

        use_case_name       = recommendation.use_case_name
        fallback_group_name = (
            recommendation.fallback_group
            or self._config.use_cases[use_case_name].fallback_group
        )
        fallback_group_cfg = self._config.groups.get(fallback_group_name)

        return await _ainvoke_with_fallback(
            messages=messages,
            winner_model=recommendation.winner_model,
            fallback_group=fallback_group_cfg,
            config=self._config
        )

    def _invoke_model_config(
        self,
        messages: List[BaseMessage],
        hints: Dict
    ) -> AIMessage:
        try:
            print(f"[wrapped_client] Trying named model: '{self._model_name}'")
            result = _invoke_direct(
                messages, self._model_name,
                self._temperature, self._max_tokens
            )
            print(f"[wrapped_client] Named model succeeded: '{self._model_name}'")
            return result

        except Exception as e:
            print(
                f"[wrapped_client] Named model '{self._model_name}' failed: {e}\n"
                f"[wrapped_client] Falling back to recommendation engine..."
            )

        return self._invoke_config_only(messages, hints)

    async def _ainvoke_model_config(
        self,
        messages: List[BaseMessage],
        hints: Dict
    ) -> AIMessage:
        try:
            print(f"[wrapped_client] [async] Trying named model: '{self._model_name}'")
            result = await _ainvoke_direct(
                messages, self._model_name,
                self._temperature, self._max_tokens
            )
            print(f"[wrapped_client] [async] Named model succeeded: '{self._model_name}'")
            return result

        except Exception as e:
            print(
                f"[wrapped_client] [async] Named model '{self._model_name}' failed: {e}\n"
                f"[wrapped_client] [async] Falling back to recommendation engine..."
            )

        return await self._ainvoke_config_only(messages, hints)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def last_recommendation(self) -> Optional[RecommendationResult]:
        return self._last_recommendation

    def explain_last(self) -> str:
        if not self._last_recommendation:
            if self._mode == _Mode.MODEL_ONLY:
                return f"Model-only mode — '{self._model_name}' invoked directly, no recommendation made."
            return "No recommendation made yet."
        return self._last_recommendation.explanation

    def available_use_cases(self) -> List[str]:
        if self._config is None:
            return []
        return list(self._config.use_cases.keys())

    def available_models(self) -> List[str]:
        if self._config is None:
            return [self._model_name] if self._model_name else []
        return list(self._config.models.keys())

    def __repr__(self) -> str:
        if self._mode == _Mode.MODEL_ONLY:
            return f"WrappedLangchainClient(mode=model_only, model='{self._model_name}')"
        if self._mode == _Mode.MODEL_CONFIG:
            return f"WrappedLangchainClient(mode=model+config, model='{self._model_name}')"
        return f"WrappedLangchainClient(mode=config_only)"