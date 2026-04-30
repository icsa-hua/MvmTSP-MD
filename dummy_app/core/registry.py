from __future__ import annotations

from typing import Any, Callable, Dict, Type


RuntimeFactory = Callable[[Dict[str, Any], int], Any]


class ModelRegistry:
    def __init__(self) -> None:
        self._runtimes: Dict[str, RuntimeFactory] = {}

    def register(self, model_name: str, factory: RuntimeFactory) -> None:
        self._runtimes[model_name] = factory

    def create(self, model_name: str, config: Dict[str, Any], trials: int) -> Any:
        if model_name not in self._runtimes:
            raise ValueError(f"Unknown model_name '{model_name}'. Registered models: {sorted(self._runtimes)}")
        return self._runtimes[model_name](config, trials)

    def list_models(self) -> list[str]:
        return sorted(self._runtimes)


_REGISTRY = ModelRegistry()


def ensure_default_registrations() -> None:
    if "milp" in _REGISTRY.list_models():
        return

    from dummy_app.models.simulation_builder import Builder

    _REGISTRY.register("milp", lambda config, trials: Builder(config, trials))


def create_runtime(config: Dict[str, Any], trials: int) -> Any:
    ensure_default_registrations()
    model_name = str(config.get("model_name", "milp"))
    return _REGISTRY.create(model_name, config, trials)
