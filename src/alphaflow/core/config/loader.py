import tomllib
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from alphaflow.core.config.alpha_config.base_config import BaseAlphaConfig
from alphaflow.core.config.alpha_config.high_config import HighAlphaConfig
from alphaflow.core.config.alpha_config.low_config import LowAlphaConfig
from alphaflow.core.config.alpha_config.mid_config import MidAlphaConfig
from alphaflow.core.config.alpha_config.rtn_config import RtnAlphaConfig
from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.model_config import ModelConfig
from alphaflow.core.config.optimizer_config import OptimizerConfig
from alphaflow.core.config.paths import DEFAULT_MODELS_DIR, resolve_config_path
from alphaflow.core.config.production_config import ProductionConfig
from alphaflow.core.config.registry_config import AlphaType, RegistryConfig
from alphaflow.core.config.system_config import SystemConfig

T = TypeVar("T", bound=BaseModel)

ALPHA_CONFIG_CLASSES: dict[AlphaType, type[BaseAlphaConfig]] = {
    "high": HighAlphaConfig, "mid": MidAlphaConfig, "low": LowAlphaConfig, "rtn": RtnAlphaConfig,
}

# Alpha TOMLs are section-organised for readability; these sections flatten onto the model.
# NOTE: the spec's example places bare `horizon`/`return_type` keys after [exit], which TOML
# would absorb into [exit]. They live in an explicit [timing] section instead.
_FLATTENED_SECTIONS = ("identity", "data", "params")
_RTN_SECTIONS = ("entry", "exit")


def read_toml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_config_path(path)
    if not resolved.is_file():
        raise ConfigValidationError(f"config file not found: {resolved}")
    try:
        with open(resolved, "rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigValidationError(f"malformed TOML in {resolved}: {exc}") from exc


def _build(model_cls: type[T], payload: dict[str, Any], source: Path | str) -> T:
    try:
        return model_cls(**payload)
    except ValidationError as exc:
        raise ConfigValidationError(f"invalid {model_cls.__name__} in {source}:\n{exc}") from exc


def load_system_config(path: str | Path = "configs/system.toml") -> SystemConfig:
    return _build(SystemConfig, read_toml(path), resolve_config_path(path))


def load_production_config(path: str | Path = "configs/production.toml") -> ProductionConfig:
    return _build(ProductionConfig, read_toml(path), resolve_config_path(path))


def load_optimizer_config(path: str | Path = "configs/optimizer.toml") -> OptimizerConfig:
    return _build(OptimizerConfig, read_toml(path), resolve_config_path(path))


def load_registry_config(path: str | Path = "configs/registry.toml") -> RegistryConfig:
    return _build(RegistryConfig, read_toml(path), resolve_config_path(path))


def load_model_config(path: str | Path) -> ModelConfig:
    resolved = resolve_config_path(path)
    payload = read_toml(resolved)
    payload["source_path"] = str(resolved)
    return _build(ModelConfig, payload, resolved)


def load_model_configs(models_dir: str | Path = DEFAULT_MODELS_DIR) -> dict[str, ModelConfig]:
    """Every *.toml in models_dir, keyed by model_id. Files starting with '_' are templates."""
    resolved = resolve_config_path(models_dir)
    if not resolved.is_dir():
        raise ConfigValidationError(f"models directory not found: {resolved}")
    configs = {}
    for toml_path in sorted(resolved.glob("*.toml")):
        if toml_path.name.startswith("_"):
            continue
        config = load_model_config(toml_path)
        configs[config.model_id] = config
    return configs


def flatten_alpha_payload(payload: dict[str, Any], alpha_type: AlphaType) -> dict[str, Any]:
    """Collapse the section-organised alpha TOML into flat model kwargs."""
    flat: dict[str, Any] = {}
    for section in _FLATTENED_SECTIONS:
        value = payload.get(section, {})
        if not isinstance(value, dict):
            raise ConfigValidationError(f"[{section}] must be a table, got {type(value).__name__}")
        flat.update(value)
    # Any bare top-level keys (written before the first table header) still apply
    flat.update({k: v for k, v in payload.items() if not isinstance(v, dict)})
    if alpha_type == "rtn":
        for section in _RTN_SECTIONS:
            if section in payload:
                flat[section] = payload[section]
        flat.update(payload.get("timing", {}))
    return flat


def load_alpha_config(path: str | Path, alpha_type: AlphaType | None = None) -> BaseAlphaConfig:
    """alpha_type is taken from the TOML unless the caller (a typed loader) pins it,
    in which case a mismatch is an error - that is check #6 of spec section 3.9."""
    resolved = resolve_config_path(path)
    payload = read_toml(resolved)
    declared = payload.get("identity", {}).get("alpha_type") or payload.get("alpha_type")
    if declared is None:
        raise ConfigValidationError(f"{resolved}: alpha_type missing from [identity]")
    if alpha_type is not None and declared != alpha_type:
        raise ConfigValidationError(f"{resolved}: alpha_type is {declared!r} but was loaded as {alpha_type!r}")
    if declared not in ALPHA_CONFIG_CLASSES:
        raise ConfigValidationError(f"{resolved}: unknown alpha_type {declared!r}, expected one of {sorted(ALPHA_CONFIG_CLASSES)}")
    return _build(ALPHA_CONFIG_CLASSES[declared], flatten_alpha_payload(payload, declared), resolved)
