import importlib
from abc import ABC, abstractmethod

from alphaflow.core.alpha_base.base_alpha import BaseAlpha
from alphaflow.core.alpha_base.high_alpha import HighAlpha
from alphaflow.core.alpha_base.low_alpha import LowAlpha
from alphaflow.core.alpha_base.mid_alpha import MidAlpha
from alphaflow.core.alpha_base.rtn_alpha import RtnAlpha
from alphaflow.core.config.alpha_config.base_config import BaseAlphaConfig
from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.loader import load_alpha_config
from alphaflow.core.config.paths import DEFAULT_UNIVERSE_DIR
from alphaflow.core.config.registry_config import AlphaType, RegistryEntry
from alphaflow.core.config.system_config import MarketConfig
from alphaflow.core.data_access.pit_manager import PITDataManager


def resolve_handler_class(handler_class: str, expected_parent: type[BaseAlpha]) -> type[BaseAlpha]:
    """Import 'pkg.module.ClassName' and assert it subclasses the right typed base -
    check #5 of spec section 3.9."""
    module_path, _, class_name = handler_class.rpartition(".")
    if not module_path:
        raise ConfigValidationError(f"handler_class {handler_class!r} must be a fully qualified 'module.ClassName' path")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigValidationError(f"cannot import module {module_path!r} for handler_class {handler_class!r}: {exc}") from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ConfigValidationError(f"module {module_path!r} has no attribute {class_name!r}")
    if not (isinstance(cls, type) and issubclass(cls, expected_parent)):
        raise ConfigValidationError(f"{handler_class} must subclass {expected_parent.__name__}, got {cls}")
    return cls


class TypedAlphaLoader(ABC):
    ALPHA_TYPE: AlphaType
    PARENT_CLASS: type[BaseAlpha]

    @abstractmethod
    def load_config(self, config_file: str) -> BaseAlphaConfig: ...

    def resolve_class(self, handler_class: str) -> type[BaseAlpha]:
        return resolve_handler_class(handler_class, self.PARENT_CLASS)

    def load(self, config_file: str, data: PITDataManager, universe_dir: str = DEFAULT_UNIVERSE_DIR, **kwargs) -> BaseAlpha:
        config = self.load_config(config_file)
        cls = self.resolve_class(config.handler_class)
        return cls(config=config, data=data, universe_dir=universe_dir, **kwargs)


class HighAlphaLoader(TypedAlphaLoader):
    ALPHA_TYPE: AlphaType = "high"
    PARENT_CLASS = HighAlpha

    def load_config(self, config_file: str) -> BaseAlphaConfig:
        return load_alpha_config(config_file, "high")


class MidAlphaLoader(TypedAlphaLoader):
    ALPHA_TYPE: AlphaType = "mid"
    PARENT_CLASS = MidAlpha

    def load_config(self, config_file: str) -> BaseAlphaConfig:
        return load_alpha_config(config_file, "mid")


class LowAlphaLoader(TypedAlphaLoader):
    ALPHA_TYPE: AlphaType = "low"
    PARENT_CLASS = LowAlpha

    def load_config(self, config_file: str) -> BaseAlphaConfig:
        return load_alpha_config(config_file, "low")


class RtnAlphaLoader(TypedAlphaLoader):
    ALPHA_TYPE: AlphaType = "rtn"
    PARENT_CLASS = RtnAlpha

    def load_config(self, config_file: str) -> BaseAlphaConfig:
        return load_alpha_config(config_file, "rtn")

    def load(self, config_file: str, data: PITDataManager, universe_dir: str = DEFAULT_UNIVERSE_DIR,
             market_config: dict[str, MarketConfig] | None = None, **kwargs) -> BaseAlpha:
        # RtnAlpha alone needs the session table, for its horizon=0 sanity check
        return super().load(config_file, data, universe_dir, market_config=market_config, **kwargs)


class AlphaLoader:
    _loader_map: dict[AlphaType, type[TypedAlphaLoader]] = {
        "high": HighAlphaLoader, "mid": MidAlphaLoader, "low": LowAlphaLoader, "rtn": RtnAlphaLoader,
    }

    @classmethod
    def load(cls, entry: RegistryEntry, data: PITDataManager, universe_dir: str = DEFAULT_UNIVERSE_DIR,
             market_config: dict[str, MarketConfig] | None = None) -> BaseAlpha:
        loader_cls = cls._loader_map.get(entry.alpha_type)
        if loader_cls is None:
            raise ConfigValidationError(f"unknown alpha_type {entry.alpha_type!r} for {entry.alpha_id}")
        loader = loader_cls()
        kwargs = {"market_config": market_config} if entry.alpha_type == "rtn" else {}
        alpha = loader.load(entry.config_file, data, universe_dir, **kwargs)
        if alpha.alpha_id != entry.alpha_id:
            raise ConfigValidationError(f"registry alpha_id {entry.alpha_id!r} does not match config name {alpha.alpha_id!r} in {entry.config_file}")
        return alpha

    @classmethod
    def loader_for(cls, alpha_type: AlphaType) -> TypedAlphaLoader:
        return cls._loader_map[alpha_type]()
