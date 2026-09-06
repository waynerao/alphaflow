from pathlib import Path

from alphaflow.core.alpha_base.loader import AlphaLoader
from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.loader import load_alpha_config, load_model_configs, load_optimizer_config, load_production_config, load_registry_config
from alphaflow.core.config.model_config import ModelConfig
from alphaflow.core.config.optimizer_config import OptimizerConfig
from alphaflow.core.config.paths import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_CONFIG,
    resolve_config_path,
)
from alphaflow.core.config.production_config import ProductionConfig
from alphaflow.core.config.registry_config import AlphaType, RegistryConfig, RegistryEntry
from alphaflow.core.config.system_config import SystemConfig


class AlphaRegistry:
    """Master alpha list plus every cross-config consistency check from spec section 3.9.

    All checks run on construction: an inconsistent config set should fail at startup, not
    halfway through a production run.
    """

    def __init__(self, registry_config: RegistryConfig, model_configs: list[ModelConfig],
                 production_config: ProductionConfig | None = None, optimizer_config: OptimizerConfig | None = None,
                 system_config: SystemConfig | None = None, models_dir: str = DEFAULT_MODELS_DIR,
                 validate_handlers: bool = True) -> None:
        self.registry_config = registry_config
        self.models = {m.model_id: m for m in model_configs}
        self.production_config = production_config
        self.optimizer_config = optimizer_config
        self.system_config = system_config
        self.models_dir = models_dir
        self.validate()
        if validate_handlers:
            self.validate_handler_classes()

    @classmethod
    def from_paths(cls, registry_path: str = DEFAULT_REGISTRY_CONFIG, models_dir: str = DEFAULT_MODELS_DIR,
                   production_path: str | None = None, optimizer_path: str | None = None,
                   system_config: SystemConfig | None = None, validate_handlers: bool = True) -> "AlphaRegistry":
        return cls(
            registry_config=load_registry_config(registry_path),
            model_configs=list(load_model_configs(models_dir).values()),
            production_config=load_production_config(production_path) if production_path else None,
            optimizer_config=load_optimizer_config(optimizer_path) if optimizer_path else None,
            system_config=system_config, models_dir=models_dir, validate_handlers=validate_handlers,
        )

    # -- lookups ------------------------------------------------------------------------

    def get_active(self) -> list[RegistryEntry]:
        return self.registry_config.get_active()

    def get_by_id(self, alpha_id: str) -> RegistryEntry | None:
        return self.registry_config.get_by_id(alpha_id)

    def get_by_type(self, alpha_type: AlphaType) -> list[RegistryEntry]:
        return self.registry_config.get_by_type(alpha_type)

    def get_model(self, model_id: str) -> ModelConfig | None:
        return self.models.get(model_id)

    def require_model(self, model_id: str) -> ModelConfig:
        model = self.get_model(model_id)
        if model is None:
            raise ConfigValidationError(f"model {model_id!r} not found in {self.models_dir}; known models: {sorted(self.models)}")
        return model

    def require_entry(self, alpha_id: str) -> RegistryEntry:
        entry = self.get_by_id(alpha_id)
        if entry is None:
            raise ConfigValidationError(f"alpha_id {alpha_id!r} not in registry")
        return entry

    def get_alphas_for_model(self, model_id: str) -> list[RegistryEntry]:
        """The model's alpha_ids, in declared order. rtn_alpha_id is deliberately excluded -
        it is the return source, not a component of the composite."""
        return [self.require_entry(alpha_id) for alpha_id in self.require_model(model_id).alpha_ids]

    def get_rtn_alpha_for_model(self, model_id: str) -> RegistryEntry:
        return self.require_entry(self.require_model(model_id).rtn_alpha_id)

    def active_model_ids(self) -> list[str]:
        if self.production_config is None:
            return sorted(self.models)
        return list(self.production_config.models.active_model_ids)

    # -- cross-config validation (spec section 3.9) -------------------------------------

    def validate(self) -> None:
        errors: list[str] = []
        errors += self._check_model_alpha_ids()
        errors += self._check_production_models()
        errors += self._check_alpha_type_consistency()
        errors += self._check_required_inputs()
        if errors:
            raise ConfigValidationError("cross-config validation failed:\n  - " + "\n  - ".join(errors))

    def _check_model_alpha_ids(self) -> list[str]:
        """Checks 1, 2 and 4: model alpha_ids and rtn_alpha_id exist, are active and are the
        right type; alpha_weights keys match (the ModelConfig validator covers the last one,
        so any failure would already have surfaced at load time)."""
        errors = []
        for model_id, model in self.models.items():
            for alpha_id in model.alpha_ids:
                entry = self.get_by_id(alpha_id)
                if entry is None:
                    errors.append(f"model {model_id!r}: alpha_id {alpha_id!r} is not in the registry")
                elif not entry.active:
                    errors.append(f"model {model_id!r}: alpha_id {alpha_id!r} is registered but inactive")
            rtn_entry = self.get_by_id(model.rtn_alpha_id)
            if rtn_entry is None:
                errors.append(f"model {model_id!r}: rtn_alpha_id {model.rtn_alpha_id!r} is not in the registry")
            else:
                if rtn_entry.alpha_type != "rtn":
                    errors.append(f"model {model_id!r}: rtn_alpha_id {model.rtn_alpha_id!r} has alpha_type {rtn_entry.alpha_type!r}, expected 'rtn'")
                if not rtn_entry.active:
                    errors.append(f"model {model_id!r}: rtn_alpha_id {model.rtn_alpha_id!r} is registered but inactive")
        return errors

    def _check_production_models(self) -> list[str]:
        """Check 3: every active_model_id in production.toml has a model TOML."""
        if self.production_config is None:
            return []
        return [f"production.toml: active model {model_id!r} has no configs/models/{model_id}.toml"
                for model_id in self.production_config.models.active_model_ids if model_id not in self.models]

    def _check_alpha_type_consistency(self) -> list[str]:
        """Check 6: per-alpha TOML alpha_type matches the registry entry. Also check 7 -
        the config_file must exist at all."""
        errors = []
        for entry in self.registry_config.alphas:
            path = Path(resolve_config_path(entry.config_file))
            if not path.is_file():
                errors.append(f"alpha {entry.alpha_id!r}: config_file not found at {path}")
                continue
            try:
                config = load_alpha_config(path)
            except ConfigValidationError as exc:
                errors.append(f"alpha {entry.alpha_id!r}: {exc}")
                continue
            if config.alpha_type != entry.alpha_type:
                errors.append(f"alpha {entry.alpha_id!r}: registry says {entry.alpha_type!r}, TOML says {config.alpha_type!r}")
            if config.name != entry.alpha_id:
                errors.append(f"alpha {entry.alpha_id!r}: TOML name is {config.name!r}")
        return errors

    def _check_required_inputs(self) -> list[str]:
        """Check 8: every required_input must be available in every market the alpha claims."""
        if self.system_config is None:
            return []
        errors = []
        for entry in self.registry_config.alphas:
            path = Path(resolve_config_path(entry.config_file))
            if not path.is_file():
                continue
            try:
                config = load_alpha_config(path)
            except ConfigValidationError:
                continue
            for market in config.market_scope:
                market_name = str(market)
                if market_name not in self.system_config.markets:
                    errors.append(f"alpha {entry.alpha_id!r}: market_scope includes {market_name!r}, absent from system.toml")
                    continue
                available = self.system_config.available_data_types(market_name)
                missing = [d for d in config.required_inputs if d not in available]
                if missing:
                    errors.append(f"alpha {entry.alpha_id!r}: required_inputs {missing} unavailable in {market_name} (available: {available})")
        return errors

    def validate_handler_classes(self) -> None:
        """Check 5: each handler_class imports and subclasses the right typed base. Split
        out from validate() because it imports application code, which tests often stub."""
        errors = []
        for entry in self.registry_config.alphas:
            path = Path(resolve_config_path(entry.config_file))
            if not path.is_file():
                continue
            try:
                config = load_alpha_config(path)
                loader = AlphaLoader.loader_for(entry.alpha_type)
                loader.resolve_class(config.handler_class)
            except ConfigValidationError as exc:
                errors.append(f"alpha {entry.alpha_id!r}: {exc}")
        if errors:
            raise ConfigValidationError("handler class validation failed:\n  - " + "\n  - ".join(errors))
