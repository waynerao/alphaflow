import pytest

from alphaflow.core.alpha_base.high_alpha import HighAlpha
from alphaflow.core.alpha_base.loader import AlphaLoader, LowAlphaLoader, resolve_handler_class
from alphaflow.core.alpha_base.low_alpha import LowAlpha
from alphaflow.core.alpha_base.registry import AlphaRegistry
from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.model_config import ModelConfig
from alphaflow.core.config.registry_config import RegistryConfig, RegistryEntry

MOMENTUM = "momentum_price_21d"
IMBALANCE = "order_imbalance_1430"
RETURN = "open_to_close_1d"


def entry(alpha_id, alpha_type, config_file, active=True):
    return RegistryEntry(alpha_id=alpha_id, alpha_type=alpha_type, config_file=config_file, active=active)


def shipped_entries(**overrides):
    entries = [
        entry(MOMENTUM, "low", "configs/alphas/low/momentum_price_21d.toml"),
        entry(IMBALANCE, "high", "configs/alphas/high/order_imbalance_1430.toml"),
        entry(RETURN, "rtn", "configs/alphas/rtn/open_to_close_1d.toml"),
    ]
    for item in entries:
        for field, value in overrides.get(item.alpha_id, {}).items():
            setattr(item, field, value)
    return RegistryConfig(alphas=entries)


def model(**overrides):
    base = {"model_id": "m", "market": "HK", "alpha_ids": [MOMENTUM], "rtn_alpha_id": RETURN,
            "start_date": "20240101", "end_date": "20241231"}
    return ModelConfig(**{**base, **overrides})


class TestShippedConfigSet:
    def test_the_repo_configs_pass_every_check(self, registry):
        assert {e.alpha_id for e in registry.get_active()} == {MOMENTUM, IMBALANCE, RETURN}

    def test_model_lookups(self, registry):
        assert [e.alpha_id for e in registry.get_alphas_for_model("hk_momentum")] == [MOMENTUM, IMBALANCE]
        assert registry.get_rtn_alpha_for_model("hk_momentum").alpha_id == RETURN

    def test_active_model_ids_come_from_production_config(self, registry):
        assert registry.active_model_ids() == ["hk_momentum"]

    def test_unknown_model_is_reported(self, registry):
        with pytest.raises(ConfigValidationError, match="not found"):
            registry.require_model("nope")

    def test_unknown_alpha_is_reported(self, registry):
        with pytest.raises(ConfigValidationError, match="not in registry"):
            registry.require_entry("nope")


class TestCrossConfigChecks:
    def test_check1_model_alpha_must_exist_in_the_registry(self):
        with pytest.raises(ConfigValidationError, match="is not in the registry"):
            AlphaRegistry(shipped_entries(), [model(alpha_ids=["ghost"])])

    def test_check1_model_alpha_must_be_active(self):
        registry_config = shipped_entries(**{MOMENTUM: {"active": False}})
        with pytest.raises(ConfigValidationError, match="registered but inactive"):
            AlphaRegistry(registry_config, [model()])

    def test_check2_rtn_alpha_must_exist(self):
        with pytest.raises(ConfigValidationError, match="rtn_alpha_id 'ghost' is not in the registry"):
            AlphaRegistry(shipped_entries(), [model(rtn_alpha_id="ghost")])

    def test_check2_rtn_alpha_must_be_of_type_rtn(self):
        with pytest.raises(ConfigValidationError, match="expected 'rtn'"):
            AlphaRegistry(shipped_entries(), [model(rtn_alpha_id=MOMENTUM)])

    def test_check3_production_model_must_have_a_config_file(self, production_config):
        with pytest.raises(ConfigValidationError, match="has no configs/models/"):
            AlphaRegistry(shipped_entries(), [], production_config=production_config)

    def test_check4_weight_keys_are_enforced_at_load_time(self):
        with pytest.raises(Exception, match="must match alpha_ids"):
            model(alpha_weights={"wrong": 1.0})

    def test_check5_handler_must_subclass_the_right_base(self):
        with pytest.raises(ConfigValidationError, match="must subclass HighAlpha"):
            resolve_handler_class("alphaflow.signal_builder.alphas.low.momentum_price_21d.MomentumPrice21D", HighAlpha)

    def test_check5_accepts_a_correct_handler(self):
        assert resolve_handler_class("alphaflow.signal_builder.alphas.low.momentum_price_21d.MomentumPrice21D", LowAlpha)

    def test_check5_reports_an_unimportable_module(self):
        with pytest.raises(ConfigValidationError, match="cannot import module"):
            resolve_handler_class("alphaflow.nope.Missing", LowAlpha)

    def test_check5_reports_a_missing_class(self):
        with pytest.raises(ConfigValidationError, match="has no attribute"):
            resolve_handler_class("alphaflow.core.alpha_base.low_alpha.NotThere", LowAlpha)

    def test_check5_rejects_an_unqualified_path(self):
        with pytest.raises(ConfigValidationError, match="fully qualified"):
            resolve_handler_class("JustAClass", LowAlpha)

    def test_check6_registry_type_must_match_the_toml(self):
        registry_config = shipped_entries(**{MOMENTUM: {"alpha_type": "mid"}})
        with pytest.raises(ConfigValidationError, match="registry says 'mid', TOML says 'low'"):
            AlphaRegistry(registry_config, [])

    def test_check7_a_missing_config_file_is_reported(self):
        registry_config = shipped_entries(**{MOMENTUM: {"config_file": "configs/alphas/low/ghost.toml"}})
        with pytest.raises(ConfigValidationError, match="config_file not found"):
            AlphaRegistry(registry_config, [])

    def test_check8_required_inputs_must_exist_in_every_scoped_market(self, system_config, tmp_path):
        toml = tmp_path / "bad_inputs.toml"
        toml.write_text('[identity]\nname = "bad_inputs"\ndescription = "d"\ncategory = "c"\nformula = "f"\n'
                        'handler_class = "alphaflow.signal_builder.alphas.low.momentum_price_21d.MomentumPrice21D"\n'
                        'alpha_type = "low"\nmarket_scope = ["KR"]\n\n[data]\nrequired_inputs = ["southbound"]\n\n'
                        '[params]\nlookback_days = 5\n')
        registry_config = RegistryConfig(alphas=[entry("bad_inputs", "low", str(toml))])
        with pytest.raises(ConfigValidationError, match="unavailable in KR"):
            AlphaRegistry(registry_config, [], system_config=system_config)

    def test_all_failures_are_reported_together(self):
        with pytest.raises(ConfigValidationError) as exc:
            AlphaRegistry(shipped_entries(), [model(alpha_ids=["ghost1", "ghost2"], rtn_alpha_id="ghost3")])
        assert exc.value.args[0].count("- ") >= 3


class TestAlphaLoader:
    def test_loads_each_shipped_alpha_with_the_right_class(self, registry, system_config):
        classes = {e.alpha_id: type(AlphaLoader.load(e, None, market_config=system_config.markets)).__name__
                   for e in registry.get_active()}
        assert classes == {MOMENTUM: "MomentumPrice21D", IMBALANCE: "OrderImbalance1430", RETURN: "OpenToClose1D"}

    def test_rtn_alphas_receive_the_market_config(self, registry, system_config):
        alpha = AlphaLoader.load(registry.require_entry(RETURN), None, market_config=system_config.markets)
        assert alpha.market_config is not None

    def test_a_registry_id_that_disagrees_with_the_toml_is_rejected(self):
        mismatched = entry("wrong_id", "low", "configs/alphas/low/momentum_price_21d.toml")
        with pytest.raises(ConfigValidationError, match="does not match config name"):
            AlphaLoader.load(mismatched, None)

    def test_an_unknown_alpha_type_is_rejected(self):
        bad = RegistryEntry.model_construct(alpha_id="x", alpha_type="weird", config_file="f.toml", active=True)
        with pytest.raises(ConfigValidationError, match="unknown alpha_type"):
            AlphaLoader.load(bad, None)

    def test_typed_loader_pins_the_alpha_type(self):
        with pytest.raises(ConfigValidationError, match="but was loaded as 'low'"):
            LowAlphaLoader().load_config("configs/alphas/rtn/open_to_close_1d.toml")
