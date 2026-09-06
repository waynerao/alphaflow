import pytest

from alphaflow.core.config.alpha_config.rtn_config import PricePoint, RtnAlphaConfig
from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.loader import load_alpha_config, load_model_config, load_model_configs
from alphaflow.core.config.model_config import ModelConfig
from alphaflow.core.config.optimizer_config import MeanVarianceConfig
from alphaflow.core.config.paths import find_project_root, resolve_config_path, resolve_shared_drive
from alphaflow.core.config.registry_config import RegistryConfig
from alphaflow.core.config.system_config import SessionConfig, SystemConfig

MODEL_KWARGS = {"model_id": "m", "market": "HK", "alpha_ids": ["a", "b"], "rtn_alpha_id": "r",
                "start_date": "20240101", "end_date": "20241231"}


class TestSystemConfig:
    def test_ships_all_seven_markets(self, system_config):
        assert sorted(system_config.markets) == ["AU", "CN", "HK", "JP", "KR", "SG", "TW"]

    def test_kdb_table_lookup(self, system_config):
        assert system_config.kdb.table_for("minute_bar")
        with pytest.raises(KeyError, match="no kdb table mapped"):
            system_config.kdb.table_for("not_a_data_type")

    def test_s3_key_substitution(self, system_config):
        assert system_config.s3.key_for("southbound", "HK", "20240115") == "southbound/HK/20240115.csv"

    def test_unknown_market_rejected(self, system_config):
        with pytest.raises(KeyError, match="not configured"):
            system_config.market("XX")

    def test_extra_top_level_key_rejected(self, system_config):
        payload = system_config.model_dump()
        payload["surprise"] = 1
        with pytest.raises(Exception):
            SystemConfig(**payload)

    @pytest.mark.parametrize("start,end", [("9:30", "12:00"), ("09:30", "24:01"), ("09:xx", "12:00")])
    def test_malformed_session_times_rejected(self, start, end):
        with pytest.raises(Exception, match="HH:MM"):
            SessionConfig(start=start, end=end)

    def test_session_end_must_follow_start(self):
        with pytest.raises(Exception, match="must precede"):
            SessionConfig(start="12:00", end="09:30")


class TestModelConfig:
    def test_weights_must_match_alpha_ids(self):
        with pytest.raises(Exception, match="must match alpha_ids"):
            ModelConfig(**MODEL_KWARGS, alpha_weights={"a": 0.5, "c": 0.5})

    def test_empty_weights_allowed_before_optimization(self):
        assert ModelConfig(**MODEL_KWARGS).is_optimized is False

    def test_matching_weights_accepted(self):
        model = ModelConfig(**MODEL_KWARGS, alpha_weights={"a": 0.6, "b": 0.4})
        assert model.is_optimized and model.weight("a") == 0.6 and model.weight("missing") == 0.0

    def test_model_id_must_match_filename(self, tmp_path):
        path = tmp_path / "other_name.toml"
        path.write_text('model_id = "m"\nmarket = "HK"\nalpha_ids = ["a"]\nrtn_alpha_id = "r"\n'
                        'start_date = "20240101"\nend_date = "20241231"\n')
        with pytest.raises(ConfigValidationError, match="does not match filename stem"):
            load_model_config(path)

    def test_end_date_must_follow_start(self):
        with pytest.raises(Exception, match="precedes start_date"):
            ModelConfig(**{**MODEL_KWARGS, "start_date": "20241231", "end_date": "20240101"})

    def test_bad_date_format_rejected(self):
        with pytest.raises(Exception):
            ModelConfig(**{**MODEL_KWARGS, "start_date": "2024-01-01"})

    def test_duplicate_alpha_ids_rejected(self):
        with pytest.raises(Exception, match="duplicate"):
            ModelConfig(**{**MODEL_KWARGS, "alpha_ids": ["a", "a"]})

    def test_templates_are_skipped_when_loading_a_directory(self):
        assert "hk_momentum" in load_model_configs()


class TestRegistryConfig:
    def test_lookups(self, registry_config):
        assert registry_config.get_by_id("momentum_price_21d").alpha_type == "low"
        assert registry_config.get_by_id("nope") is None
        assert [e.alpha_id for e in registry_config.get_by_type("rtn")] == ["open_to_close_1d"]

    def test_duplicate_alpha_ids_rejected(self):
        entry = {"alpha_id": "x", "alpha_type": "low", "config_file": "f.toml", "active": True}
        with pytest.raises(Exception, match="duplicate alpha_id"):
            RegistryConfig(alphas=[entry, dict(entry)])


class TestAlphaConfigs:
    def test_shipped_alpha_configs_load_with_the_right_class(self):
        assert type(load_alpha_config("configs/alphas/low/momentum_price_21d.toml")).__name__ == "LowAlphaConfig"
        assert type(load_alpha_config("configs/alphas/high/order_imbalance_1430.toml")).__name__ == "HighAlphaConfig"
        assert type(load_alpha_config("configs/alphas/rtn/open_to_close_1d.toml")).__name__ == "RtnAlphaConfig"

    @pytest.mark.parametrize("alpha_type", ["high", "mid", "low", "rtn"])
    def test_every_template_is_valid(self, alpha_type):
        assert load_alpha_config(f"configs/alphas/{alpha_type}/_template.toml").alpha_type == alpha_type

    def test_type_mismatch_is_rejected(self):
        with pytest.raises(ConfigValidationError, match="but was loaded as"):
            load_alpha_config("configs/alphas/low/momentum_price_21d.toml", "high")

    def test_rtn_timing_section_is_flattened(self):
        config = load_alpha_config("configs/alphas/rtn/open_to_close_1d.toml")
        assert config.horizon == 1 and config.return_type == ["raw", "hedged"]
        assert config.entry.session == "am" and config.exit.session == "pm"

    def test_missing_file_reports_the_path(self):
        with pytest.raises(ConfigValidationError, match="config file not found"):
            load_alpha_config("configs/alphas/low/does_not_exist.toml")

    def test_malformed_toml_is_reported(self, tmp_path):
        path = tmp_path / "bad.toml"
        path.write_text("[identity\nname = broken")
        with pytest.raises(ConfigValidationError, match="malformed TOML"):
            load_alpha_config(path)


class TestPricePoint:
    def test_vwap_requires_a_window(self):
        with pytest.raises(Exception, match="requires both vwap_start and vwap_end"):
            PricePoint(price_type="vwap")

    def test_vwap_window_must_be_ordered(self):
        with pytest.raises(Exception, match="must precede"):
            PricePoint(price_type="vwap", vwap_start="15:00", vwap_end="10:00")

    def test_vwap_fields_rejected_for_open_and_close(self):
        with pytest.raises(Exception, match="only valid for price_type='vwap'"):
            PricePoint(price_type="close", vwap_start="10:00", vwap_end="11:00")

    @pytest.mark.parametrize("bad", ["mid", "OPEN", ""])
    def test_unknown_price_type_rejected(self, bad):
        with pytest.raises(Exception, match="price_type must be one of"):
            PricePoint(price_type=bad)

    def test_unknown_session_rejected(self):
        with pytest.raises(Exception, match="session must be one of"):
            PricePoint(price_type="open", session="evening")


class TestRtnAlphaConfig:
    def _kwargs(self, **overrides):
        base = {"name": "r", "description": "d", "category": "return", "formula": "f",
                "handler_class": "m.C", "alpha_type": "rtn", "market_scope": ["HK"],
                "required_inputs": ["minute_bar"], "entry": PricePoint(price_type="open"),
                "exit": PricePoint(price_type="close"), "horizon": 1, "return_type": ["raw"]}
        return {**base, **overrides}

    def test_negative_horizon_rejected(self):
        with pytest.raises(Exception, match="horizon must be >= 0"):
            RtnAlphaConfig(**self._kwargs(horizon=-1))

    def test_horizon_zero_is_intraday(self):
        assert RtnAlphaConfig(**self._kwargs(horizon=0)).is_intraday

    def test_unknown_return_type_rejected(self):
        with pytest.raises(Exception, match="return_type entries must be in"):
            RtnAlphaConfig(**self._kwargs(return_type=["raw", "gross"]))

    def test_duplicate_return_type_rejected(self):
        with pytest.raises(Exception, match="duplicate entries in return_type"):
            RtnAlphaConfig(**self._kwargs(return_type=["raw", "raw"]))


class TestOptimizerConfig:
    def test_min_above_max_rejected(self):
        with pytest.raises(Exception, match="exceeds max"):
            MeanVarianceConfig(min_single_alpha_weight=0.9, max_single_alpha_weight=0.5)

    def test_feasibility_check(self):
        assert MeanVarianceConfig(max_single_alpha_weight=0.5).feasible_for(2) is True
        assert MeanVarianceConfig(max_single_alpha_weight=0.2).feasible_for(3) is False
        assert MeanVarianceConfig(min_single_alpha_weight=0.4, max_single_alpha_weight=0.9).feasible_for(3) is False


class TestPaths:
    def test_project_root_has_a_pyproject(self):
        assert (find_project_root() / "pyproject.toml").is_file()

    def test_relative_config_paths_anchor_to_the_root_not_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path("configs/system.toml").is_file()

    def test_absolute_paths_pass_through(self, tmp_path):
        assert resolve_config_path(tmp_path / "x.toml") == tmp_path / "x.toml"

    def test_env_var_overrides_the_configured_shared_drive(self, monkeypatch):
        monkeypatch.setenv("ALPHAFLOW_SHARED_DRIVE", "/somewhere/else")
        assert str(resolve_shared_drive("data")) == "/somewhere/else"
