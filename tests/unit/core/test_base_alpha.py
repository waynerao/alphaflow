from datetime import date

import pandas as pd
import pytest

from alphaflow.core.alpha_base.base_alpha import BaseAlpha
from alphaflow.core.alpha_base.low_alpha import LowAlpha
from alphaflow.core.config.alpha_config.low_config import LowAlphaConfig
from alphaflow.core.config.exceptions import ConfigValidationError

CONFIG_KWARGS = {"name": "test_alpha", "description": "d", "category": "c", "formula": "f",
                 "handler_class": "m.C", "alpha_type": "low", "market_scope": ["HK", "CN"],
                 "required_inputs": ["minute_bar"], "lookback_days": 5}


def make_config(**overrides):
    return LowAlphaConfig(**{**CONFIG_KWARGS, **overrides})


class PassThroughAlpha(LowAlpha):
    def compute(self, market, as_of_date):
        return self.working_data.set_index("RIC")["value"]


class ScaledAlpha(LowAlpha):
    SHIFT = 1.0
    SCALE = 2.0

    def compute(self, market, as_of_date):
        return self.working_data.set_index("RIC")["value"]

    def normalize(self, raw_signal):
        return (raw_signal - self.SHIFT) / self.SCALE


class NotASeriesAlpha(LowAlpha):
    def compute(self, market, as_of_date):
        return {"0700.HK": 1.0}


@pytest.fixture
def working_data():
    return pd.DataFrame({"RIC": ["0700.HK", "0941.HK"], "value": [1.0, 3.0]})


class TestLifecycle:
    def test_working_data_must_be_injected_before_compute(self, system_config):
        alpha = PassThroughAlpha(make_config(), None)
        with pytest.raises(RuntimeError, match="_working_data not injected"):
            alpha.compute("HK", date(2024, 1, 1))

    def test_normalize_defaults_to_identity(self, working_data):
        alpha = PassThroughAlpha(make_config(), None)
        alpha._working_data = working_data
        result = alpha.run("HK", date(2024, 1, 1))
        assert list(result) == [1.0, 3.0]

    def test_run_applies_compute_then_normalize(self, working_data):
        alpha = ScaledAlpha(make_config(), None)
        alpha._working_data = working_data
        assert list(alpha.run("HK", date(2024, 1, 1))) == [0.0, 1.0]

    def test_result_index_is_named_ric(self, working_data):
        alpha = PassThroughAlpha(make_config(), None)
        alpha._working_data = working_data
        assert alpha.run("HK", date(2024, 1, 1)).index.name == "RIC"

    def test_compute_returning_a_non_series_is_rejected(self, working_data):
        alpha = NotASeriesAlpha(make_config(), None)
        alpha._working_data = working_data
        with pytest.raises(TypeError, match="must return a pd.Series"):
            alpha.run("HK", date(2024, 1, 1))


class TestMarketScope:
    def test_market_outside_scope_is_rejected(self, working_data):
        alpha = PassThroughAlpha(make_config(), None)
        alpha._working_data = working_data
        with pytest.raises(ConfigValidationError, match="outside market_scope"):
            alpha.run("JP", date(2024, 1, 1))

    def test_properties_expose_the_config(self):
        alpha = PassThroughAlpha(make_config(), None)
        assert alpha.alpha_id == "test_alpha" and alpha.alpha_type == "low"
        assert alpha.market_scope == ["HK", "CN"] and alpha.required_inputs == ["minute_bar"]


class TestUniverse:
    def test_universe_loads_from_the_market_csv(self):
        universe = PassThroughAlpha(make_config(), None).get_universe("HK")
        assert "0700.HK" in set(universe) and len(universe) > 1

    def test_custom_exclusions_are_removed(self):
        alpha = PassThroughAlpha(make_config(custom_exclusions=["0700.HK"]), None)
        assert "0700.HK" not in set(alpha.get_universe("HK"))

    def test_missing_universe_file_reports_the_path(self):
        with pytest.raises(FileNotFoundError, match="universe file not found"):
            PassThroughAlpha(make_config(market_scope=["HK"]), None).get_universe("XX")


class TestValidateConfig:
    def test_a_type_mismatch_is_caught_on_construction(self):
        class MislabelledAlpha(BaseAlpha):
            EXPECTED_ALPHA_TYPE = "high"

            def compute(self, market, as_of_date):
                return pd.Series(dtype=float)

        with pytest.raises(ConfigValidationError, match="loaded as 'high'"):
            MislabelledAlpha(make_config(), None)
