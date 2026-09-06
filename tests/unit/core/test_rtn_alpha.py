from datetime import date, datetime

import pandas as pd
import pytest

from alphaflow.core.alpha_base.rtn_alpha import RtnAlpha
from alphaflow.core.config.alpha_config.rtn_config import PricePoint, RtnAlphaConfig
from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.loader import load_alpha_config
from alphaflow.signal_builder.alphas.rtn.open_to_close_1d import OpenToClose1D

MARKET = "HK"
AS_OF = date(2024, 1, 30)
NEXT_DAY = date(2024, 1, 31)


@pytest.fixture
def config():
    return load_alpha_config("configs/alphas/rtn/open_to_close_1d.toml")


def make_config(**overrides):
    base = {"name": "r", "description": "d", "category": "return", "formula": "f",
            "handler_class": "m.C", "alpha_type": "rtn", "market_scope": ["HK"],
            "required_inputs": ["minute_bar"], "entry": PricePoint(price_type="open", session="am"),
            "exit": PricePoint(price_type="close", session="pm"), "horizon": 1, "return_type": ["raw"]}
    return RtnAlphaConfig(**{**base, **overrides})


def bars(prices_by_day):
    """prices_by_day: {day: {RIC: (am_open, pm_close)}}"""
    rows = []
    for day, per_ric in prices_by_day.items():
        for ric, (am_open, pm_close) in per_ric.items():
            rows.append({"RIC": ric, "timestamp": datetime(day.year, day.month, day.day, 9, 30),
                         "open": am_open, "close": am_open, "volume": 100})
            rows.append({"RIC": ric, "timestamp": datetime(day.year, day.month, day.day, 15, 59),
                         "open": pm_close, "close": pm_close, "volume": 100})
    return pd.DataFrame(rows)


class TestForwardReturn:
    def test_raw_return_uses_the_day_horizon_steps_ahead(self):
        alpha = OpenToClose1D(make_config(), None)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 100.0)}, NEXT_DAY: {"A.HK": (100.0, 110.0)}})
        assert alpha.compute(MARKET, AS_OF)["A.HK"] == pytest.approx(0.10)

    def test_horizon_zero_uses_the_same_day(self):
        alpha = OpenToClose1D(make_config(horizon=0), None)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 105.0)}})
        assert alpha.compute(MARKET, AS_OF)["A.HK"] == pytest.approx(0.05)

    def test_hedged_return_is_cross_sectionally_demeaned(self):
        alpha = OpenToClose1D(make_config(return_type=["hedged"]), None)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 100.0), "B.HK": (100.0, 100.0)},
                                    NEXT_DAY: {"A.HK": (100.0, 110.0), "B.HK": (100.0, 90.0)}})
        hedged = alpha.compute(MARKET, AS_OF)
        assert hedged.sum() == pytest.approx(0.0) and hedged["A.HK"] > 0 > hedged["B.HK"]

    def test_requesting_both_return_types_yields_the_hedged_series(self):
        alpha = OpenToClose1D(make_config(return_type=["raw", "hedged"]), None)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 100.0), "B.HK": (100.0, 100.0)},
                                    NEXT_DAY: {"A.HK": (100.0, 110.0), "B.HK": (100.0, 110.0)}})
        assert alpha.compute(MARKET, AS_OF).abs().sum() == pytest.approx(0.0)

    def test_running_past_the_end_of_the_data_is_an_explicit_error(self):
        alpha = OpenToClose1D(make_config(), None)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 100.0)}})
        with pytest.raises(ValueError, match="runs past the end"):
            alpha.compute(MARKET, AS_OF)

    def test_an_as_of_date_absent_from_the_data_is_reported(self):
        alpha = OpenToClose1D(make_config(), None)
        alpha._working_data = bars({NEXT_DAY: {"A.HK": (100.0, 100.0)}})
        with pytest.raises(ValueError, match="absent from working data"):
            alpha.compute(MARKET, AS_OF)

    def test_a_zero_entry_price_drops_the_ric(self):
        alpha = OpenToClose1D(make_config(), None)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 100.0), "Z.HK": (100.0, 100.0)},
                                    NEXT_DAY: {"A.HK": (100.0, 110.0), "Z.HK": (0.0, 110.0)}})
        assert set(alpha.compute(MARKET, AS_OF).index) == {"A.HK"}

    def test_missing_columns_are_reported(self):
        alpha = OpenToClose1D(make_config(), None)
        alpha._working_data = pd.DataFrame({"RIC": ["A.HK"], "close": [1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            alpha.compute(MARKET, AS_OF)


class TestIntradaySessionValidation:
    def test_horizon_zero_rejects_an_exit_before_the_entry(self, system_config):
        config = make_config(horizon=0, entry=PricePoint(price_type="close", session="pm"),
                             exit=PricePoint(price_type="open", session="am"))
        alpha = OpenToClose1D(config, None, market_config=system_config.markets)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 105.0)}})
        with pytest.raises(ConfigValidationError, match="does not precede exit"):
            alpha.run(MARKET, AS_OF)

    def test_horizon_zero_accepts_a_well_ordered_pair(self, system_config):
        alpha = OpenToClose1D(make_config(horizon=0), None, market_config=system_config.markets)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 105.0)}})
        assert alpha.run(MARKET, AS_OF)["A.HK"] == pytest.approx(0.05)

    def test_a_session_is_rejected_for_a_single_session_market(self, system_config):
        config = make_config(horizon=0, market_scope=["JP"])
        alpha = OpenToClose1D(config, None, market_config=system_config.markets)
        alpha._working_data = bars({AS_OF: {"A.T": (100.0, 105.0)}})
        with pytest.raises(ConfigValidationError, match="single-session market"):
            alpha.run("JP", AS_OF)

    def test_a_vwap_window_outside_the_sessions_is_rejected(self, system_config):
        config = make_config(horizon=0, entry=PricePoint(price_type="vwap", vwap_start="06:00", vwap_end="07:00"))
        alpha = OpenToClose1D(config, None, market_config=system_config.markets)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 105.0)}})
        with pytest.raises(ConfigValidationError, match="falls outside HK sessions"):
            alpha.run(MARKET, AS_OF)

    def test_validation_is_skipped_when_horizon_is_positive(self, system_config):
        config = make_config(horizon=1, entry=PricePoint(price_type="close", session="pm"),
                             exit=PricePoint(price_type="open", session="am"))
        alpha = OpenToClose1D(config, None, market_config=system_config.markets)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 100.0)}, NEXT_DAY: {"A.HK": (110.0, 100.0)}})
        assert alpha.run(MARKET, AS_OF)["A.HK"] == pytest.approx(0.10)

    def test_no_market_config_means_no_session_check(self):
        config = make_config(horizon=0, entry=PricePoint(price_type="close", session="pm"),
                             exit=PricePoint(price_type="open", session="am"))
        alpha = OpenToClose1D(config, None, market_config=None)
        alpha._working_data = bars({AS_OF: {"A.HK": (100.0, 105.0)}})
        alpha.run(MARKET, AS_OF)


class TestShippedConfig:
    def test_shipped_alpha_exposes_its_timing(self, config):
        alpha = OpenToClose1D(config, None)
        assert alpha.horizon == 1 and alpha.return_type == ["raw", "hedged"]
        assert isinstance(alpha, RtnAlpha) and alpha.alpha_type == "rtn"
