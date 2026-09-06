from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from alphaflow.core.config.loader import load_alpha_config
from alphaflow.signal_builder.alphas.low.momentum_price_21d import MomentumPrice21D

MARKET = "HK"
AS_OF = date(2024, 1, 31)


@pytest.fixture
def config():
    return load_alpha_config("configs/alphas/low/momentum_price_21d.toml")


def bars(price_paths, days):
    """price_paths: {RIC: [close per day]} - one bar per day is enough for a daily alpha."""
    rows = []
    for ric, prices in price_paths.items():
        for day, price in zip(days, prices):
            rows.append({"RIC": ric, "timestamp": datetime(day.year, day.month, day.day, 15, 59), "close": price})
    return pd.DataFrame(rows)


@pytest.fixture
def days():
    return [AS_OF - timedelta(days=offset) for offset in range(30, -1, -1)]


class TestMomentumCalculation:
    def test_momentum_is_the_ratio_over_the_lookback_skipping_skip_days(self, config, days):
        # 31 days: index -2 is t-1 (skip_days=1), index -23 is t-1-21
        flat = [100.0] * 31
        rising = list(range(100, 131))
        alpha = MomentumPrice21D(config, None)
        alpha._working_data = bars({"FLAT.HK": flat, "RISE.HK": [float(p) for p in rising]}, days)
        raw = alpha.compute(MARKET, AS_OF)
        assert raw["FLAT.HK"] == pytest.approx(0.0)
        assert raw["RISE.HK"] == pytest.approx(129.0 / 108.0 - 1.0)

    def test_too_little_history_is_an_explicit_error(self, config, days):
        alpha = MomentumPrice21D(config, None)
        alpha._working_data = bars({"A.HK": [100.0] * 5}, days[:5])
        with pytest.raises(ValueError, match="need 23 daily closes"):
            alpha.compute(MARKET, AS_OF)

    def test_a_non_positive_base_price_drops_the_ric(self, config, days):
        prices = [0.0] + [100.0] * 30
        alpha = MomentumPrice21D(config, None)
        alpha._working_data = bars({"ZERO.HK": prices, "OK.HK": [100.0] * 31}, days)
        # the zero sits at index 0, outside the 21d window, so both survive here
        assert set(alpha.compute(MARKET, AS_OF).index) == {"ZERO.HK", "OK.HK"}

        prices_in_window = [100.0] * 31
        prices_in_window[-23] = 0.0
        alpha._working_data = bars({"ZERO.HK": prices_in_window, "OK.HK": [100.0] * 31}, days)
        assert set(alpha.compute(MARKET, AS_OF).index) == {"OK.HK"}

    def test_missing_columns_are_reported(self, config):
        alpha = MomentumPrice21D(config, None)
        alpha._working_data = pd.DataFrame({"RIC": ["A.HK"], "close": [1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            alpha.compute(MARKET, AS_OF)

    def test_only_the_last_bar_of_each_day_is_used(self, config, days):
        rows = []
        for day in days:
            for hour, price in [(9, 999.0), (15, 100.0)]:
                rows.append({"RIC": "A.HK", "timestamp": datetime(day.year, day.month, day.day, hour, 30), "close": price})
        alpha = MomentumPrice21D(config, None)
        alpha._working_data = pd.DataFrame(rows)
        assert alpha.compute(MARKET, AS_OF)["A.HK"] == pytest.approx(0.0)


class TestNormalization:
    def test_default_scale_is_the_cross_sectional_std(self, config):
        alpha = MomentumPrice21D(config, None)
        raw = pd.Series({"A.HK": 0.01, "B.HK": 0.03, "C.HK": 0.05})
        assert alpha.normalize(raw).std() == pytest.approx(1.0)

    def test_a_zero_scale_yields_zeros_rather_than_infinities(self, config):
        alpha = MomentumPrice21D(config, None)
        raw = pd.Series({"A.HK": 0.02, "B.HK": 0.02})
        assert list(alpha.normalize(raw)) == [0.0, 0.0]
