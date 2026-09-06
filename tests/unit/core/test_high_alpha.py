from datetime import date, datetime

import pandas as pd
import pytest

from alphaflow.core.config.loader import load_alpha_config
from alphaflow.signal_builder.alphas.high.order_imbalance_1430 import OrderImbalance1430

MARKET = "HK"
AS_OF = date(2024, 1, 31)


@pytest.fixture
def config():
    return load_alpha_config("configs/alphas/high/order_imbalance_1430.toml")


def book(rows):
    return pd.DataFrame([{"RIC": r, "timestamp": t, "level": lv, "bid_size": b, "ask_size": a}
                         for r, t, lv, b, a in rows])


class TestImbalance:
    def test_balanced_book_scores_zero(self, config):
        alpha = OrderImbalance1430(config, None)
        alpha._working_data = book([("A.HK", datetime(2024, 1, 31, 14, 0), 1, 100, 100)])
        assert alpha.compute(MARKET, AS_OF)["A.HK"] == pytest.approx(0.0)

    def test_all_bid_scores_plus_one_all_ask_minus_one(self, config):
        alpha = OrderImbalance1430(config, None)
        alpha._working_data = book([("BID.HK", datetime(2024, 1, 31, 14, 0), 1, 100, 0),
                                    ("ASK.HK", datetime(2024, 1, 31, 14, 0), 1, 0, 100)])
        result = alpha.compute(MARKET, AS_OF)
        assert result["BID.HK"] == pytest.approx(1.0) and result["ASK.HK"] == pytest.approx(-1.0)

    def test_only_levels_up_to_depth_levels_contribute(self, config):
        # depth_levels = 5, so the level-6 row must be ignored
        rows = [("A.HK", datetime(2024, 1, 31, 14, 0), lv, 100, 100) for lv in range(1, 6)]
        rows.append(("A.HK", datetime(2024, 1, 31, 14, 0), 6, 10_000, 0))
        alpha = OrderImbalance1430(config, None)
        alpha._working_data = book(rows)
        assert alpha.compute(MARKET, AS_OF)["A.HK"] == pytest.approx(0.0)

    def test_a_ric_with_no_resting_size_is_dropped_not_zeroed(self, config):
        alpha = OrderImbalance1430(config, None)
        alpha._working_data = book([("EMPTY.HK", datetime(2024, 1, 31, 14, 0), 1, 0, 0),
                                    ("OK.HK", datetime(2024, 1, 31, 14, 0), 1, 100, 50)])
        assert set(alpha.compute(MARKET, AS_OF).index) == {"OK.HK"}


class TestSnapshotSelection:
    def test_rows_after_the_snapshot_time_are_excluded(self, config):
        alpha = OrderImbalance1430(config, None)
        alpha._working_data = book([("A.HK", datetime(2024, 1, 31, 14, 29), 1, 100, 0),
                                    ("A.HK", datetime(2024, 1, 31, 14, 31), 1, 0, 100)])
        assert alpha.compute(MARKET, AS_OF)["A.HK"] == pytest.approx(1.0)

    def test_the_latest_row_at_or_before_the_cutoff_wins(self, config):
        alpha = OrderImbalance1430(config, None)
        alpha._working_data = book([("A.HK", datetime(2024, 1, 31, 10, 0), 1, 0, 100),
                                    ("A.HK", datetime(2024, 1, 31, 14, 30), 1, 100, 0)])
        assert alpha.compute(MARKET, AS_OF)["A.HK"] == pytest.approx(1.0)

    def test_no_data_before_the_cutoff_is_an_explicit_error(self, config):
        alpha = OrderImbalance1430(config, None)
        alpha._working_data = book([("A.HK", datetime(2024, 1, 31, 15, 0), 1, 100, 0)])
        with pytest.raises(ValueError, match="no book data at or before 14:30"):
            alpha.compute(MARKET, AS_OF)

    def test_a_file_without_timestamps_is_treated_as_a_single_snapshot(self, config):
        alpha = OrderImbalance1430(config, None)
        alpha._working_data = pd.DataFrame({"RIC": ["A.HK"], "level": [1], "bid_size": [150], "ask_size": [50]})
        assert alpha.compute(MARKET, AS_OF)["A.HK"] == pytest.approx(0.5)


class TestNormalization:
    def test_bounded_signal_is_left_alone_by_default(self, config):
        alpha = OrderImbalance1430(config, None)
        raw = pd.Series({"A.HK": 0.5, "B.HK": -0.25})
        assert list(alpha.normalize(raw)) == [0.5, -0.25]
