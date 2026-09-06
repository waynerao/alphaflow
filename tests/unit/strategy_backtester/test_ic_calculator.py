from datetime import date

import numpy as np
import pandas as pd
import pytest

from alphaflow.strategy_backtester.metrics.ic_calculator import ICCalculator

DAYS = [date(2024, 1, d) for d in range(1, 6)]
RICS = [f"{i:04d}.HK" for i in range(10)]


def frames(pairs_by_day):
    """pairs_by_day: {day: (scores, returns)} as equal-length sequences over RICS."""
    alpha_rows, return_rows = [], []
    for day, (scores, returns) in pairs_by_day.items():
        for ric, s, r in zip(RICS, scores, returns):
            alpha_rows.append({"date": day, "RIC": ric, "raw_signal": s, "score": s})
            return_rows.append({"date": day, "RIC": ric, "raw_signal": r, "score": r})
    return pd.DataFrame(alpha_rows), pd.DataFrame(return_rows)


class TestDailyIC:
    def test_a_perfectly_ranked_day_scores_one(self):
        values = list(range(10))
        alpha, returns = frames({DAYS[0]: (values, values)})
        assert ICCalculator().compute_daily_ic(alpha, returns).iloc[0] == pytest.approx(1.0)

    def test_a_perfectly_inverted_day_scores_minus_one(self):
        values = list(range(10))
        alpha, returns = frames({DAYS[0]: (values, values[::-1])})
        assert ICCalculator().compute_daily_ic(alpha, returns).iloc[0] == pytest.approx(-1.0)

    def test_it_is_a_rank_correlation_not_a_linear_one(self):
        values = list(range(10))
        monotone = [v ** 3 for v in values]
        alpha, returns = frames({DAYS[0]: (values, monotone)})
        assert ICCalculator().compute_daily_ic(alpha, returns).iloc[0] == pytest.approx(1.0)

    def test_one_series_per_date(self):
        values = list(range(10))
        alpha, returns = frames({day: (values, values) for day in DAYS})
        assert len(ICCalculator().compute_daily_ic(alpha, returns)) == len(DAYS)

    def test_a_constant_score_day_is_dropped_not_scored_zero(self):
        alpha, returns = frames({DAYS[0]: ([1.0] * 10, list(range(10))), DAYS[1]: (list(range(10)), list(range(10)))})
        result = ICCalculator().compute_daily_ic(alpha, returns)
        assert list(result.index) == [DAYS[1]]

    def test_a_day_with_too_few_names_is_dropped(self):
        alpha = pd.DataFrame([{"date": DAYS[0], "RIC": "A.HK", "raw_signal": 1.0, "score": 1.0},
                              {"date": DAYS[0], "RIC": "B.HK", "raw_signal": 2.0, "score": 2.0}])
        returns = alpha.copy()
        assert ICCalculator().compute_daily_ic(alpha, returns).empty

    def test_unmatched_rics_are_excluded_by_the_join(self):
        alpha, returns = frames({DAYS[0]: (list(range(10)), list(range(10)))})
        returns = returns[returns["RIC"] != RICS[0]]
        assert ICCalculator().compute_daily_ic(alpha, returns).iloc[0] == pytest.approx(1.0)

    def test_no_overlap_yields_an_empty_series(self):
        alpha, returns = frames({DAYS[0]: (list(range(10)), list(range(10)))})
        returns["RIC"] = "OTHER.HK"
        assert ICCalculator().compute_daily_ic(alpha, returns).empty


class TestSummary:
    def test_icir_is_mean_over_std(self):
        series = pd.Series([0.1, 0.2, 0.3], index=DAYS[:3])
        summary = ICCalculator().summarize(series)
        assert summary["mean_IC"] == pytest.approx(0.2)
        assert summary["std_IC"] == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1))
        assert summary["ICIR"] == pytest.approx(summary["mean_IC"] / summary["std_IC"])

    def test_a_single_day_has_no_dispersion(self):
        summary = ICCalculator().summarize(pd.Series([0.1], index=DAYS[:1]))
        assert summary["mean_IC"] == 0.1 and np.isnan(summary["std_IC"]) and np.isnan(summary["ICIR"])

    def test_an_empty_series_yields_nans(self):
        assert all(np.isnan(v) for v in ICCalculator().summarize(pd.Series(dtype=float)).values())

    def test_a_constant_ic_series_has_undefined_icir(self):
        assert np.isnan(ICCalculator().summarize(pd.Series([0.2, 0.2, 0.2]))["ICIR"])
