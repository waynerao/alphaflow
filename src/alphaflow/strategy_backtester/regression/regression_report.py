import pandas as pd

from alphaflow.strategy_backtester.regression.cross_sectional_ols import OLS_MEASURES


class RegressionReport:
    """Formats CrossSectionalOLS output for the measures x rtn_alpha_id result table."""

    @classmethod
    def to_series(cls, results: dict[str, float]) -> pd.Series:
        return pd.Series({measure: results.get(measure) for measure in OLS_MEASURES}, dtype=float)

    @classmethod
    def to_frame(cls, results_by_rtn: dict[str, dict[str, float]]) -> pd.DataFrame:
        return pd.DataFrame({rtn_id: cls.to_series(values) for rtn_id, values in results_by_rtn.items()})
