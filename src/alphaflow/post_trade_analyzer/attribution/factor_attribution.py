from datetime import date

import numpy as np
import pandas as pd

from alphaflow.core.data_access.exceptions import DataNotFoundError, DataSourceError
from alphaflow.core.integrations.logging_setup import setup_logger
from alphaflow.strategy_backtester.risk.barra_exposure import BARRA_PREFIX, discover_factors

RIC_COLUMN = "RIC"
DATE_COLUMN = "date"
DIFFERENCE_COLUMN = "difference"
COEFFICIENT_SUFFIX = "_coefficient"
MIN_OBSERVATIONS = 10
# An OLS with as many columns as rows fits perfectly and means nothing, so require a margin
# of observations over factors on top of the absolute floor.
MIN_OBSERVATIONS_PER_FACTOR = 5
log = setup_logger(__name__)


class FactorAttribution:
    """Explains the gap between what an alpha predicted and what actually happened.

        difference(RIC, t) = realized_return(RIC, t) - alpha_score(RIC, t)

    then regresses that pooled across all (RIC, date) on the Barra factors. Barra industry
    factors double as the sector breakdown, so no separate sector mapping is needed.
    """

    def __init__(self, data=None) -> None:
        self.data = data

    def compute_difference(self, alpha_scores: pd.DataFrame, realized_returns: pd.DataFrame) -> pd.DataFrame:
        """Inner join on (date, RIC) - a name only contributes where both sides exist."""
        for name, frame in (("alpha_scores", alpha_scores), ("realized_returns", realized_returns)):
            missing = [c for c in (DATE_COLUMN, RIC_COLUMN, "score") if c not in frame.columns]
            if missing:
                raise ValueError(f"{name} missing columns {missing}; got {list(frame.columns)}")
        predicted = alpha_scores[[DATE_COLUMN, RIC_COLUMN, "score"]].rename(columns={"score": "predicted"})
        realized = realized_returns[[DATE_COLUMN, RIC_COLUMN, "score"]].rename(columns={"score": "realized"})
        merged = predicted.merge(realized, on=[DATE_COLUMN, RIC_COLUMN], how="inner").dropna()
        merged[DIFFERENCE_COLUMN] = merged["realized"] - merged["predicted"]
        return merged.reset_index(drop=True)

    def summarize_difference(self, difference_df: pd.DataFrame) -> dict[str, float]:
        if difference_df.empty:
            return {"mean_diff": np.nan, "std_diff": np.nan}
        return {"mean_diff": float(difference_df[DIFFERENCE_COLUMN].mean()),
                "std_diff": float(difference_df[DIFFERENCE_COLUMN].std(ddof=1)) if len(difference_df) > 1 else np.nan}

    def run_regression(self, difference_df: pd.DataFrame, market: str, dates: list[date]) -> dict[str, float]:
        """Pooled OLS of difference on every Barra factor. Returns one
        {factor}_coefficient per factor, plus r_squared, n_obs and intercept."""
        empty = {"r_squared": np.nan, "n_obs": float(len(difference_df)), "intercept": np.nan}
        if difference_df.empty or self.data is None:
            return empty
        rics = sorted(difference_df[RIC_COLUMN].astype(str).unique())
        try:
            risk = self.data.get_risk_factors(rics, market, min(dates))
        except (DataNotFoundError, DataSourceError, KeyError, NotImplementedError) as exc:
            log.warning(f"Barra factors unavailable for {market}: {exc}")
            return empty
        factors = discover_factors(risk)
        if not factors:
            log.warning(f"risk frame for {market} carries no factor columns")
            return empty

        keys = [RIC_COLUMN, DATE_COLUMN] if DATE_COLUMN in risk.columns else [RIC_COLUMN]
        merged = difference_df.merge(risk[keys + factors], on=keys, how="inner").dropna(subset=factors + [DIFFERENCE_COLUMN])
        usable = [f for f in factors if merged[f].nunique() > 1]
        required = max(MIN_OBSERVATIONS, len(usable) * MIN_OBSERVATIONS_PER_FACTOR)
        if len(merged) < required or not usable:
            log.warning(f"too few usable observations to attribute {market}: {len(merged)} rows for {len(usable)} "
                        f"varying factor(s), need {required}")
            return {**empty, "n_obs": float(len(merged))}

        import statsmodels.api as sm
        design = sm.add_constant(merged[usable].astype(float), has_constant="add")
        if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
            # Collinear factors would still "fit", with coefficients that are not identified
            log.warning(f"rank-deficient design for {market}: {len(usable)} factor(s) are collinear, skipping attribution")
            return {**empty, "n_obs": float(len(merged))}
        model = sm.OLS(merged[DIFFERENCE_COLUMN].astype(float), design).fit()
        results = {f"{BARRA_PREFIX}{factor}{COEFFICIENT_SUFFIX}": float(model.params[factor]) for factor in usable}
        results.update({"r_squared": float(model.rsquared), "n_obs": float(len(merged)), "intercept": float(model.params["const"])})
        return results
