from datetime import date

import numpy as np
import pandas as pd

from alphaflow.core.data_access.exceptions import DataNotFoundError, DataSourceError
from alphaflow.core.integrations.logging_setup import setup_logger
from alphaflow.strategy_backtester.score_loader import DATE_COLUMN, RIC_COLUMN

BARRA_PREFIX = "barra_"
# Columns that identify a row rather than describe a factor. Everything else in the risk
# frame is treated as a factor, so no factor list is hardcoded anywhere (decision D7).
NON_FACTOR_COLUMNS = {RIC_COLUMN, DATE_COLUMN, "timestamp", "datetime", "model", "market", "currency", "asof_date"}
log = setup_logger(__name__)


def discover_factors(risk_frame: pd.DataFrame) -> list[str]:
    """Every numeric, non-identifying column is a Barra factor - style and industry alike."""
    return [c for c in risk_frame.columns
            if c not in NON_FACTOR_COLUMNS and pd.api.types.is_numeric_dtype(risk_frame[c])]


class BarraExposure:
    """Score-weighted mean factor exposure, averaged over the period.

    exposure_f = mean over dates of  sum_i(w_i * exposure_i,f),  w = |score| normalised to 1.
    Absolute scores are used as weights so that a short leg contributes exposure rather than
    cancelling the long leg out.
    """

    def __init__(self, data) -> None:
        self.data = data

    def compute(self, alpha_scores: pd.DataFrame, market: str, dates: list[date]) -> dict[str, float]:
        if alpha_scores.empty or not dates:
            return {}
        rics = sorted(alpha_scores[RIC_COLUMN].astype(str).unique())
        try:
            risk = self.data.get_risk_factors(rics, market, min(dates))
        except (DataNotFoundError, DataSourceError, KeyError, NotImplementedError) as exc:
            log.warning(f"Barra exposures unavailable for {market}: {exc}")
            return {}
        factors = discover_factors(risk)
        if not factors:
            log.warning(f"risk frame for {market} carries no factor columns; got {list(risk.columns)}")
            return {}
        return self._weighted_exposures(alpha_scores, risk, factors)

    @classmethod
    def _weighted_exposures(cls, alpha_scores: pd.DataFrame, risk: pd.DataFrame, factors: list[str]) -> dict[str, float]:
        keys = [RIC_COLUMN, DATE_COLUMN] if DATE_COLUMN in risk.columns and DATE_COLUMN in alpha_scores.columns else [RIC_COLUMN]
        merged = alpha_scores.merge(risk[keys + factors], on=keys, how="inner")
        if merged.empty:
            return {}
        per_day = []
        grouper = merged.groupby(DATE_COLUMN) if DATE_COLUMN in merged.columns else [(None, merged)]
        for _, group in grouper:
            weights = group["score"].abs()
            total = weights.sum()
            if not total:
                continue
            per_day.append((group[factors].mul(weights / total, axis=0)).sum())
        if not per_day:
            return {}
        averaged = pd.concat(per_day, axis=1).mean(axis=1)
        return {f"{BARRA_PREFIX}{factor}": float(value) for factor, value in averaged.items() if not np.isnan(value)}
