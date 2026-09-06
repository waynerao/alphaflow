from datetime import date

import pandas as pd

from alphaflow.core.alpha_base.low_alpha import LowAlpha

RIC_COLUMN = "RIC"
CLOSE_COLUMN = "close"
TIMESTAMP_COLUMN = "timestamp"


class MomentumPrice21D(LowAlpha):
    """21-day price momentum, skipping the most recent day to sidestep short-term reversal.

    signal = close(t - skip) / close(t - skip - lookback) - 1
    """

    SHIFT: float = 0.0
    SCALE: float | None = None    # None -> use the cross-sectional std of the day

    def compute(self, market: str, as_of_date: date) -> pd.Series:
        prices = self._daily_close()
        needed = self.config.lookback_days + self.config.skip_days + 1
        if len(prices) < needed:
            raise ValueError(f"{self.alpha_id}: need {needed} daily closes, working data has {len(prices)}")
        recent = prices.iloc[-(self.config.skip_days + 1)]
        base = prices.iloc[-(self.config.skip_days + self.config.lookback_days + 1)]
        # Non-positive base prices make the ratio meaningless, so those RICs drop out
        signal = (recent / base.where(base > 0) - 1.0).dropna()
        signal.name = self.alpha_id
        return signal

    def normalize(self, raw_signal: pd.Series) -> pd.Series:
        scale = self.SCALE if self.SCALE is not None else raw_signal.std()
        if not scale or pd.isna(scale):
            return pd.Series(0.0, index=raw_signal.index, name=raw_signal.name)
        return (raw_signal - self.SHIFT) / scale

    def _daily_close(self) -> pd.DataFrame:
        """Collapse minute bars to a date x RIC close matrix, oldest row first."""
        df = self.working_data
        missing = [c for c in (RIC_COLUMN, CLOSE_COLUMN, TIMESTAMP_COLUMN) if c not in df.columns]
        if missing:
            raise ValueError(f"{self.alpha_id}: working data missing columns {missing}")
        frame = df[[TIMESTAMP_COLUMN, RIC_COLUMN, CLOSE_COLUMN]].copy()
        frame["_day"] = pd.to_datetime(frame[TIMESTAMP_COLUMN]).dt.date
        last_of_day = frame.sort_values(TIMESTAMP_COLUMN).groupby(["_day", RIC_COLUMN], as_index=False)[CLOSE_COLUMN].last()
        return last_of_day.pivot(index="_day", columns=RIC_COLUMN, values=CLOSE_COLUMN).sort_index()
