from datetime import date

import pandas as pd

from alphaflow.core.alpha_base.high_alpha import HighAlpha

RIC_COLUMN = "RIC"
TIMESTAMP_COLUMN = "timestamp"
LEVEL_COLUMN = "level"
BID_SIZE_COLUMN = "bid_size"
ASK_SIZE_COLUMN = "ask_size"


class OrderImbalance1430(HighAlpha):
    """Order-book size imbalance at a fixed intraday snapshot.

    signal = (sum(bid_size) - sum(ask_size)) / (sum(bid_size) + sum(ask_size))
    summed over the top config.depth_levels levels, bounded in [-1, 1].
    """

    SHIFT: float = 0.0
    SCALE: float | None = 1.0     # already bounded, so no rescaling by default

    def compute(self, market: str, as_of_date: date) -> pd.Series:
        snapshot = self._snapshot_rows(as_of_date)
        depth = snapshot[snapshot[LEVEL_COLUMN] <= self.config.depth_levels] if LEVEL_COLUMN in snapshot.columns else snapshot
        sizes = depth.groupby(RIC_COLUMN)[[BID_SIZE_COLUMN, ASK_SIZE_COLUMN]].sum()
        total = sizes[BID_SIZE_COLUMN] + sizes[ASK_SIZE_COLUMN]
        # A RIC with no resting size has no imbalance to speak of - exclude rather than zero it
        signal = ((sizes[BID_SIZE_COLUMN] - sizes[ASK_SIZE_COLUMN]) / total.where(total > 0)).dropna()
        signal.name = self.alpha_id
        return signal

    def normalize(self, raw_signal: pd.Series) -> pd.Series:
        scale = self.SCALE if self.SCALE is not None else raw_signal.std()
        if not scale or pd.isna(scale):
            return pd.Series(0.0, index=raw_signal.index, name=raw_signal.name)
        return (raw_signal - self.SHIFT) / scale

    def _snapshot_rows(self, as_of_date: date) -> pd.DataFrame:
        """The last book state at or before config.snapshot_time on as_of_date."""
        df = self.working_data
        missing = [c for c in (RIC_COLUMN, BID_SIZE_COLUMN, ASK_SIZE_COLUMN) if c not in df.columns]
        if missing:
            raise ValueError(f"{self.alpha_id}: working data missing columns {missing}")
        if TIMESTAMP_COLUMN not in df.columns:
            return df    # single-snapshot file - nothing to select
        frame = df.copy()
        stamps = pd.to_datetime(frame[TIMESTAMP_COLUMN])
        cutoff_hour, cutoff_minute = (int(p) for p in self.config.snapshot_time.split(":"))
        cutoff = stamps.dt.normalize() + pd.Timedelta(hours=cutoff_hour, minutes=cutoff_minute)
        frame = frame[stamps <= cutoff]
        if frame.empty:
            raise ValueError(f"{self.alpha_id}: no book data at or before {self.config.snapshot_time} on {as_of_date}")
        frame["_stamp"] = pd.to_datetime(frame[TIMESTAMP_COLUMN])
        latest = frame.groupby(RIC_COLUMN)["_stamp"].transform("max")
        return frame[frame["_stamp"] == latest].drop(columns=["_stamp"])
