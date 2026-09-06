from datetime import datetime

import pandas as pd

from alphaflow.core.config.production_config import ProductionOutputConfig
from alphaflow.core.data_access.kdb_client import KDBClient
from alphaflow.core.integrations.logging_setup import setup_logger

SCORE_SCHEMA = ["timestamp", "market", "alpha_id", "RIC", "raw_signal", "score", "frequency"]
FREQUENCY_DAILY = "daily"
FREQUENCY_INTRADAY = "intraday"
log = setup_logger(__name__)


class KDBWriter:
    """Persists individual alpha scores to kdb+.

    Individual scores only. Composites are never written here - the Consumer builds them on
    demand, so that N models sharing an alpha still cost one computation and one row set.
    """

    def __init__(self, output_config: ProductionOutputConfig, kdb_client: KDBClient) -> None:
        self.config = output_config
        self.client = kdb_client

    def write_individual_scores(self, market: str, daily_scores: dict[str, pd.Series],
                                intraday_scores: dict[str, pd.Series]) -> pd.DataFrame:
        """Returns the rows written, so a caller (or a test) can inspect them."""
        stamp = datetime.now()
        rows = self._to_rows(market, daily_scores, FREQUENCY_DAILY, stamp) + \
            self._to_rows(market, intraday_scores, FREQUENCY_INTRADAY, stamp)
        if not rows:
            return pd.DataFrame(columns=SCORE_SCHEMA)
        frame = pd.DataFrame(rows, columns=SCORE_SCHEMA)
        for frequency, table in ((FREQUENCY_DAILY, self.config.kdb_table_daily), (FREQUENCY_INTRADAY, self.config.kdb_table_intraday)):
            subset = frame[frame["frequency"] == frequency]
            if not subset.empty:
                self._insert(table, subset)
        return frame

    def _insert(self, table: str, frame: pd.DataFrame) -> None:
        insert = getattr(self.client, "insert", None)
        if insert is None:
            # KDBClient is read-only in the spec; the write path lands via desktool on-prem
            desktool_insert = getattr(getattr(self.client, "_dt", None), "insert_kdb", None)
            if desktool_insert is None:
                log.warning(f"no kdb insert available - {len(frame)} row(s) for {table!r} not persisted")
                return
            desktool_insert(table=table, df=frame)
            return
        insert(table, frame)

    @classmethod
    def _to_rows(cls, market: str, scores: dict[str, pd.Series], frequency: str, stamp: datetime) -> list[dict]:
        rows = []
        for alpha_id, series in (scores or {}).items():
            if series is None or series.empty:
                continue
            for ric, value in series.items():
                rows.append({"timestamp": stamp, "market": market, "alpha_id": alpha_id, "RIC": str(ric),
                             "raw_signal": float(value), "score": float(value), "frequency": frequency})
        return rows

    def query_latest_scores(self, market: str, alpha_ids: list[str]) -> dict[str, pd.Series]:
        """Newest score per (alpha_id, RIC). What the Consumer reads."""
        frames = {}
        for table in (self.config.kdb_table_daily, self.config.kdb_table_intraday):
            try:
                df = self.client.query_latest(table, alpha_ids, SCORE_SCHEMA)
            except Exception as exc:
                log.warning(f"could not read latest scores from {table!r}: {exc}")
                continue
            if df is None or df.empty:
                continue
            for alpha_id, group in df[df["alpha_id"].isin(alpha_ids)].groupby("alpha_id"):
                latest = group.sort_values("timestamp").groupby("RIC")["score"].last()
                frames[alpha_id] = latest.astype(float)
        return frames
