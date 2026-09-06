from datetime import date

import pandas as pd

from alphaflow.core.config.system_config import KdbConfig
from alphaflow.core.data_access.exceptions import DataNotFoundError, DataSourceError
from alphaflow.core.integrations.desktool_adapter import DeskToolProtocol, get_desktool

RIC_COLUMN = "RIC"
TIMESTAMP_COLUMNS = ("timestamp", "date", "datetime")


class KDBClient:
    """kdb+ reads, routed through apcr_desktool (decision D13 - no pykx). All outputs RIC-keyed.

    Table names come from [kdb.tables] in system.toml so the real kdb+ layout is a config
    change rather than a code change.
    """

    def __init__(self, config: KdbConfig, desktool: DeskToolProtocol | None = None) -> None:
        self.config = config
        self._dt = desktool if desktool is not None else get_desktool()

    def query(self, table: str, rics: list[str], fields: list[str], start_date: date, as_of_date: date) -> pd.DataFrame:
        """Rows with timestamp <= as_of_date. The PIT cut is re-applied locally in
        PITDataManager too - this is belt and braces, since a source-side filter is only
        as trustworthy as the source."""
        if not rics:
            raise ValueError("rics must not be empty")
        try:
            df = self._dt.query_kdb(table=table, rics=rics, fields=fields, start_date=start_date, end_date=as_of_date)
        except Exception as exc:
            raise DataSourceError(f"kdb query failed for table {table!r}: {exc}") from exc
        return self._normalize(df, table)

    def query_latest(self, table: str, rics: list[str], fields: list[str]) -> pd.DataFrame:
        """Production path - newest available row per RIC, no PIT filter."""
        if not rics:
            raise ValueError("rics must not be empty")
        try:
            df = self._dt.query_kdb_latest(table=table, rics=rics, fields=fields)
        except Exception as exc:
            raise DataSourceError(f"kdb latest-query failed for table {table!r}: {exc}") from exc
        return self._normalize(df, table)

    @classmethod
    def _normalize(cls, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """Guarantee the RIC contract regardless of how the source spells its symbol column."""
        if df is None or df.empty:
            raise DataNotFoundError(f"kdb table {table!r} returned no rows")
        out = df.copy()
        if RIC_COLUMN not in out.columns:
            renamed = next((c for c in out.columns if c.lower() in ("ric", "sym", "symbol")), None)
            if renamed is None:
                raise DataSourceError(f"kdb table {table!r} returned no RIC column; got {list(out.columns)}")
            out = out.rename(columns={renamed: RIC_COLUMN})
        return out.reset_index(drop=True)

    @classmethod
    def timestamp_column(cls, df: pd.DataFrame) -> str | None:
        return next((c for c in TIMESTAMP_COLUMNS if c in df.columns), None)
