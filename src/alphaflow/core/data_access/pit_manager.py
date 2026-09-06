from datetime import date, datetime

import pandas as pd

from alphaflow.core.config.system_config import SystemConfig
from alphaflow.core.data_access.bloomberg_client import BBGClient
from alphaflow.core.data_access.exceptions import DataNotFoundError, PITViolationError
from alphaflow.core.data_access.kdb_client import KDBClient
from alphaflow.core.data_access.s3_client import S3Client
from alphaflow.core.integrations.desktool_adapter import DeskToolProtocol
from alphaflow.core.utils.dates import to_datestr

RIC_COLUMN = "RIC"
PIT_TIMESTAMP_COLUMNS = ("timestamp", "datetime", "date", "asof_date")


class PITMode:
    RESEARCH = "research"        # as_of_date is a past date - strict filtering enforced
    PRODUCTION = "production"    # as_of_date = now() - no filtering, minimal latency


class PITDataManager:
    """Central data gateway. Every data read in the system goes through this class, and
    every frame it returns is RIC-keyed.

    In RESEARCH mode any row stamped after as_of_date is a lookahead bug, so the cut is
    re-applied here on top of whatever the source did. In PRODUCTION mode nothing is
    filtered - the point is latency, and 'now' is the as-of date by definition.
    """

    def __init__(self, system_config: SystemConfig, as_of_date: date | None = None, mode: str = PITMode.PRODUCTION,
                 desktool: DeskToolProtocol | None = None) -> None:
        if mode not in (PITMode.RESEARCH, PITMode.PRODUCTION):
            raise ValueError(f"mode must be {PITMode.RESEARCH!r} or {PITMode.PRODUCTION!r}, got {mode!r}")
        if mode == PITMode.RESEARCH and as_of_date is None:
            raise ValueError("research mode requires an explicit as_of_date")
        self.system_config = system_config
        self.mode = mode
        self.as_of_date = as_of_date if as_of_date is not None else date.today()
        self._kdb = KDBClient(system_config.kdb, desktool=desktool)
        self._bbg = BBGClient(system_config.bloomberg, desktool=desktool)
        self._s3 = S3Client(system_config.s3)

    @classmethod
    def for_research(cls, system_config: SystemConfig, as_of_date: date, desktool: DeskToolProtocol | None = None) -> "PITDataManager":
        return cls(system_config, as_of_date=as_of_date, mode=PITMode.RESEARCH, desktool=desktool)

    @classmethod
    def for_production(cls, system_config: SystemConfig, desktool: DeskToolProtocol | None = None) -> "PITDataManager":
        return cls(system_config, as_of_date=None, mode=PITMode.PRODUCTION, desktool=desktool)

    @property
    def is_research(self) -> bool:
        return self.mode == PITMode.RESEARCH

    # -- kdb-backed getters -------------------------------------------------------------

    def get_ohlcv(self, rics: list[str], market: str, start_date: date, fields: list[str] | None = None) -> pd.DataFrame:
        return self._kdb_read("ohlcv", rics, market, start_date, fields or ["open", "high", "low", "close", "volume"])

    def get_tick(self, rics: list[str], market: str, start_date: date) -> pd.DataFrame:
        return self._kdb_read("tick", rics, market, start_date, ["timestamp", "price", "size"])

    def get_depth(self, rics: list[str], market: str, start_date: date, snapshot: str) -> pd.DataFrame:
        data_type = f"depth_{snapshot}"
        self._check_available(market, data_type)
        levels = self.system_config.kdb.tables
        if data_type not in levels:
            raise KeyError(f"no kdb table mapped for {data_type!r}")
        return self._kdb_read(data_type, rics, market, start_date, ["timestamp", "bid_price", "bid_size", "ask_price", "ask_size", "level"])

    def get_indic(self, rics: list[str], market: str, start_date: date, snapshot: str) -> pd.DataFrame:
        return self._kdb_read(f"indic_{snapshot}", rics, market, start_date, ["timestamp", "indic_price", "indic_size"])

    def get_minute_bar(self, rics: list[str], market: str, start_date: date) -> pd.DataFrame:
        return self._kdb_read("minute_bar", rics, market, start_date, ["timestamp", "open", "high", "low", "close", "volume"])

    def get_risk_factors(self, rics: list[str], market: str, start_date: date, model: str = "ASE2S") -> pd.DataFrame:
        """Barra exposures. Factor columns are whatever the risk model returns - nothing
        here hardcodes a factor list (decision D7)."""
        df = self._kdb_read("risk_factors", rics, market, start_date, [], check_available=False)
        return df.loc[df["model"] == model].reset_index(drop=True) if "model" in df.columns else df

    def _kdb_read(self, data_type: str, rics: list[str], market: str, start_date: date, fields: list[str],
                  check_available: bool = True) -> pd.DataFrame:
        if check_available:
            self._check_available(market, data_type)
        table = self.system_config.kdb.table_for(data_type)
        if self.is_research:
            df = self._kdb.query(table, rics, fields, start_date, self.as_of_date)
        else:
            df = self._kdb.query_latest(table, rics, fields)
        return self._apply_pit(df, f"kdb:{table}")

    # -- Bloomberg / S3 -----------------------------------------------------------------

    def get_bbg_reference(self, rics: list[str], fields: list[str]) -> pd.DataFrame:
        return self._apply_pit(self._bbg.bdp(rics, fields), "bbg:bdp")

    def get_bbg_history(self, rics: list[str], fields: list[str], start_date: date) -> pd.DataFrame:
        return self._apply_pit(self._bbg.bdh(rics, fields, start_date, self.as_of_date), "bbg:bdh")

    def get_s3_data(self, key: str, file_type: str = "csv") -> pd.DataFrame:
        if file_type not in ("csv", "parquet"):
            raise ValueError(f"file_type must be 'csv' or 'parquet', got {file_type!r}")
        reader = self._s3.read_csv if file_type == "csv" else self._s3.read_parquet
        return self._apply_pit(reader(key, self.as_of_date), f"s3:{key}")

    def s3_key(self, data_type: str, market: str, as_of: date | None = None) -> str:
        return self.system_config.s3.key_for(data_type, market, to_datestr(as_of or self.as_of_date))

    # -- internals ----------------------------------------------------------------------

    def _check_available(self, market: str, data_type: str) -> None:
        available = self.system_config.available_data_types(market)
        if data_type not in available:
            raise KeyError(f"data_type {data_type!r} is not available for {market}; configured: {available}")

    def _apply_pit(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Research mode drops - and complains about - anything stamped after as_of_date."""
        if df is None or df.empty:
            raise DataNotFoundError(f"{source} returned no rows")
        if not self.is_research:
            return df
        column = next((c for c in PIT_TIMESTAMP_COLUMNS if c in df.columns), None)
        if column is None:
            return df
        stamps = pd.to_datetime(df[column], errors="coerce")
        cutoff = pd.Timestamp(datetime.combine(self.as_of_date, datetime.max.time()))
        violating = stamps > cutoff
        if violating.any():
            raise PITViolationError(f"{source} returned {int(violating.sum())} row(s) stamped after as_of_date {self.as_of_date}")
        return df
