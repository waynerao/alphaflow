from datetime import date

import pandas as pd

from alphaflow.core.data_access.kdb_client import KDBClient
from alphaflow.data_dumper.extractors.base_extractor import BaseExtractor

# Fields requested per data_type. Placeholders alongside [kdb.tables] - correct both together
# once the real kdb+ schema is known.
KDB_FIELDS = {
    "depth_open": ["timestamp", "level", "bid_price", "bid_size", "ask_price", "ask_size"],
    "depth_close": ["timestamp", "level", "bid_price", "bid_size", "ask_price", "ask_size"],
    "depth_intraday": ["timestamp", "level", "bid_price", "bid_size", "ask_price", "ask_size"],
    "indic_open": ["timestamp", "indic_price", "indic_size"],
    "indic_close": ["timestamp", "indic_price", "indic_size"],
    "minute_bar": ["timestamp", "open", "high", "low", "close", "volume"],
}


class KDBExtractor(BaseExtractor):
    """depth_open, depth_close, depth_intraday, indic_open, indic_close, minute_bar."""

    def __init__(self, kdb_client: KDBClient) -> None:
        self.client = kdb_client

    @property
    def supported_data_types(self) -> list[str]:
        return list(KDB_FIELDS)

    def extract(self, data_type: str, market: str, rics: list[str], date: date) -> pd.DataFrame:
        if not self.supports(data_type):
            raise ValueError(f"KDBExtractor does not handle {data_type!r}; supported: {self.supported_data_types}")
        table = self.client.config.table_for(data_type)
        # A dump is a single day, so start and as-of are the same date
        return self.client.query(table, rics, KDB_FIELDS[data_type], date, date)
