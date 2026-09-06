from datetime import date

import pandas as pd

from alphaflow.core.config.system_config import StorageConfig
from alphaflow.core.utils.storage import StoragePaths

RIC_COLUMN = "RIC"


class CSVWriter:
    """raw_data/{market}/{data_type}/YYYYMMDD.csv, RIC as primary key.

    depth_intraday is a single file per day carrying a 'timestamp' column for all snapshots -
    the layout is identical, only the row count differs.
    """

    def __init__(self, storage_config: StorageConfig) -> None:
        self.paths = StoragePaths(storage_config)

    def write(self, df: pd.DataFrame, market: str, data_type: str, date: date) -> None:
        if RIC_COLUMN not in df.columns:
            raise ValueError(f"raw data for {market}/{data_type} must carry a {RIC_COLUMN} column; got {list(df.columns)}")
        path = StoragePaths.ensure_parent(self.paths.raw_data(market, data_type, date))
        ordered = [RIC_COLUMN] + [c for c in df.columns if c != RIC_COLUMN]
        df[ordered].to_csv(path, index=False)

    def file_exists(self, market: str, data_type: str, date: date) -> bool:
        return self.paths.raw_data(market, data_type, date).is_file()

    def read(self, market: str, data_type: str, date: date) -> pd.DataFrame:
        path = self.paths.raw_data(market, data_type, date)
        if not path.is_file():
            raise FileNotFoundError(f"no raw data at {path}")
        return pd.read_csv(path)
