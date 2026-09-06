"""Shared-drive layout (spec section 2.3).

Files under shared_drive_path are how packages hand data to each other, so every path is
built here rather than string-formatted at each call site.

    raw_data/<MARKET>/<data_type>/YYYYMMDD.csv
    alpha_scores/<MARKET>/<alpha_id>/YYYYMMDD.parquet
    alpha_scores/<model_id>/YYYYMMDD.parquet
    backtest_results/<MARKET>/<alpha_id>_YYYYMMDD-YYYYMMDD.parquet
    optimizer_results/<MARKET>/<model_id>.parquet
    post_trade_results/<model_id>/YYYYMMDD.parquet
"""

from datetime import date
from pathlib import Path

from alphaflow.core.config.paths import resolve_shared_drive
from alphaflow.core.config.system_config import StorageConfig
from alphaflow.core.utils.dates import to_datestr

RAW_DATA_DIR = "raw_data"
ALPHA_SCORES_DIR = "alpha_scores"
BACKTEST_RESULTS_DIR = "backtest_results"
OPTIMIZER_RESULTS_DIR = "optimizer_results"
POST_TRADE_RESULTS_DIR = "post_trade_results"


class StoragePaths:
    def __init__(self, storage_config: StorageConfig) -> None:
        self.config = storage_config
        self.root = resolve_shared_drive(storage_config.shared_drive_path)

    def raw_data(self, market: str, data_type: str, date_value: str | date) -> Path:
        return self.root / RAW_DATA_DIR / market / data_type / f"{to_datestr(date_value)}.csv"

    def raw_data_dir(self, market: str, data_type: str) -> Path:
        return self.root / RAW_DATA_DIR / market / data_type

    def alpha_score(self, market: str, alpha_id: str, date_value: str | date) -> Path:
        return self.root / ALPHA_SCORES_DIR / market / alpha_id / f"{to_datestr(date_value)}.parquet"

    def alpha_score_dir(self, market: str, alpha_id: str) -> Path:
        return self.root / ALPHA_SCORES_DIR / market / alpha_id

    def model_score(self, model_id: str, date_value: str | date) -> Path:
        return self.root / ALPHA_SCORES_DIR / model_id / f"{to_datestr(date_value)}.parquet"

    def backtest_result(self, market: str, alpha_id: str, start: str | date, end: str | date) -> Path:
        return self.root / BACKTEST_RESULTS_DIR / market / f"{alpha_id}_{to_datestr(start)}-{to_datestr(end)}.parquet"

    def optimizer_result(self, market: str, model_id: str) -> Path:
        return self.root / OPTIMIZER_RESULTS_DIR / market / f"{model_id}.parquet"

    def post_trade_result(self, model_id: str, date_value: str | date) -> Path:
        return self.root / POST_TRADE_RESULTS_DIR / model_id / f"{to_datestr(date_value)}.parquet"

    @classmethod
    def ensure_parent(cls, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
