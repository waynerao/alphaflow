from datetime import date

import pandas as pd

from alphaflow.core.config.system_config import StorageConfig
from alphaflow.signal_builder.storage.score_writer import ScoreWriter


class RealizedReturnLoader:
    """Realised side: the model's rtn_alpha_id, read from the same score store."""

    def __init__(self, storage_config: StorageConfig) -> None:
        self.writer = ScoreWriter(storage_config)

    def load(self, market: str, rtn_alpha_id: str, dates: list[str | date]) -> pd.DataFrame:
        return self.writer.read_dates(market, rtn_alpha_id, dates)
