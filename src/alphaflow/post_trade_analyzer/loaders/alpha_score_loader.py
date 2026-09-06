from datetime import date

import pandas as pd

from alphaflow.core.config.system_config import StorageConfig
from alphaflow.signal_builder.storage.score_writer import ScoreWriter


class AlphaScoreLoader:
    """Predicted side: the alpha scores that were published for each date."""

    def __init__(self, storage_config: StorageConfig) -> None:
        self.writer = ScoreWriter(storage_config)

    def load(self, market: str, alpha_id: str, dates: list[str | date]) -> pd.DataFrame:
        return self.writer.read_dates(market, alpha_id, dates)
