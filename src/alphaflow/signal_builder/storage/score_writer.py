from datetime import date

import pandas as pd

from alphaflow.core.config.system_config import StorageConfig
from alphaflow.core.utils.dates import to_date
from alphaflow.core.utils.storage import StoragePaths

RIC_COLUMN = "RIC"
SCORE_COLUMNS = [RIC_COLUMN, "raw_signal", "score"]
DATE_COLUMN = "date"


class ScoreWriter:
    """alpha_scores/{market}/{alpha_id}/YYYYMMDD.parquet with columns RIC, raw_signal, score.

    The date is in the filename, not the file - a single day's frame carries no date column.
    read_range() adds one, because there the date is what distinguishes the rows.
    """

    def __init__(self, storage_config: StorageConfig) -> None:
        self.paths = StoragePaths(storage_config)

    def write(self, scores: pd.DataFrame, market: str, alpha_id: str, date: date) -> None:
        missing = [c for c in SCORE_COLUMNS if c not in scores.columns]
        if missing:
            raise ValueError(f"score frame for {alpha_id} missing columns {missing}; got {list(scores.columns)}")
        path = StoragePaths.ensure_parent(self.paths.alpha_score(market, alpha_id, date))
        scores[SCORE_COLUMNS].to_parquet(path, index=False)

    def file_exists(self, market: str, alpha_id: str, date: date) -> bool:
        return self.paths.alpha_score(market, alpha_id, date).is_file()

    def read(self, market: str, alpha_id: str, date: date) -> pd.DataFrame:
        path = self.paths.alpha_score(market, alpha_id, date)
        if not path.is_file():
            raise FileNotFoundError(f"no scores at {path}")
        return pd.read_parquet(path)

    def read_range(self, market: str, alpha_id: str, start: date, end: date) -> pd.DataFrame:
        """Every stored day between start and end inclusive, stacked with a date column.
        Absent days are simply not present - a gap is not an error here."""
        start_d, end_d = to_date(start), to_date(end)
        directory = self.paths.alpha_score_dir(market, alpha_id)
        if not directory.is_dir():
            return pd.DataFrame(columns=[DATE_COLUMN, *SCORE_COLUMNS])
        frames = []
        for path in sorted(directory.glob("*.parquet")):
            try:
                day = to_date(path.stem)
            except ValueError:
                continue
            if start_d <= day <= end_d:
                frame = pd.read_parquet(path)
                frame.insert(0, DATE_COLUMN, day)
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[DATE_COLUMN, *SCORE_COLUMNS])

    def read_dates(self, market: str, alpha_id: str, dates: list[str | date]) -> pd.DataFrame:
        """Explicit date list rather than a range - what the backtester and optimizer want."""
        frames = []
        for value in dates:
            day = to_date(value)
            path = self.paths.alpha_score(market, alpha_id, day)
            if not path.is_file():
                continue
            frame = pd.read_parquet(path)
            frame.insert(0, DATE_COLUMN, day)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[DATE_COLUMN, *SCORE_COLUMNS])

    def available_dates(self, market: str, alpha_id: str) -> list[str]:
        directory = self.paths.alpha_score_dir(market, alpha_id)
        return sorted(p.stem for p in directory.glob("*.parquet")) if directory.is_dir() else []
