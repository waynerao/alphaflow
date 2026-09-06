from datetime import date
from pathlib import Path

import pandas as pd

from alphaflow.core.config.loader import load_system_config
from alphaflow.core.config.paths import DEFAULT_SYSTEM_CONFIG, DEFAULT_UNIVERSE_DIR, resolve_config_path
from alphaflow.core.config.system_config import SystemConfig
from alphaflow.core.data_access.bloomberg_client import BBGClient
from alphaflow.core.data_access.exceptions import DataNotFoundError, DataSourceError
from alphaflow.core.data_access.kdb_client import KDBClient
from alphaflow.core.data_access.s3_client import S3Client
from alphaflow.core.integrations.logging_setup import setup_logger
from alphaflow.core.utils.dates import date_range, to_date, to_datestr
from alphaflow.data_dumper.extractors.bbg_extractor import BBGExtractor
from alphaflow.data_dumper.extractors.kdb_extractor import KDBExtractor
from alphaflow.data_dumper.extractors.s3_extractor import S3Extractor
from alphaflow.data_dumper.writers.csv_writer import CSVWriter

RIC_COLUMN = "RIC"
log = setup_logger(__name__)


class DataDumperRunner:
    """Manual backfill - pulls raw data and writes daily CSVs to the shared drive.

    Deliberately not scheduled: this is a research/backfill tool, run by hand.
    """

    def __init__(self, system_config: SystemConfig, universe_dir: str = DEFAULT_UNIVERSE_DIR) -> None:
        self.system_config = system_config
        self.universe_dir = universe_dir
        self.writer = CSVWriter(system_config.storage)
        self.extractors = [
            KDBExtractor(KDBClient(system_config.kdb)),
            S3Extractor(S3Client(system_config.s3)),
            BBGExtractor(BBGClient(system_config.bloomberg)),
        ]

    @classmethod
    def from_config(cls, system_config_path: str = DEFAULT_SYSTEM_CONFIG, universe_dir: str = DEFAULT_UNIVERSE_DIR) -> "DataDumperRunner":
        return cls(load_system_config(system_config_path), universe_dir)

    def run(self, market: str, date: str | None = None, start: str | None = None, end: str | None = None,
            source: str | None = None, overwrite: bool = False) -> dict[str, list[str]]:
        """source=None dumps every available_data_type for the market.

        Returns {data_type: [YYYYMMDD written]}. One failing (data_type, date) is logged and
        skipped rather than aborting the backfill - a missing day should not cost the rest.
        """
        dates = self._resolve_dates(date, start, end)
        data_types = self._resolve_data_types(market, source)
        rics = self.load_universe(market)
        log.info(f"dumping {market}: {len(data_types)} data_type(s) x {len(dates)} date(s) x {len(rics)} RIC(s)")

        written: dict[str, list[str]] = {dt: [] for dt in data_types}
        for data_type in data_types:
            extractor = self._extractor_for(data_type)
            for day in dates:
                if not overwrite and self.writer.file_exists(market, data_type, day):
                    log.info(f"skip {market}/{data_type}/{to_datestr(day)} - already present")
                    continue
                try:
                    df = extractor.extract(data_type, market, rics, day)
                except NotImplementedError:
                    log.warning(f"skip {market}/{data_type} - extractor not implemented")
                    break
                except (DataNotFoundError, DataSourceError, ValueError, KeyError) as exc:
                    log.error(f"failed {market}/{data_type}/{to_datestr(day)}: {exc}")
                    continue
                self.writer.write(df, market, data_type, day)
                written[data_type].append(to_datestr(day))
                log.info(f"wrote {market}/{data_type}/{to_datestr(day)} ({len(df)} rows)")
        return written

    def load_universe(self, market: str) -> list[str]:
        path = Path(resolve_config_path(self.universe_dir)) / f"universe_{market}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"universe file not found: {path}")
        universe = pd.read_csv(path)
        if RIC_COLUMN not in universe.columns:
            raise ValueError(f"{path} must have a '{RIC_COLUMN}' column, got {list(universe.columns)}")
        return universe[RIC_COLUMN].dropna().astype(str).str.strip().drop_duplicates().tolist()

    @classmethod
    def _resolve_dates(cls, date_value: str | None, start: str | None, end: str | None) -> list[date]:
        if date_value is not None:
            if start is not None or end is not None:
                raise ValueError("pass either date= or start=/end=, not both")
            return [to_date(date_value)]
        if start is None or end is None:
            raise ValueError("pass date=, or both start= and end=")
        return date_range(start, end)

    def _resolve_data_types(self, market: str, source: str | None) -> list[str]:
        available = self.system_config.available_data_types(market)
        if source is None:
            return list(available)
        if source not in available:
            raise ValueError(f"data_type {source!r} not available for {market}; configured: {available}")
        return [source]

    def _extractor_for(self, data_type: str):
        extractor = next((e for e in self.extractors if e.supports(data_type)), None)
        if extractor is None:
            raise ValueError(f"no extractor handles data_type {data_type!r}")
        return extractor
