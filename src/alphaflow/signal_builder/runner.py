from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd

from alphaflow.core.alpha_base.base_alpha import BaseAlpha
from alphaflow.core.alpha_base.loader import AlphaLoader
from alphaflow.core.alpha_base.registry import AlphaRegistry
from alphaflow.core.config.loader import load_system_config
from alphaflow.core.config.paths import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_CONFIG,
    DEFAULT_SYSTEM_CONFIG,
    DEFAULT_UNIVERSE_DIR,
)
from alphaflow.core.config.system_config import SystemConfig
from alphaflow.core.data_access.pit_manager import PITDataManager
from alphaflow.core.integrations.logging_setup import setup_logger
from alphaflow.core.utils.dates import to_date, to_datestr
from alphaflow.core.utils.storage import StoragePaths
from alphaflow.signal_builder.storage.score_writer import ScoreWriter

RIC_COLUMN = "RIC"
ALL_ALPHAS = "all"
SOURCE_RAW_DATA = "raw_data"
SOURCE_LIVE = "live"

# How many calendar days of history to load behind as_of_date, per alpha type. Low alphas
# need their whole lookback window; intraday alphas only need the day itself.
LOOKBACK_BUFFER_DAYS = 7
log = setup_logger(__name__)


class SignalBuilderRunner:
    """Computes alpha signal values.

    build_one_day_alpha() is the single code path: research notebooks and the production
    Producer both call it, differing only in save= and source=.

      source="raw_data" - inputs come from the data_dumper CSVs on the shared drive
      source="live"     - inputs come straight from PITDataManager

    The live path exists because the Producer recomputes intraday alphas every few seconds,
    and a once-a-day CSV dump cannot be fresh (see PLAN.md decision D9).
    """

    def __init__(self, system_config: SystemConfig, registry: AlphaRegistry, universe_dir: str = DEFAULT_UNIVERSE_DIR) -> None:
        self.system_config = system_config
        self.registry = registry
        self.universe_dir = universe_dir
        self.paths = StoragePaths(system_config.storage)
        self.score_writer = ScoreWriter(system_config.storage)

    @classmethod
    def from_config(cls, system_config_path: str = DEFAULT_SYSTEM_CONFIG, registry_path: str = DEFAULT_REGISTRY_CONFIG,
                    models_dir: str = DEFAULT_MODELS_DIR, universe_dir: str = DEFAULT_UNIVERSE_DIR) -> "SignalBuilderRunner":
        system_config = load_system_config(system_config_path)
        registry = AlphaRegistry.from_paths(registry_path, models_dir, system_config=system_config)
        return cls(system_config, registry, universe_dir)

    # -- public API ---------------------------------------------------------------------

    def build_one_day_alpha(self, alpha_id: str | list[str], market: str, date: str, save: bool = False,
                            overwrite: bool = False, source: str = SOURCE_RAW_DATA) -> dict[str, pd.DataFrame]:
        """Per alpha: load required_inputs, inner-join on RIC, drop nulls, inject as
        _working_data, run(), then optionally persist.

        Returns {alpha_id: DataFrame(RIC, raw_signal, score)} - no date column, since the
        date is fixed and lives in the filename.
        """
        as_of_date = to_date(date)
        results: dict[str, pd.DataFrame] = {}
        for entry_id in self._resolve_alpha_ids(alpha_id, market):
            alpha = self._build_alpha(entry_id, as_of_date, source)
            if market not in alpha.market_scope:
                log.info(f"skip {entry_id} - {market} outside market_scope {alpha.market_scope}")
                continue
            if save and not overwrite and self.score_writer.file_exists(market, entry_id, as_of_date):
                log.info(f"skip {entry_id} {market} {to_datestr(as_of_date)} - scores already present")
                results[entry_id] = self.score_writer.read(market, entry_id, as_of_date)
                continue
            working = self._load_and_join(alpha, market, as_of_date, source)
            alpha._working_data = working
            raw_signal = alpha.compute(market, as_of_date)
            scores = self._to_score_frame(raw_signal, alpha)
            if save:
                self.score_writer.write(scores, market, entry_id, as_of_date)
                log.info(f"wrote scores {entry_id} {market} {to_datestr(as_of_date)} ({len(scores)} RICs)")
            results[entry_id] = scores
        return results

    def build_multi_day_alpha(self, alpha_id: str | list[str], market: str, dates: list[str], save: bool = False,
                              overwrite: bool = False, n_jobs: int = 1, source: str = SOURCE_RAW_DATA) -> dict[str, pd.DataFrame]:
        """build_one_day_alpha across an explicit date list, stacked with a date column.
        n_jobs > 1 fans out one process per date."""
        if not dates:
            raise ValueError("dates must not be empty")
        per_date = self._run_dates_parallel(alpha_id, market, dates, save, overwrite, source, n_jobs) if n_jobs > 1 \
            else {d: self.build_one_day_alpha(alpha_id, market, d, save, overwrite, source) for d in dates}

        stacked: dict[str, list[pd.DataFrame]] = {}
        for date_str in dates:
            for entry_id, frame in (per_date.get(date_str) or {}).items():
                dated = frame.copy()
                dated.insert(0, "date", to_date(date_str))
                stacked.setdefault(entry_id, []).append(dated)
        return {entry_id: pd.concat(frames, ignore_index=True) for entry_id, frames in stacked.items()}

    # -- internals ----------------------------------------------------------------------

    def _run_dates_parallel(self, alpha_id, market, dates, save, overwrite, source, n_jobs) -> dict[str, dict[str, pd.DataFrame]]:
        """One process per date. Failures are logged and the date dropped, so one bad day
        does not take down a multi-year build."""
        results: dict[str, dict[str, pd.DataFrame]] = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(self.build_one_day_alpha, alpha_id, market, d, save, overwrite, source): d for d in dates}
            for future in as_completed(futures):
                date_str = futures[future]
                try:
                    results[date_str] = future.result()
                except Exception as exc:
                    log.error(f"date {date_str} failed: {exc}")
        return results

    def _resolve_alpha_ids(self, alpha_id: str | list[str], market: str) -> list[str]:
        if alpha_id == ALL_ALPHAS:
            return [e.alpha_id for e in self.registry.get_active()]
        requested = [alpha_id] if isinstance(alpha_id, str) else list(alpha_id)
        for entry_id in requested:
            self.registry.require_entry(entry_id)
        return requested

    def _build_alpha(self, alpha_id: str, as_of_date: date, source: str) -> BaseAlpha:
        entry = self.registry.require_entry(alpha_id)
        # Research reads a dated snapshot; live reads whatever kdb has right now
        data = PITDataManager.for_production(self.system_config) if source == SOURCE_LIVE \
            else PITDataManager.for_research(self.system_config, as_of_date)
        alpha = AlphaLoader.load(entry, data, self.universe_dir, market_config=self.system_config.markets)
        if entry.alpha_type == "rtn" and source != SOURCE_LIVE and alpha.horizon > 0:
            # A forward return signalled on t is only observable once t+horizon has traded.
            # Extending the PIT cut is correct here, not a lookahead: this is the label, not
            # a predictor, and the backtester lines it up against scores from t.
            alpha.data = PITDataManager.for_research(self.system_config, as_of_date + timedelta(days=alpha.horizon + LOOKBACK_BUFFER_DAYS))
        return alpha

    def _load_and_join(self, alpha: BaseAlpha, market: str, as_of_date: date, source: str) -> pd.DataFrame:
        """The implicit universe: inner-join every required_input on RIC and drop null rows.
        Whichever RICs survive are the ones the alpha scores."""
        if source not in (SOURCE_RAW_DATA, SOURCE_LIVE):
            raise ValueError(f"source must be {SOURCE_RAW_DATA!r} or {SOURCE_LIVE!r}, got {source!r}")
        frames = [self._load_input(alpha, market, as_of_date, data_type, source) for data_type in alpha.required_inputs]
        joined = frames[0]
        for frame in frames[1:]:
            overlap = [c for c in frame.columns if c in joined.columns and c != RIC_COLUMN]
            joined = joined.merge(frame.drop(columns=overlap), on=RIC_COLUMN, how="inner")
        universe = set(alpha.get_universe(market))
        joined = joined[joined[RIC_COLUMN].isin(universe)]
        joined = joined.dropna()
        if joined.empty:
            raise ValueError(f"{alpha.alpha_id}: no RICs survive the join of {alpha.required_inputs} for {market} {to_datestr(as_of_date)}")
        return joined.reset_index(drop=True)

    def _load_input(self, alpha: BaseAlpha, market: str, as_of_date: date, data_type: str, source: str) -> pd.DataFrame:
        if source == SOURCE_LIVE:
            return self._load_live(alpha, market, as_of_date, data_type)
        return self._load_raw_csv(alpha, market, as_of_date, data_type)

    def _load_raw_csv(self, alpha: BaseAlpha, market: str, as_of_date: date, data_type: str) -> pd.DataFrame:
        """Daily CSVs from data_dumper. Low alphas need history, so several days are stacked."""
        days = self._history_days(alpha, as_of_date)
        frames = []
        for day in days:
            path = self.paths.raw_data(market, data_type, day)
            if path.is_file():
                frames.append(pd.read_csv(path))
        if not frames:
            raise FileNotFoundError(f"{alpha.alpha_id}: no {data_type} CSV for {market} in {days[0]}..{days[-1]}")
        combined = pd.concat(frames, ignore_index=True)
        if RIC_COLUMN not in combined.columns:
            raise ValueError(f"{market}/{data_type} raw data has no {RIC_COLUMN} column; got {list(combined.columns)}")
        return combined

    def _load_live(self, alpha: BaseAlpha, market: str, as_of_date: date, data_type: str) -> pd.DataFrame:
        """Straight from PITDataManager - the Producer's path."""
        rics = alpha.get_universe(market).tolist()
        start = self._history_days(alpha, as_of_date)[0]
        data = alpha.data
        if data_type.startswith("depth_"):
            return data.get_depth(rics, market, start, data_type.removeprefix("depth_"))
        if data_type.startswith("indic_"):
            return data.get_indic(rics, market, start, data_type.removeprefix("indic_"))
        if data_type == "minute_bar":
            return data.get_minute_bar(rics, market, start)
        if data_type == "ohlcv":
            return data.get_ohlcv(rics, market, start)
        if data_type == "tick":
            return data.get_tick(rics, market, start)
        if data_type in ("intropic", "southbound"):
            return data.get_s3_data(data.s3_key(data_type, market, as_of_date))
        raise ValueError(f"{alpha.alpha_id}: no live loader for data_type {data_type!r}")

    @classmethod
    def _history_days(cls, alpha: BaseAlpha, as_of_date: date) -> list[date]:
        """Calendar days to load, oldest first.

        Low alphas get their whole lookback plus a buffer for non-trading days. Rtn alphas
        reach forward instead, because the return signalled on t settles at t+horizon.
        Everything else needs only the day itself.
        """
        if alpha.alpha_type == "low":
            span = getattr(alpha.config, "lookback_days", 0) + getattr(alpha.config, "skip_days", 0) + LOOKBACK_BUFFER_DAYS
            return [as_of_date - timedelta(days=offset) for offset in range(span, -1, -1)]
        if alpha.alpha_type == "rtn":
            horizon = getattr(alpha.config, "horizon", 0)
            if horizon > 0:
                return [as_of_date + timedelta(days=offset) for offset in range(horizon + LOOKBACK_BUFFER_DAYS + 1)]
        return [as_of_date]

    @classmethod
    def _to_score_frame(cls, raw_signal: pd.Series, alpha: BaseAlpha) -> pd.DataFrame:
        """raw_signal is compute()'s output; score is normalize()'s. Both are kept so a
        normalisation change can be audited without recomputing."""
        normalized = alpha.normalize(raw_signal)
        frame = pd.DataFrame({RIC_COLUMN: raw_signal.index.astype(str), "raw_signal": raw_signal.to_numpy(),
                              "score": normalized.reindex(raw_signal.index).to_numpy()})
        return frame.reset_index(drop=True)
