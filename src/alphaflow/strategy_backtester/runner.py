from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from alphaflow.core.alpha_base.registry import AlphaRegistry
from alphaflow.core.config.loader import load_system_config
from alphaflow.core.config.paths import DEFAULT_MODELS_DIR, DEFAULT_REGISTRY_CONFIG, DEFAULT_SYSTEM_CONFIG
from alphaflow.core.config.system_config import SystemConfig
from alphaflow.core.data_access.pit_manager import PITDataManager
from alphaflow.core.integrations.logging_setup import setup_logger
from alphaflow.core.utils.dates import to_date_list
from alphaflow.strategy_backtester.engine.backtest_engine import BacktestEngine
from alphaflow.strategy_backtester.reporting.report_generator import ReportGenerator
from alphaflow.strategy_backtester.score_loader import ScoreLoader

log = setup_logger(__name__)


class StrategyBacktesterRunner:
    """Evaluates single-alpha performance against one or more return definitions.

    Read-only: scores must already exist on the shared drive, put there by signal_builder.
    """

    def __init__(self, system_config: SystemConfig, registry: AlphaRegistry) -> None:
        self.system_config = system_config
        self.registry = registry
        self.loader = ScoreLoader(system_config.storage)
        self.reporter = ReportGenerator(system_config.storage)

    @classmethod
    def from_config(cls, system_config_path: str = DEFAULT_SYSTEM_CONFIG, registry_path: str = DEFAULT_REGISTRY_CONFIG,
                    models_dir: str = DEFAULT_MODELS_DIR) -> "StrategyBacktesterRunner":
        system_config = load_system_config(system_config_path)
        registry = AlphaRegistry.from_paths(registry_path, models_dir, system_config=system_config)
        return cls(system_config, registry)

    def run(self, alpha_id: str | list[str], market: str, dates: list[str], rtn_alpha_id: str | list[str],
            overwrite: bool = False, n_jobs: int = 1) -> dict[str, pd.DataFrame]:
        """One result table per alpha: rows are measures, columns are rtn_alpha_ids.
        Multiple alphas fan out across processes when n_jobs > 1."""
        alpha_ids = [alpha_id] if isinstance(alpha_id, str) else list(alpha_id)
        rtn_ids = [rtn_alpha_id] if isinstance(rtn_alpha_id, str) else list(rtn_alpha_id)
        if not dates:
            raise ValueError("dates must not be empty")
        for entry_id in alpha_ids:
            self.registry.require_entry(entry_id)
        for rtn_id in rtn_ids:
            entry = self.registry.require_entry(rtn_id)
            if entry.alpha_type != "rtn":
                raise ValueError(f"{rtn_id!r} has alpha_type {entry.alpha_type!r}; a return definition must be 'rtn'")

        if n_jobs > 1 and len(alpha_ids) > 1:
            return self._run_parallel(alpha_ids, market, dates, rtn_ids, overwrite, n_jobs)
        return {a: self.run_one(a, market, dates, rtn_ids, overwrite) for a in alpha_ids}

    def run_one(self, alpha_id: str, market: str, dates: list[str], rtn_alpha_ids: list[str], overwrite: bool = False) -> pd.DataFrame:
        date_objects = to_date_list(dates)
        alpha_scores = self.loader.load_alpha_scores(market, alpha_id, dates)
        if alpha_scores.empty:
            raise FileNotFoundError(f"no stored scores for {alpha_id} in {market} over the requested dates - run signal_builder first")
        engine = BacktestEngine(PITDataManager.for_research(self.system_config, max(date_objects)))
        results = {}
        for rtn_id in rtn_alpha_ids:
            return_scores = self.loader.load_return_scores(market, rtn_id, dates)
            if return_scores.empty:
                raise FileNotFoundError(f"no stored return scores for {rtn_id} in {market} over the requested dates")
            results[rtn_id] = engine.run(alpha_scores, return_scores, market, date_objects)
            log.info(f"backtested {alpha_id} vs {rtn_id} on {len(date_objects)} date(s)")
        table = self.reporter.assemble(results)
        path = self.reporter.save(table, market, alpha_id, min(date_objects), max(date_objects), overwrite)
        log.info(f"saved backtest result to {path}")
        return table

    def _run_parallel(self, alpha_ids, market, dates, rtn_ids, overwrite, n_jobs) -> dict[str, pd.DataFrame]:
        results = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(self.run_one, a, market, dates, rtn_ids, overwrite): a for a in alpha_ids}
            for future in as_completed(futures):
                alpha = futures[future]
                try:
                    results[alpha] = future.result()
                except Exception as exc:
                    log.error(f"backtest failed for {alpha}: {exc}")
        return results
