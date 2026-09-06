import pandas as pd

from alphaflow.core.alpha_base.registry import AlphaRegistry
from alphaflow.core.config.loader import load_optimizer_config, load_system_config
from alphaflow.core.config.optimizer_config import OptimizerConfig
from alphaflow.core.config.paths import (
    DEFAULT_MODELS_DIR,
    DEFAULT_OPTIMIZER_CONFIG,
    DEFAULT_REGISTRY_CONFIG,
    DEFAULT_SYSTEM_CONFIG,
)
from alphaflow.core.config.system_config import SystemConfig
from alphaflow.core.data_access.pit_manager import PITDataManager
from alphaflow.core.integrations.logging_setup import setup_logger
from alphaflow.core.utils.dates import date_range, to_datestr
from alphaflow.optimizer.analysis.correlation import CorrelationAnalyzer
from alphaflow.optimizer.analysis.standalone_metrics import StandaloneMetrics
from alphaflow.optimizer.optimizers.mean_variance_optimizer import MeanVarianceOptimizer
from alphaflow.optimizer.returns.strategy_return_builder import StrategyReturnBuilder
from alphaflow.optimizer.risk.barra_decomposer import BarraDecomposer
from alphaflow.optimizer.storage.model_config_writer import ModelConfigWriter
from alphaflow.optimizer.storage.result_writer import ResultWriter
from alphaflow.strategy_backtester.score_loader import ScoreLoader

log = setup_logger(__name__)


class OptimizerRunner:
    """Allocates weights across the alphas in a model by mean-variance optimisation.

    Each alpha is treated as an independent strategy with its own daily return series;
    correlation enters through the covariance matrix rather than as an exclusion rule.
    """

    def __init__(self, system_config: SystemConfig, optimizer_config: OptimizerConfig, registry: AlphaRegistry,
                 models_dir: str = DEFAULT_MODELS_DIR) -> None:
        self.system_config = system_config
        self.optimizer_config = optimizer_config
        self.registry = registry
        self.models_dir = models_dir
        self.score_loader = ScoreLoader(system_config.storage)
        self.return_builder = StrategyReturnBuilder()
        self.correlation = CorrelationAnalyzer()
        self.standalone = StandaloneMetrics()
        self.optimizer = MeanVarianceOptimizer(optimizer_config.mean_variance)
        self.config_writer = ModelConfigWriter()
        self.result_writer = ResultWriter(system_config.storage)

    @classmethod
    def from_config(cls, system_config_path: str = DEFAULT_SYSTEM_CONFIG, optimizer_config_path: str = DEFAULT_OPTIMIZER_CONFIG,
                    registry_path: str = DEFAULT_REGISTRY_CONFIG, models_dir: str = DEFAULT_MODELS_DIR) -> "OptimizerRunner":
        system_config = load_system_config(system_config_path)
        registry = AlphaRegistry.from_paths(registry_path, models_dir, optimizer_path=optimizer_config_path, system_config=system_config)
        return cls(system_config, load_optimizer_config(optimizer_config_path), registry, models_dir)

    def run(self, model_id: str | list[str], overwrite: bool = False) -> dict[str, pd.DataFrame]:
        model_ids = [model_id] if isinstance(model_id, str) else list(model_id)
        return {m: self.run_one(m, overwrite) for m in model_ids}

    def run_one(self, model_id: str, overwrite: bool = False) -> pd.DataFrame:
        """1. load config  2. per-alpha daily returns  3. correlation + standalone + Barra
        4. optimise  5. combined stats + Barra  6. write weights back  7. save audit table."""
        model = self.registry.require_model(model_id)
        market = str(model.market)
        dates = [to_datestr(d) for d in date_range(model.start_date, model.end_date)]
        log.info(f"optimizing {model_id}: {len(model.alpha_ids)} alpha(s) over {model.start_date}-{model.end_date}")

        daily_returns = self.return_builder.build_for_model(model.alpha_ids, market, model.rtn_alpha_id, dates, self.score_loader)
        if daily_returns.empty:
            raise ValueError(f"{model_id}: no overlapping dates across {model.alpha_ids} - cannot build a covariance matrix")

        corr_matrix = self.correlation.compute(daily_returns)
        avg_correlations = {a: self.correlation.average_correlation(corr_matrix, a) for a in daily_returns.columns}
        standalone_metrics = self.standalone.compute_all(daily_returns)

        decomposer = BarraDecomposer(PITDataManager.for_research(self.system_config, max(date_range(model.start_date, model.end_date))))
        date_objects = date_range(model.start_date, model.end_date)
        standalone_barra = {a: decomposer.compute_standalone(self.score_loader.load_alpha_scores(market, a, dates), market, date_objects)
                            for a in daily_returns.columns}

        weights = self.optimizer.optimize(daily_returns)
        combined_stats = self.optimizer.compute_portfolio_stats(daily_returns, weights)
        combined_barra = decomposer.compute_combined(standalone_barra, weights)

        config_path = self.config_writer.write_weights(model_id, self.models_dir, weights)
        log.info(f"wrote weights to {config_path}: {weights.round(4).to_dict()}")

        table = self.result_writer.build_table(weights, standalone_metrics, avg_correlations,
                                               standalone_barra, combined_stats, combined_barra)
        result_path = self.result_writer.save(table, market, model_id, overwrite)
        log.info(f"saved optimizer audit table to {result_path}")
        return table
