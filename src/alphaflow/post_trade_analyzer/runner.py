import pandas as pd

from alphaflow.core.alpha_base.registry import AlphaRegistry
from alphaflow.core.config.loader import load_system_config
from alphaflow.core.config.paths import DEFAULT_MODELS_DIR, DEFAULT_REGISTRY_CONFIG, DEFAULT_SYSTEM_CONFIG
from alphaflow.core.config.system_config import SystemConfig
from alphaflow.core.data_access.pit_manager import PITDataManager
from alphaflow.core.integrations.logging_setup import setup_logger
from alphaflow.core.utils.dates import to_date_list
from alphaflow.post_trade_analyzer.attribution.factor_attribution import FactorAttribution
from alphaflow.post_trade_analyzer.loaders.alpha_score_loader import AlphaScoreLoader
from alphaflow.post_trade_analyzer.loaders.realized_return_loader import RealizedReturnLoader
from alphaflow.post_trade_analyzer.reporting.report_generator import ReportGenerator

log = setup_logger(__name__)


class PostTradeAnalyzerRunner:
    """Compares predicted alpha scores against realised returns and attributes the gap to
    Barra factors. All alphas in a model land in one report."""

    def __init__(self, system_config: SystemConfig, registry: AlphaRegistry) -> None:
        self.system_config = system_config
        self.registry = registry
        self.score_loader = AlphaScoreLoader(system_config.storage)
        self.return_loader = RealizedReturnLoader(system_config.storage)
        self.reporter = ReportGenerator(system_config.storage)

    @classmethod
    def from_config(cls, system_config_path: str = DEFAULT_SYSTEM_CONFIG, registry_path: str = DEFAULT_REGISTRY_CONFIG,
                    models_dir: str = DEFAULT_MODELS_DIR) -> "PostTradeAnalyzerRunner":
        system_config = load_system_config(system_config_path)
        registry = AlphaRegistry.from_paths(registry_path, models_dir, system_config=system_config)
        return cls(system_config, registry)

    def run(self, model_id: str | list[str], dates: list[str], overwrite: bool = False) -> dict[str, pd.DataFrame]:
        model_ids = [model_id] if isinstance(model_id, str) else list(model_id)
        if not dates:
            raise ValueError("dates must not be empty")
        return {m: self.run_one(m, dates, overwrite) for m in model_ids}

    def run_one(self, model_id: str, dates: list[str], overwrite: bool = False) -> pd.DataFrame:
        model = self.registry.require_model(model_id)
        market = str(model.market)
        date_objects = to_date_list(dates)

        realized = self.return_loader.load(market, model.rtn_alpha_id, dates)
        if realized.empty:
            raise FileNotFoundError(f"no stored returns for {model.rtn_alpha_id} in {market} over the requested dates")

        attribution = FactorAttribution(PITDataManager.for_research(self.system_config, max(date_objects)))
        results, summaries = {}, {}
        for alpha_id in model.alpha_ids:
            predicted = self.score_loader.load(market, alpha_id, dates)
            if predicted.empty:
                log.warning(f"{model_id}: no stored scores for {alpha_id} - excluded from the report")
                continue
            difference = attribution.compute_difference(predicted, realized)
            summaries[alpha_id] = attribution.summarize_difference(difference)
            results[alpha_id] = attribution.run_regression(difference, market, date_objects)
            log.info(f"{model_id}/{alpha_id}: attributed {len(difference)} observation(s)")

        if not summaries:
            raise FileNotFoundError(f"{model_id}: no alpha had stored scores over the requested dates")
        report = self.reporter.assemble(results, summaries)
        path = self.reporter.save(report, model_id, max(date_objects), overwrite)
        log.info(f"saved post-trade report to {path}")
        return report
