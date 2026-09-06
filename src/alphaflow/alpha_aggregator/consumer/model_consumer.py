from datetime import date

import pandas as pd

from alphaflow.alpha_aggregator.storage.kdb_writer import KDBWriter
from alphaflow.alpha_aggregator.storage.parquet_writer import ParquetWriter
from alphaflow.core.alpha_base.registry import AlphaRegistry
from alphaflow.core.config.system_config import SystemConfig
from alphaflow.core.integrations.logging_setup import setup_logger

log = setup_logger(__name__)


class ModelConsumer:
    """Combines individual alpha scores into a model composite, on demand.

    Not a background thread - instantiated per call. Composites are never written to kdb+;
    the Producer's individual scores are the single source of truth, and any number of models
    can be combined from them without recomputation.
    """

    def __init__(self, system_config: SystemConfig, registry: AlphaRegistry, kdb_writer: KDBWriter) -> None:
        self.system_config = system_config
        self.registry = registry
        self.kdb_writer = kdb_writer
        self.parquet_writer = ParquetWriter(system_config.storage)

    def get_composite_score(self, model_id: str, as_of_date: date | None = None) -> pd.Series:
        """composite(RIC) = sum_i( weight_i * score_i(RIC) ) over the union of RICs.

        A RIC missing from one alpha contributes 0 for that alpha; the sum is NOT
        renormalised, so a name covered by fewer alphas legitimately scores smaller.
        """
        model = self.registry.require_model(model_id)
        if not model.is_optimized:
            raise ValueError(f"model {model_id!r} has no alpha_weights - run OptimizerRunner first")

        individual = self.kdb_writer.query_latest_scores(str(model.market), model.alpha_ids)
        missing = [a for a in model.alpha_ids if a not in individual or individual[a].empty]
        if missing:
            log.warning(f"{model_id}: no live scores for {missing} - they contribute 0 to the composite")
        available = {a: s for a, s in individual.items() if a in model.alpha_ids and s is not None and not s.empty}
        if not available:
            raise ValueError(f"{model_id}: no live scores available for any of {model.alpha_ids}")

        rics = sorted(set().union(*(set(s.index) for s in available.values())))
        composite = pd.Series(0.0, index=pd.Index(rics, name="RIC"), name=model_id)
        for alpha_id, series in available.items():
            composite += series.reindex(rics).fillna(0.0) * model.weight(alpha_id)

        if model.save_to_parquet:
            path = self.parquet_writer.write(model_id, available, composite, as_of_date or date.today())
            log.info(f"{model_id}: archived composite to {path}")
        return composite
