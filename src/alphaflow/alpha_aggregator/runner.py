from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from alphaflow.alpha_aggregator.consumer.model_consumer import ModelConsumer
from alphaflow.alpha_aggregator.producer.market_producer import MarketProducerThread
from alphaflow.alpha_aggregator.storage.kdb_writer import KDBWriter
from alphaflow.core.alpha_base.registry import AlphaRegistry
from alphaflow.core.config.loader import load_production_config, load_system_config
from alphaflow.core.config.paths import (
    DEFAULT_MODELS_DIR,
    DEFAULT_PRODUCTION_CONFIG,
    DEFAULT_REGISTRY_CONFIG,
    DEFAULT_SYSTEM_CONFIG,
)
from alphaflow.core.config.production_config import ProductionConfig
from alphaflow.core.config.system_config import SystemConfig
from alphaflow.core.data_access.kdb_client import KDBClient
from alphaflow.core.integrations.logging_setup import setup_logger
from alphaflow.signal_builder.runner import SignalBuilderRunner

log = setup_logger(__name__)


class AlphaAggregatorRunner:
    """Production runtime: one producer thread per market with active models, plus an
    on-demand consumer.

    A scheduler fires force_refresh_daily() at the configured wall-clock time; the intraday
    cadence is the producer's own loop interval.
    """

    def __init__(self, system_config: SystemConfig, production_config: ProductionConfig, registry: AlphaRegistry) -> None:
        self.system_config = system_config
        self.production_config = production_config
        self.registry = registry
        self.signal_builder = SignalBuilderRunner(system_config, registry)
        self.kdb_writer = KDBWriter(production_config.output, KDBClient(system_config.kdb))
        self.producers: dict[str, MarketProducerThread] = {}
        self._scheduler: BackgroundScheduler | None = None

    @classmethod
    def from_config(cls, system_config_path: str = DEFAULT_SYSTEM_CONFIG, production_config_path: str = DEFAULT_PRODUCTION_CONFIG,
                    registry_path: str = DEFAULT_REGISTRY_CONFIG, models_dir: str = DEFAULT_MODELS_DIR) -> "AlphaAggregatorRunner":
        system_config = load_system_config(system_config_path)
        registry = AlphaRegistry.from_paths(registry_path, models_dir, production_path=production_config_path, system_config=system_config)
        return cls(system_config, load_production_config(production_config_path), registry)

    def active_markets(self) -> list[str]:
        markets = []
        for model_id in self.production_config.models.active_model_ids:
            model = self.registry.get_model(model_id)
            if model is not None and str(model.market) not in markets:
                markets.append(str(model.market))
        return markets

    def start_producers(self) -> dict[str, MarketProducerThread]:
        """One thread per market. A market that fails to start is logged and skipped, so a
        single bad configuration cannot take the whole runtime down with it."""
        for market in self.active_markets():
            if market in self.producers and self.producers[market].is_alive():
                continue
            try:
                producer = MarketProducerThread(market, self.signal_builder, self.registry, self.production_config, self.kdb_writer)
                producer.start()
                self.producers[market] = producer
            except Exception as exc:
                log.error(f"could not start producer for {market}: {exc}")
        self._start_daily_scheduler()
        return self.producers

    def stop_producers(self) -> None:
        for market, producer in self.producers.items():
            producer.stop()
            log.info(f"[{market}] producer stop requested")
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

    def get_consumer(self) -> ModelConsumer:
        return ModelConsumer(self.system_config, self.registry, self.kdb_writer)

    def _start_daily_scheduler(self) -> None:
        """Fires the daily-alpha cache refresh at production.scheduler.daily_refresh_time."""
        if self._scheduler is not None:
            return
        hour, minute = (int(p) for p in self.production_config.scheduler.daily_refresh_time.split(":"))
        self._scheduler = BackgroundScheduler(timezone=self.system_config.scheduler.timezone)
        self._scheduler.add_job(self.refresh_all_daily, CronTrigger(hour=hour, minute=minute), id="daily_refresh")
        self._scheduler.start()
        log.info(f"daily refresh scheduled at {self.production_config.scheduler.daily_refresh_time} {self.system_config.scheduler.timezone}")

    def refresh_all_daily(self) -> None:
        for producer in self.producers.values():
            producer.force_refresh_daily()
