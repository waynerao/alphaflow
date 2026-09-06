import threading
from datetime import date

import pandas as pd

from alphaflow.alpha_aggregator.alerting.alert import send_critical_alert
from alphaflow.alpha_aggregator.producer.daily_cache import DAILY_ALPHA_TYPES, DailyAlphaCache
from alphaflow.alpha_aggregator.storage.kdb_writer import KDBWriter
from alphaflow.core.alpha_base.registry import AlphaRegistry
from alphaflow.core.config.production_config import ProductionConfig
from alphaflow.core.integrations.logging_setup import setup_logger
from alphaflow.core.utils.dates import to_datestr
from alphaflow.signal_builder.runner import SOURCE_LIVE, SignalBuilderRunner

RIC_COLUMN = "RIC"
log = setup_logger(__name__)


def resolve_market_alpha_ids(market: str, production_config: ProductionConfig, registry: AlphaRegistry) -> list[str]:
    """Union of alpha_ids and rtn_alpha_ids across every active model in this market.

    Computed once at thread startup and never recalculated: the whole point of the
    Producer/Consumer split is that an alpha shared by five models is computed once.
    """
    alpha_ids: list[str] = []
    for model_id in production_config.models.active_model_ids:
        model = registry.get_model(model_id)
        if model is None or str(model.market) != market:
            continue
        for alpha_id in [*model.alpha_ids, model.rtn_alpha_id]:
            if alpha_id not in alpha_ids:
                alpha_ids.append(alpha_id)
    return alpha_ids


class MarketProducerThread(threading.Thread):
    """One per market. Computes individual alpha scores in a loop and writes them to kdb+.

    Daily alphas (low/rtn) are computed once per trading day and cached; intraday alphas
    (high/mid) are recomputed every iteration from live data. A market that fails past its
    retry budget stops itself and alerts - the other markets' threads are unaffected.
    """

    def __init__(self, market: str, signal_builder: SignalBuilderRunner, registry: AlphaRegistry,
                 production_config: ProductionConfig, kdb_writer: KDBWriter | None = None,
                 cache: DailyAlphaCache | None = None) -> None:
        super().__init__(name=f"producer_{market}", daemon=True)
        self.market = market
        self.signal_builder = signal_builder
        self.registry = registry
        self.production_config = production_config
        self.kdb_writer = kdb_writer
        self.cache = cache if cache is not None else DailyAlphaCache()
        self.alpha_ids = resolve_market_alpha_ids(market, production_config, registry)
        self.daily_alpha_ids = [a for a in self.alpha_ids if self._alpha_type(a) in DAILY_ALPHA_TYPES]
        self.intraday_alpha_ids = [a for a in self.alpha_ids if self._alpha_type(a) not in DAILY_ALPHA_TYPES]
        self._stop_event = threading.Event()
        self.failure_count = 0
        self.last_error: Exception | None = None

    def _alpha_type(self, alpha_id: str) -> str:
        entry = self.registry.get_by_id(alpha_id)
        return entry.alpha_type if entry is not None else ""

    def start(self) -> None:
        log.info(f"[{self.market}] starting producer: {len(self.daily_alpha_ids)} daily, {len(self.intraday_alpha_ids)} intraday")
        super().start()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        interval = self.production_config.scheduler.intraday_update_interval_sec
        max_attempts = self.production_config.threading.max_restart_attempts
        while not self._stop_event.is_set():
            try:
                self.run_once()
                self.failure_count = 0
            except Exception as exc:
                self.failure_count += 1
                self.last_error = exc
                log.error(f"[{self.market}] iteration failed ({self.failure_count}/{max_attempts}): {exc}")
                if not self.production_config.threading.restart_on_failure or self.failure_count >= max_attempts:
                    send_critical_alert(f"AlphaFlow producer stopped: {self.market}",
                                        f"{self.failure_count} consecutive failures, last error: {exc}")
                    self._stop_event.set()
                    return
            self._stop_event.wait(interval)

    def run_once(self, current_date: date | None = None) -> dict[str, pd.Series]:
        """One production cycle. Returns everything computed, keyed by alpha_id."""
        today = current_date or date.today()
        daily = self._daily_scores(today)
        intraday = self._compute(self.intraday_alpha_ids, today)
        if self.kdb_writer is not None:
            self.kdb_writer.write_individual_scores(self.market, daily, intraday)
        return {**daily, **intraday}

    def _daily_scores(self, today: date) -> dict[str, pd.Series]:
        if self.cache.is_fresh(today):
            return self.cache.all_scores()
        computed = self._compute(self.daily_alpha_ids, today)
        for alpha_id, series in computed.items():
            self.cache.set(alpha_id, series, today)
        log.info(f"[{self.market}] refreshed {len(computed)} daily alpha(s) for {to_datestr(today)}")
        return computed

    def _compute(self, alpha_ids: list[str], today: date) -> dict[str, pd.Series]:
        """save=False - production never persists individual scores to the shared drive;
        kdb+ is the production sink."""
        if not alpha_ids:
            return {}
        frames = self.signal_builder.build_one_day_alpha(alpha_ids, self.market, to_datestr(today),
                                                         save=False, source=SOURCE_LIVE)
        return {alpha_id: frame.set_index(RIC_COLUMN)["score"] for alpha_id, frame in frames.items()}

    def force_refresh_daily(self) -> None:
        """Called by the daily scheduler trigger, and available manually."""
        log.info(f"[{self.market}] forcing daily alpha refresh")
        self.cache.force_refresh()
