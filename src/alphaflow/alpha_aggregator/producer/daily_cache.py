import threading
from datetime import date

import pandas as pd

# Alpha types whose value is fixed for the trading day, so recomputing intraday is wasted work
DAILY_ALPHA_TYPES = ("low", "rtn")


class DailyAlphaCache:
    """Holds the current trading day's LowAlpha/RtnAlpha results in memory.

    Thread-safe: MarketProducerThread reads and writes it from its own loop, and
    force_refresh() can be called from outside.
    """

    def __init__(self) -> None:
        self._scores: dict[str, pd.Series] = {}
        self._cached_date: date | None = None
        self._lock = threading.Lock()

    def is_fresh(self, current_date: date) -> bool:
        with self._lock:
            return self._cached_date == current_date and bool(self._scores)

    def get(self, alpha_id: str) -> pd.Series | None:
        with self._lock:
            return self._scores.get(alpha_id)

    def set(self, alpha_id: str, scores: pd.Series, current_date: date) -> None:
        with self._lock:
            # A new trading day invalidates everything held for the old one
            if self._cached_date != current_date:
                self._scores = {}
                self._cached_date = current_date
            self._scores[alpha_id] = scores

    def all_scores(self) -> dict[str, pd.Series]:
        with self._lock:
            return dict(self._scores)

    def force_refresh(self) -> None:
        with self._lock:
            self._scores = {}
            self._cached_date = None
