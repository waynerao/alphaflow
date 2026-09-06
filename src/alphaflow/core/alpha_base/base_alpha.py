from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import pandas as pd

from alphaflow.core.config.alpha_config.base_config import BaseAlphaConfig
from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.paths import DEFAULT_UNIVERSE_DIR, resolve_config_path
from alphaflow.core.data_access.pit_manager import PITDataManager

RIC_COLUMN = "RIC"


class BaseAlpha(ABC):
    """Abstract base for every alpha. All alphas return a pd.Series indexed by RIC.

    compute() reads self._working_data, which SignalBuilderRunner injects after loading and
    RIC-joining the alpha's required_inputs. The alpha never fetches its own data - that is
    what keeps the research and production paths identical.
    """

    def __init__(self, config: BaseAlphaConfig, data: PITDataManager, universe_dir: str = DEFAULT_UNIVERSE_DIR) -> None:
        self.config = config
        self.data = data
        self.universe_dir = universe_dir
        self._working_data: pd.DataFrame | None = None
        self.validate_config()

    @property
    def alpha_id(self) -> str:
        return self.config.name

    @property
    def alpha_type(self) -> str:
        return self.config.alpha_type

    @property
    def market_scope(self) -> list[str]:
        return [str(m) for m in self.config.market_scope]

    @property
    def required_inputs(self) -> list[str]:
        return self.config.required_inputs

    @property
    def working_data(self) -> pd.DataFrame:
        if self._working_data is None:
            raise RuntimeError(f"{self.alpha_id}: _working_data not injected - compute() must be driven by SignalBuilderRunner")
        return self._working_data

    @abstractmethod
    def compute(self, market: str, as_of_date: date) -> pd.Series:
        """Return a pd.Series indexed by RIC. Stocks with missing data are excluded
        outright, never zero-filled - a zero is a real signal value."""

    def normalize(self, raw_signal: pd.Series) -> pd.Series:
        """Identity by default. Subclasses override with their own SHIFT/SCALE."""
        return raw_signal

    def run(self, market: str, as_of_date: date) -> pd.Series:
        self.check_market(market)
        raw = self.compute(market, as_of_date)
        if not isinstance(raw, pd.Series):
            raise TypeError(f"{self.alpha_id}.compute() must return a pd.Series, got {type(raw).__name__}")
        normalized = self.normalize(raw)
        normalized.index.name = RIC_COLUMN
        return normalized

    def get_universe(self, market: str) -> pd.Series:
        """RICs for the market, minus this alpha's custom_exclusions."""
        path = Path(resolve_config_path(self.universe_dir)) / f"universe_{market}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"universe file not found: {path}")
        universe = pd.read_csv(path)
        if RIC_COLUMN not in universe.columns:
            raise ConfigValidationError(f"{path} must have a single '{RIC_COLUMN}' column, got {list(universe.columns)}")
        rics = universe[RIC_COLUMN].dropna().astype(str).str.strip()
        excluded = set(self.config.custom_exclusions)
        return rics[~rics.isin(excluded)].drop_duplicates().reset_index(drop=True)

    def check_market(self, market: str) -> None:
        if market not in self.market_scope:
            raise ConfigValidationError(f"{self.alpha_id}: market {market!r} outside market_scope {self.market_scope}")

    def validate_config(self) -> None:
        """Called on construction. Subclasses extend, never replace, this."""
        if not self.config.name:
            raise ConfigValidationError("alpha config has an empty name")
        if not self.config.required_inputs:
            raise ConfigValidationError(f"{self.config.name}: required_inputs must not be empty")
        if self.config.alpha_type != self.EXPECTED_ALPHA_TYPE:
            raise ConfigValidationError(f"{self.config.name}: alpha_type {self.config.alpha_type!r} loaded as {self.EXPECTED_ALPHA_TYPE!r}")

    EXPECTED_ALPHA_TYPE: str = ""
