from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

RIC_COLUMN = "RIC"


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, data_type: str, market: str, rics: list[str], date: date) -> pd.DataFrame:
        """Return a RIC-keyed frame for one data_type on one date."""

    @property
    @abstractmethod
    def supported_data_types(self) -> list[str]: ...

    def supports(self, data_type: str) -> bool:
        return data_type in self.supported_data_types
