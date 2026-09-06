from datetime import date

import pandas as pd

from alphaflow.data_dumper.extractors.base_extractor import BaseExtractor

BBG_DATA_TYPES = ["bbg"]


class BBGExtractor(BaseExtractor):
    """Placeholder. The Bloomberg field list is still to be decided and the pull is to be
    implemented via desktool - see the spec's appendix of open items."""

    def __init__(self, bbg_client=None) -> None:
        self.client = bbg_client

    @property
    def supported_data_types(self) -> list[str]:
        return list(BBG_DATA_TYPES)

    def extract(self, data_type: str, market: str, rics: list[str], date: date) -> pd.DataFrame:
        raise NotImplementedError("BBGExtractor is not implemented yet - the bbg field list is still open (spec appendix)")
