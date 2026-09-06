from datetime import date

import pandas as pd

from alphaflow.core.data_access.s3_client import S3Client
from alphaflow.core.utils.dates import to_datestr
from alphaflow.data_dumper.extractors.base_extractor import RIC_COLUMN, BaseExtractor

S3_DATA_TYPES = ["intropic", "southbound"]


class S3Extractor(BaseExtractor):
    """intropic (intrinsic) and southbound flow. Key templates live in [s3.keys]."""

    def __init__(self, s3_client: S3Client) -> None:
        self.client = s3_client

    @property
    def supported_data_types(self) -> list[str]:
        return list(S3_DATA_TYPES)

    def extract(self, data_type: str, market: str, rics: list[str], date: date) -> pd.DataFrame:
        if not self.supports(data_type):
            raise ValueError(f"S3Extractor does not handle {data_type!r}; supported: {self.supported_data_types}")
        key = self.client.config.key_for(data_type, market, to_datestr(date))
        df = self.client.read_parquet(key, date) if key.endswith(".parquet") else self.client.read_csv(key, date)
        # S3 files are market-wide; restrict to the requested universe
        return df[df[RIC_COLUMN].isin(rics)].reset_index(drop=True)
