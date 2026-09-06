from datetime import date
from io import BytesIO

import pandas as pd

from alphaflow.core.config.system_config import S3Config
from alphaflow.core.data_access.exceptions import DataNotFoundError, DataSourceError

RIC_COLUMN = "RIC"


class S3Client:
    """Reads intrinsic / southbound files from S3. boto3 is imported lazily so that
    constructing the client costs nothing until a read actually happens."""

    def __init__(self, config: S3Config, client=None) -> None:
        self.config = config
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                session = boto3.Session(profile_name=self.config.profile_name) if self.config.profile_name else boto3.Session()
                self._client = session.client("s3", region_name=self.config.region)
            except Exception as exc:
                raise DataSourceError(f"could not create S3 client: {exc}") from exc
        return self._client

    def read_csv(self, key: str, as_of_date: date | None = None) -> pd.DataFrame:
        return self._read(key, as_of_date, pd.read_csv)

    def read_parquet(self, key: str, as_of_date: date | None = None) -> pd.DataFrame:
        return self._read(key, as_of_date, pd.read_parquet)

    def _read(self, key: str, as_of_date: date | None, reader) -> pd.DataFrame:
        try:
            response = self.client.get_object(Bucket=self.config.bucket_name, Key=key)
            body = response["Body"].read()
        except Exception as exc:
            raise DataSourceError(f"S3 read failed for s3://{self.config.bucket_name}/{key}: {exc}") from exc
        df = reader(BytesIO(body))
        if df is None or df.empty:
            raise DataNotFoundError(f"s3://{self.config.bucket_name}/{key} is empty")
        if RIC_COLUMN not in df.columns:
            source = next((c for c in df.columns if c.lower() == "ric"), None)
            if source is None:
                raise DataSourceError(f"s3://{self.config.bucket_name}/{key} has no RIC column; got {list(df.columns)}")
            df = df.rename(columns={source: RIC_COLUMN})
        return df.reset_index(drop=True)
