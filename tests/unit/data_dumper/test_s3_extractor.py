from datetime import date

import pandas as pd
import pytest

from alphaflow.core.data_access.exceptions import DataNotFoundError, DataSourceError
from alphaflow.core.data_access.s3_client import S3Client
from alphaflow.data_dumper.extractors.s3_extractor import S3Extractor

DAY = date(2024, 1, 15)
RICS = ["0700.HK", "0941.HK"]


class FakeS3Body:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload


class FakeBoto:
    def __init__(self, payload: bytes | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls = []

    def get_object(self, Bucket, Key):
        self.calls.append((Bucket, Key))
        if self.error is not None:
            raise self.error
        return {"Body": FakeS3Body(self.payload)}


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


class TestS3Extractor:
    def test_supports_only_the_two_s3_data_types(self, system_config):
        extractor = S3Extractor(S3Client(system_config.s3, client=FakeBoto()))
        assert set(extractor.supported_data_types) == {"intropic", "southbound"}

    def test_an_unsupported_data_type_is_rejected(self, system_config):
        extractor = S3Extractor(S3Client(system_config.s3, client=FakeBoto()))
        with pytest.raises(ValueError, match="does not handle 'minute_bar'"):
            extractor.extract("minute_bar", "HK", RICS, DAY)

    def test_the_configured_key_template_is_used(self, system_config):
        boto = FakeBoto(csv_bytes(pd.DataFrame({"RIC": RICS, "flow": [1, 2]})))
        S3Extractor(S3Client(system_config.s3, client=boto)).extract("southbound", "HK", RICS, DAY)
        assert boto.calls[0][1] == "southbound/HK/20240115.csv"

    def test_rows_outside_the_requested_universe_are_dropped(self, system_config):
        payload = csv_bytes(pd.DataFrame({"RIC": RICS + ["9999.HK"], "flow": [1, 2, 3]}))
        extractor = S3Extractor(S3Client(system_config.s3, client=FakeBoto(payload)))
        assert set(extractor.extract("southbound", "HK", RICS, DAY)["RIC"]) == set(RICS)

    def test_a_lowercase_ric_column_is_normalized(self, system_config):
        payload = csv_bytes(pd.DataFrame({"ric": RICS, "flow": [1, 2]}))
        client = S3Client(system_config.s3, client=FakeBoto(payload))
        assert "RIC" in client.read_csv("k").columns

    def test_an_empty_object_raises_not_found(self, system_config):
        payload = csv_bytes(pd.DataFrame({"RIC": [], "flow": []}))
        client = S3Client(system_config.s3, client=FakeBoto(payload))
        with pytest.raises(DataNotFoundError, match="is empty"):
            client.read_csv("k")

    def test_a_missing_ric_column_is_an_error(self, system_config):
        client = S3Client(system_config.s3, client=FakeBoto(csv_bytes(pd.DataFrame({"flow": [1]}))))
        with pytest.raises(DataSourceError, match="has no RIC column"):
            client.read_csv("k")

    def test_a_transport_failure_is_wrapped(self, system_config):
        client = S3Client(system_config.s3, client=FakeBoto(error=RuntimeError("no such key")))
        with pytest.raises(DataSourceError, match="S3 read failed"):
            client.read_csv("k")

    def test_an_unmapped_data_type_is_reported(self, system_config):
        with pytest.raises(KeyError, match="no S3 key template"):
            system_config.s3.key_for("nope", "HK", "20240115")
