from pydantic import BaseModel, ConfigDict, Field, field_validator

from alphaflow.core.config.system_config import validate_hhmm


class ProductionModelConfig(BaseModel):
    active_model_ids: list[str] = Field(min_length=1)


class ProductionSchedulerConfig(BaseModel):
    daily_refresh_time: str
    intraday_update_interval_sec: int = Field(gt=0)

    _check_refresh_time = field_validator("daily_refresh_time")(staticmethod(validate_hhmm))


class ThreadingConfig(BaseModel):
    max_worker_threads: int = Field(default=4, gt=0)
    daily_thread_name: str = "daily_aggregator"
    intraday_thread_name: str = "intraday_aggregator"
    restart_on_failure: bool = True
    max_restart_attempts: int = Field(default=3, ge=1)


class ProductionOutputConfig(BaseModel):
    kdb_table_daily: str
    kdb_table_intraday: str
    parquet_partition_by: str = "date"


class ProductionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: ProductionModelConfig
    scheduler: ProductionSchedulerConfig
    threading: ThreadingConfig
    output: ProductionOutputConfig
