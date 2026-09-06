# AlphaFlow — Master Technical Specification

**Version:** 1.0
**Purpose:** Alpha research, calculation, and post-trade reconciliation platform for APAC equities (HK, CN, SG, JP, KR, AU, TW)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repo & Project Structure](#2-repo--project-structure)
3. [Config File Hierarchy](#3-config-file-hierarchy)
4. [Core Layer](#4-core-layer)
5. [data_dumper](#5-data_dumper)
6. [signal_builder](#6-signal_builder)
7. [strategy_backtester](#7-strategy_backtester)
8. [optimizer](#8-optimizer)
9. [alpha_aggregator](#9-alpha_aggregator)
10. [post_trade_analyzer](#10-post_trade_analyzer)
11. [Global Design Principles](#11-global-design-principles)

---

## 1. Project Overview

| Attribute | Decision |
|---|---|
| **Project type** | Internal tool for a hedge fund / asset manager |
| **Package name** | `alphaflow` |
| **Language** | Python 3.11+ |
| **Package manager** | `uv` |
| **Structure** | Monorepo, multiple sub-packages under one `src/alphaflow` namespace |
| **Interface** | No CLI — Python functions called directly, notebooks for research |
| **Deployment** | On-premise |
| **Asset class** | Equities only |
| **Markets** | HK (HKEX), CN (SSE/SZSE), SG, JP, KR, AU, TW |
| **Primary identifier** | `RIC` (Reuters Instrument Code) — used everywhere except internal Bloomberg calls |
| **Team** | Small team, all hats |
| **Data sources** | kdb+ (tick, OHLCV, Barra risk factors), Bloomberg (via xbbg/desktool), S3 (intrinsic, southbound) |
| **Internal dependency** | `apcr_desktool` — pins Python/numpy/xbbg versions, provides BBL↔RIC mapping and auth |

### High-Level Pipeline

```
data_dumper → signal_builder → strategy_backtester → optimizer
                     ↓                                    ↓
              alpha_aggregator (production) ←—————— model weights
                     ↓
           post_trade_analyzer
```

---

## 2. Repo & Project Structure

### 2.1 Full Directory Tree

```
alphaflow/
│
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── README.md
│
├── configs/
│   ├── system.toml
│   ├── production.toml
│   ├── optimizer.toml
│   ├── registry.toml
│   ├── alphas/
│   │   ├── high/
│   │   │   ├── _template.toml
│   │   │   └── <alpha_id>.toml
│   │   ├── mid/
│   │   │   ├── _template.toml
│   │   │   └── <alpha_id>.toml
│   │   ├── low/
│   │   │   ├── _template.toml
│   │   │   └── <alpha_id>.toml
│   │   └── rtn/
│   │       ├── _template.toml
│   │       └── <alpha_id>.toml
│   ├── models/
│   │   └── <model_id>.toml
│   └── universe/
│       ├── universe_HK.csv
│       ├── universe_CN.csv
│       ├── universe_SG.csv
│       ├── universe_JP.csv
│       ├── universe_KR.csv
│       ├── universe_AU.csv
│       └── universe_TW.csv
│
├── notebooks/
│   ├── README.md
│   ├── alpha_research/
│   ├── portfolio_optimization/
│   └── post_trade/
│
├── src/
│   └── alphaflow/
│       ├── __init__.py
│       │
│       ├── core/                          # Shared Python CODE
│       │   ├── __init__.py
│       │   ├── data_access/
│       │   │   ├── __init__.py
│       │   │   ├── pit_manager.py         # PITDataManager
│       │   │   ├── kdb_client.py          # KDBClient
│       │   │   ├── bloomberg_client.py    # BBGClient
│       │   │   ├── s3_client.py           # S3Client
│       │   │   └── exceptions.py
│       │   ├── alpha_base/
│       │   │   ├── __init__.py
│       │   │   ├── base_alpha.py          # BaseAlpha (ABC)
│       │   │   ├── high_alpha.py          # HighAlpha
│       │   │   ├── mid_alpha.py           # MidAlpha
│       │   │   ├── low_alpha.py           # LowAlpha
│       │   │   ├── rtn_alpha.py           # RtnAlpha
│       │   │   ├── market.py              # Market enum
│       │   │   ├── registry.py            # AlphaRegistry
│       │   │   └── loader.py              # AlphaLoader
│       │   └── config/
│       │       ├── __init__.py
│       │       ├── system_config.py
│       │       ├── production_config.py
│       │       ├── optimizer_config.py
│       │       ├── registry_config.py
│       │       ├── model_config.py
│       │       ├── loader.py
│       │       └── alpha_config/
│       │           ├── __init__.py
│       │           ├── base_config.py
│       │           ├── high_config.py
│       │           ├── mid_config.py
│       │           ├── low_config.py
│       │           └── rtn_config.py
│       │
│       ├── data_dumper/
│       │   ├── __init__.py
│       │   ├── extractors/
│       │   │   ├── __init__.py
│       │   │   ├── base_extractor.py
│       │   │   ├── kdb_extractor.py
│       │   │   ├── s3_extractor.py
│       │   │   └── bbg_extractor.py       # placeholder — implemented via desktool later
│       │   ├── writers/
│       │   │   ├── __init__.py
│       │   │   └── csv_writer.py
│       │   └── runner.py                  # DataDumperRunner
│       │
│       ├── signal_builder/
│       │   ├── __init__.py
│       │   ├── alphas/
│       │   │   ├── __init__.py
│       │   │   ├── high/
│       │   │   │   ├── __init__.py
│       │   │   │   └── <alpha_id>.py
│       │   │   ├── mid/
│       │   │   │   ├── __init__.py
│       │   │   │   └── <alpha_id>.py
│       │   │   ├── low/
│       │   │   │   ├── __init__.py
│       │   │   │   └── momentum_price.py
│       │   │   └── rtn/
│       │   │       ├── __init__.py
│       │   │       └── <alpha_id>.py
│       │   ├── storage/
│       │   │   ├── __init__.py
│       │   │   └── score_writer.py        # ScoreWriter
│       │   └── runner.py                  # SignalBuilderRunner
│       │
│       ├── strategy_backtester/
│       │   ├── __init__.py
│       │   ├── engine/
│       │   │   ├── __init__.py
│       │   │   └── backtest_engine.py     # BacktestEngine
│       │   ├── regression/
│       │   │   ├── __init__.py
│       │   │   ├── cross_sectional_ols.py # CrossSectionalOLS
│       │   │   └── regression_report.py
│       │   ├── metrics/
│       │   │   ├── __init__.py
│       │   │   ├── ic_calculator.py       # ICCalculator
│       │   │   ├── performance_metrics.py # PerformanceMetrics
│       │   │   └── decile_analyzer.py     # DecileAnalyzer
│       │   ├── risk/
│       │   │   ├── __init__.py
│       │   │   └── barra_exposure.py      # BarraExposure
│       │   ├── reporting/
│       │   │   ├── __init__.py
│       │   │   └── report_generator.py    # ReportGenerator
│       │   ├── score_loader.py            # ScoreLoader (shared with optimizer)
│       │   └── runner.py                  # StrategyBacktesterRunner
│       │
│       ├── optimizer/
│       │   ├── __init__.py
│       │   ├── returns/
│       │   │   ├── __init__.py
│       │   │   └── strategy_return_builder.py # StrategyReturnBuilder
│       │   ├── analysis/
│       │   │   ├── __init__.py
│       │   │   ├── correlation.py         # CorrelationAnalyzer
│       │   │   └── standalone_metrics.py  # StandaloneMetrics
│       │   ├── optimizers/
│       │   │   ├── __init__.py
│       │   │   └── mean_variance_optimizer.py # MeanVarianceOptimizer
│       │   ├── risk/
│       │   │   ├── __init__.py
│       │   │   └── barra_decomposer.py    # BarraDecomposer
│       │   ├── storage/
│       │   │   ├── __init__.py
│       │   │   ├── model_config_writer.py # ModelConfigWriter
│       │   │   └── result_writer.py       # ResultWriter
│       │   └── runner.py                  # OptimizerRunner
│       │
│       ├── alpha_aggregator/
│       │   ├── __init__.py
│       │   ├── producer/
│       │   │   ├── __init__.py
│       │   │   ├── market_producer.py     # MarketProducerThread
│       │   │   └── daily_cache.py         # DailyAlphaCache
│       │   ├── consumer/
│       │   │   ├── __init__.py
│       │   │   └── model_consumer.py      # ModelConsumer
│       │   ├── storage/
│       │   │   ├── __init__.py
│       │   │   ├── kdb_writer.py          # KDBWriter
│       │   │   └── parquet_writer.py      # ParquetWriter
│       │   ├── alerting/
│       │   │   ├── __init__.py
│       │   │   └── alert.py               # send_critical_alert()
│       │   └── runner.py                  # AlphaAggregatorRunner
│       │
│       └── post_trade_analyzer/
│           ├── __init__.py
│           ├── loaders/
│           │   ├── __init__.py
│           │   ├── alpha_score_loader.py  # AlphaScoreLoader
│           │   └── realized_return_loader.py # RealizedReturnLoader
│           ├── attribution/
│           │   ├── __init__.py
│           │   └── factor_attribution.py  # FactorAttribution
│           ├── reporting/
│           │   ├── __init__.py
│           │   └── report_generator.py    # ReportGenerator
│           └── runner.py                  # PostTradeAnalyzerRunner
│
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── core/
    │   │   ├── test_pit_manager.py
    │   │   ├── test_base_alpha.py
    │   │   ├── test_high_alpha.py
    │   │   ├── test_mid_alpha.py
    │   │   ├── test_low_alpha.py
    │   │   └── test_rtn_alpha.py
    │   ├── data_dumper/
    │   │   ├── test_kdb_extractor.py
    │   │   └── test_s3_extractor.py
    │   ├── signal_builder/
    │   │   ├── test_score_writer.py
    │   │   └── test_signal_builder_runner.py
    │   ├── strategy_backtester/
    │   │   ├── test_cross_sectional_ols.py
    │   │   ├── test_ic_calculator.py
    │   │   └── test_performance_metrics.py
    │   ├── optimizer/
    │   │   ├── test_strategy_return_builder.py
    │   │   └── test_mean_variance_optimizer.py
    │   ├── alpha_aggregator/
    │   │   ├── test_market_producer.py
    │   │   └── test_model_consumer.py
    │   └── post_trade_analyzer/
    │       └── test_factor_attribution.py
    └── integration/
        ├── test_data_dumper_pipeline.py
        ├── test_signal_builder_pipeline.py
        ├── test_backtester_pipeline.py
        ├── test_optimizer_pipeline.py
        ├── test_aggregator_pipeline.py
        └── test_post_trade_pipeline.py
```

### 2.2 `pyproject.toml`

```toml
[project]
name            = "alphaflow"
version         = "0.1.0"
description     = "Alpha research, calculation and post-trade reconciliation platform"
requires-python = ">=3.11"

dependencies = [
    "apcr-desktool",
    "pandas",
    "numpy",
    "pyarrow",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "xbbg",
    "boto3",
    "pykx",
    "pydantic>=2.7",
    "apscheduler>=3.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "pytest-mock>=3.14",
    "ipykernel>=6.0",
    "jupyterlab>=4.0",
]

[build-system]
requires      = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/alphaflow"]

[tool.uv]
dev-dependencies = [
    "pytest>=8.2", "pytest-cov>=5.0", "pytest-mock>=3.14",
    "ipykernel>=6.0", "jupyterlab>=4.0",
]

# Internal dependency — pins Python/numpy/xbbg versions, editable local path
[tool.uv.sources]
apcr-desktool = { path = "../apcr_desktool", editable = true }

[tool.pytest.ini_options]
testpaths  = ["tests"]
pythonpath = ["src"]

[tool.ruff]
src         = ["src"]
line-length = 100
```

### 2.3 Shared Drive Storage Layout (Complete)

```
{shared_drive_path}/
│
├── raw_data/
│   └── <MARKET>/
│       └── <data_type>/
│           └── YYYYMMDD.csv                      # RIC primary key
│                                                   # depth_intraday: single file, 'timestamp' col
│
├── alpha_scores/
│   ├── <MARKET>/
│   │   └── <alpha_id>/
│   │       └── YYYYMMDD.parquet                 # Columns: RIC, raw_signal, score
│   └── <model_id>/
│       └── YYYYMMDD.parquet                     # Consumer composite archive
│                                                   # (individual scores + composite)
│
├── backtest_results/
│   └── <MARKET>/
│       └── <alpha_id>_YYYYMMDD-YYYYMMDD.parquet # measure x rtn_alpha_id table
│
├── optimizer_results/
│   └── <MARKET>/
│       └── <model_id>.parquet                   # weights + risk audit table
│
└── post_trade_results/
    └── <model_id>/
        └── YYYYMMDD.parquet                     # PAA report, combined across alphas
```

### 2.4 Namespace Import Convention

```python
# ☑ Correct
from alphaflow.core.data_access.pit_manager import PITDataManager
from alphaflow.core.alpha_base.base_alpha import BaseAlpha
from alphaflow.core.alpha_base.market import Market
from alphaflow.signal_builder.runner import SignalBuilderRunner
from alphaflow.optimizer.runner import OptimizerRunner

# ✗ Wrong — no relative cross-package imports
from ..core.data_access import PITDataManager
```

Each package `__init__.py` re-exports its primary public interface (e.g. the Runner class).

### 2.5 Runner Pattern (No CLI)

Every package exposes exactly one public `{Package}Runner` class with a `from_config()` classmethod and one or more callable methods. No CLI, no click — called directly from notebooks or from other packages (e.g. `alpha_aggregator` calls `SignalBuilderRunner`).

### 2.6 Alpha Type Hierarchy

```
BaseAlpha                   core/alpha_base/base_alpha.py
   ├── HighAlpha            — tick-level intraday (data updates on order book change)
   ├── MidAlpha             — interval-based intraday (fixed time interval)
   ├── LowAlpha             — daily / multi-day
   └── RtnAlpha             — return signals; ONLY type with entry/exit timing fields
```

### 2.7 Alpha & Model Control — Layered Design

```
registry.toml     → WHAT EXISTS      : master list of all configured alphas (with alpha_type)
configs/models/   → WHAT COMBINES    : market + alpha_ids + rtn_alpha_id + test period + weights
production.toml   → WHAT RUNS LIVE   : which model_ids the aggregator computes
optimizer.toml    → HOW TO OPTIMIZE  : mean-variance parameters + risk model (applies to all models)
```

### 2.8 Key Interfaces Summary (Complete)

| File | Primary Export | Role |
|---|---|---|
| `core/data_access/pit_manager.py` | `PITDataManager` | Single gateway for all data reads, RIC-keyed |
| `core/data_access/kdb_client.py` | `KDBClient` | kdb+ wrapper |
| `core/data_access/bloomberg_client.py` | `BBGClient` | xbbg wrapper; BBL↔RIC mapping internal via desktool |
| `core/data_access/s3_client.py` | `S3Client` | S3 reader |
| `core/alpha_base/base_alpha.py` | `BaseAlpha` | Abstract base — `compute()`, `normalize()`, `run()`, `get_universe()`, `validate_config()` |
| `core/alpha_base/high_alpha.py` / `mid_alpha.py` / `low_alpha.py` / `rtn_alpha.py` | `HighAlpha` / `MidAlpha` / `LowAlpha` / `RtnAlpha` | Type-specific bases |
| `core/alpha_base/registry.py` | `AlphaRegistry` | Registry lookup + cross-config validation |
| `core/alpha_base/loader.py` | `AlphaLoader` | Dispatch to typed loader by `alpha_type` |
| `core/alpha_base/market.py` | `Market` | `StrEnum`: HK, CN, SG, JP, KR, AU, TW |
| `core/config/model_config.py` | `ModelConfig` | Market + alpha_ids + rtn_alpha_id + dates + weights + save_to_parquet |
| `data_dumper/runner.py` | `DataDumperRunner` | `run()` — manual backfill |
| `signal_builder/runner.py` | `SignalBuilderRunner` | `build_one_day_alpha()`, `build_multi_day_alpha()` |
| `strategy_backtester/runner.py` | `StrategyBacktesterRunner` | `run()` |
| `optimizer/runner.py` | `OptimizerRunner` | `run()` |
| `alpha_aggregator/runner.py` | `AlphaAggregatorRunner` | `start_producers()`, `get_consumer()` |
| `post_trade_analyzer/runner.py` | `PostTradeAnalyzerRunner` | `run()` |

---

## 3. Config File Hierarchy

### 3.1 Overview

| File | Scope | Who reads it |
|---|---|---|
| `configs/system.toml` | Storage, connections, per-market config | All packages |
| `configs/production.toml` | Runtime, threading, active model_ids | `alpha_aggregator` |
| `configs/optimizer.toml` | Mean-variance params, risk model | `optimizer` |
| `configs/registry.toml` | Master alpha list with `alpha_type` | `AlphaRegistry` |
| `configs/models/<model_id>.toml` | Market + alphas + rtn_alpha_id + dates + weights | `optimizer`, `alpha_aggregator`, `post_trade_analyzer` |
| `configs/alphas/<type>/*.toml` | Per-alpha definitions | Typed loaders |
| `configs/universe/universe_<MKT>.csv` | RIC universe per market | `data_dumper`, `BaseAlpha.get_universe()` |

All TOML validated via **Pydantic v2**. `ConfigValidationError` raised on any invalid/missing field.

### 3.2 System Config — `configs/system.toml`

```toml
[storage]
shared_drive_path = "/mnt/shared/alphaflow"
parquet_cold_path = "/mnt/shared/alphaflow/cold"
log_path          = "/mnt/shared/alphaflow/logs"

[kdb]
host = "localhost"
port = 5001

[bloomberg]
timeout_seconds = 30

[s3]
bucket_name  = "my-alpha-bucket"
region       = "ap-east-1"
profile_name = ""

[scheduler]
timezone = "Asia/Hong_Kong"

[markets.HK]
available_data_types = ["depth_open", "depth_close", "depth_intraday", "indic_open", "indic_close", "minute_bar", "bbg", "southbound"]
sessions = [{ start = "09:30", end = "12:00" }, { start = "13:00", end = "16:00" }]

[markets.CN]
available_data_types = ["depth_open", "depth_close", "depth_intraday", "indic_open", "indic_close", "minute_bar", "bbg", "southbound"]
sessions = [{ start = "09:30", end = "11:30" }, { start = "13:00", end = "15:00" }]

[markets.SG]
available_data_types = ["depth_open", "depth_close", "depth_intraday", "indic_open", "indic_close", "minute_bar", "bbg"]
sessions = [{ start = "09:00", end = "12:00" }, { start = "13:00", end = "17:00" }]

[markets.JP]
available_data_types = ["depth_open", "depth_close", "depth_intraday", "indic_open", "indic_close", "minute_bar", "bbg"]
sessions = [{ start = "09:00", end = "15:30" }]

[markets.KR]
available_data_types = ["depth_open", "depth_close", "depth_intraday", "minute_bar", "bbg"]
sessions = [{ start = "09:00", end = "15:30" }]

[markets.AU]
available_data_types = ["depth_open", "depth_close", "depth_intraday", "minute_bar", "bbg"]
sessions = [{ start = "10:00", end = "16:00" }]

[markets.TW]
available_data_types = ["depth_open", "depth_close", "depth_intraday", "minute_bar", "bbg"]
sessions = [{ start = "09:00", end = "13:30" }]
```

```python
class SessionConfig(BaseModel):
    start : str
    end   : str

class MarketConfig(BaseModel):
    available_data_types : list[str]
    sessions             : list[SessionConfig] = Field(min_length=1)

class StorageConfig(BaseModel):
    shared_drive_path : str
    parquet_cold_path : str
    log_path          : str

class KdbConfig(BaseModel):
    host : str
    port : int = Field(gt=0, lt=65536)

class BloombergConfig(BaseModel):
    timeout_seconds : int = Field(default=30, gt=0)

class S3Config(BaseModel):
    bucket_name  : str
    region       : str
    profile_name : str = ""

class SchedulerConfig(BaseModel):
    timezone : str

class SystemConfig(BaseModel):
    storage   : StorageConfig
    kdb       : KdbConfig
    bloomberg : BloombergConfig
    s3        : S3Config
    scheduler : SchedulerConfig
    markets   : dict[str, MarketConfig]
```

### 3.3 Production Config — `configs/production.toml`

```toml
[models]
active_model_ids = ["hk_momentum", "cn_factor"]

[scheduler]
daily_refresh_time            = "17:30"
intraday_update_interval_sec = 5

[threading]
max_worker_threads   = 4
daily_thread_name    = "daily_aggregator"
intraday_thread_name = "intraday_aggregator"
restart_on_failure   = true
max_restart_attempts = 3

[output]
kdb_table_daily      = "alphaflow_scores_daily"
kdb_table_intraday   = "alphaflow_scores_intraday"
parquet_partition_by = "date"
```

```python
class ProductionModelConfig(BaseModel):
    active_model_ids : list[str] = Field(min_length=1)

class ProductionSchedulerConfig(BaseModel):
    daily_refresh_time           : str
    intraday_update_interval_sec : int = Field(gt=0)

    @field_validator("daily_refresh_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str: ...

class ThreadingConfig(BaseModel):
    max_worker_threads   : int  = Field(default=4, gt=0)
    daily_thread_name    : str  = "daily_aggregator"
    intraday_thread_name : str  = "intraday_aggregator"
    restart_on_failure   : bool = True
    max_restart_attempts : int  = Field(default=3, ge=1)

class ProductionOutputConfig(BaseModel):
    kdb_table_daily      : str
    kdb_table_intraday   : str
    parquet_partition_by : str = "date"

class ProductionConfig(BaseModel):
    models    : ProductionModelConfig
    scheduler : ProductionSchedulerConfig
    threading : ThreadingConfig
    output    : ProductionOutputConfig
```

### 3.4 Optimizer Config — `configs/optimizer.toml`

```toml
[mean_variance]
max_single_alpha_weight = 0.8
min_single_alpha_weight = 0.0
risk_aversion           = 1.0

risk_model = "barra_ase2s"
```

```python
class MeanVarianceConfig(BaseModel):
    max_single_alpha_weight : float = Field(default=0.8, ge=0.0, le=1.0)
    min_single_alpha_weight : float = Field(default=0.0, ge=0.0, le=1.0)
    risk_aversion           : float = Field(default=1.0, gt=0.0)

class OptimizerConfig(BaseModel):
    mean_variance : MeanVarianceConfig
    risk_model    : str
```
This applies to **all** models — parameters, not model selection (model_ids passed directly to `OptimizerRunner.run()`).

### 3.5 Registry Config — `configs/registry.toml`

```toml
[[alphas]]
alpha_id    = "momentum_price_21d"
alpha_type  = "low"
config_file = "configs/alphas/low/momentum_price_21d.toml"
active      = true

[[alphas]]
alpha_id    = "order_imbalance_1430"
alpha_type  = "high"
config_file = "configs/alphas/high/order_imbalance_1430.toml"
active      = true

[[alphas]]
alpha_id    = "open_to_close_1d"
alpha_type  = "rtn"
config_file = "configs/alphas/rtn/open_to_close_1d.toml"
active      = true
```

```python
from typing import Literal

AlphaType = Literal["high", "mid", "low", "rtn"]

class RegistryEntry(BaseModel):
    alpha_id    : str
    alpha_type  : AlphaType
    config_file : str
    active      : bool

class RegistryConfig(BaseModel):
    alphas : list[RegistryEntry]

    def get_active(self) -> list[RegistryEntry]: ...
    def get_by_id(self, alpha_id: str) -> RegistryEntry | None: ...
    def get_by_type(self, alpha_type: AlphaType) -> list[RegistryEntry]: ...
```

### 3.6 Model Config — `configs/models/<model_id>.toml` (Final Version)

A model = a combination of alpha "strategies" that enter/exit together. Only `rtn_alpha_id` carries entry/exit timing (other alpha types have no comparable field).

```toml
model_id        = "hk_momentum"
market          = "HK"
alpha_ids       = ["momentum_price_21d", "earnings_revision"]
rtn_alpha_id    = "open_to_close_1d"
start_date      = "20230101"
end_date        = "20231231"
save_to_parquet = true

[alpha_weights]
# Empty until OptimizerRunner.run() populates it
momentum_price_21d = 0.62
earnings_revision  = 0.38
```

```python
class ModelConfig(BaseModel):
    model_id        : str
    market          : Market
    alpha_ids       : list[str] = Field(min_length=1)
    rtn_alpha_id    : str
    start_date      : str
    end_date        : str
    save_to_parquet : bool = False
    alpha_weights   : dict[str, float] = {}

    @model_validator(mode="after")
    def validate_model_id_matches_filename(self) -> "ModelConfig": ...

    @model_validator(mode="after")
    def validate_weights_keys(self) -> "ModelConfig":
        """If alpha_weights non-empty, keys must exactly match alpha_ids."""
        ...
```

### 3.7 Per-Alpha Config

#### Shared Base — `core/config/alpha_config/base_config.py`

```python
class BaseAlphaConfig(BaseModel):
    """
    Shared fields for all alpha types.
    required_inputs: data_type names → maps to raw_data/{market}/{data_type}/.
    Fields used inside compute() are specified in the Python subclass, not here.
    """
    name              : str
    description       : str
    category          : str
    tags              : list[str]    = []
    formula           : str
    handler_class     : str
    alpha_type        : AlphaType
    market_scope      : list[Market] = Field(min_length=1)
    required_inputs   : list[str]    = Field(min_length=1)
    custom_exclusions : list[str]    = []
```

#### Type-Specific Models

```python
class HighAlphaConfig(BaseAlphaConfig): ...   # Field-level details TBD when type is implemented
class MidAlphaConfig(BaseAlphaConfig):  ...   # Field-level details TBD when type is implemented
class LowAlphaConfig(BaseAlphaConfig):  ...   # Field-level details TBD when type is implemented
```

#### RtnAlphaConfig — Fully Specified

```python
class PricePoint(BaseModel):
    """
    For open/close: price_type + optional session (am/pm for HK/CN/SG; omit for single-session markets)
    For vwap: price_type='vwap' + vwap_start/vwap_end time range
    """
    price_type : str                    # "open" | "close" | "vwap"
    session    : str | None = None      # "am" | "pm" | None
    vwap_start : str | None = None      # "HH:MM"
    vwap_end   : str | None = None      # "HH:MM"

    @model_validator(mode="after")
    def validate_fields(self) -> "PricePoint": ...

class RtnAlphaConfig(BaseAlphaConfig):
    entry       : PricePoint
    exit        : PricePoint
    horizon     : int                   # Days: exit_day - entry_day. 0 = intraday.
    return_type : list[str]             # ["raw"] | ["hedged"] | ["raw", "hedged"]

    @field_validator("return_type")
    @classmethod
    def validate_return_type(cls, v: list[str]) -> list[str]: ...

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        assert v >= 0
        return v
```

**Example RtnAlpha TOML:**
```toml
[identity]
name          = "open_to_close_1d"
description   = "Next day AM open to PM close return, raw and hedged"
category      = "return"
formula       = "close(t+1, pm) / open(t+1, am) - 1, hedged by index return"
handler_class = "alphaflow.signal_builder.alphas.rtn.open_to_close_1d.OpenToClose1D"
alpha_type    = "rtn"
market_scope  = ["HK", "CN", "SG"]

[data]
required_inputs = ["minute_bar"]

[entry]
price_type = "open"
session    = "am"

[exit]
price_type = "close"
session    = "pm"

horizon     = 1
return_type = ["raw", "hedged"]
```

### 3.8 Universe CSV — `configs/universe/universe_<MARKET>.csv`

```
RIC
0700.HK
0941.HK
```

Single `RIC` column only. `BBL` never stored — handled internally by `BBGClient` via `desktool`.

### 3.9 Cross-Config Validation (Complete)

| Check | Rule |
|---|---|
| Model alpha_ids in registry | Every `alpha_id` in a `ModelConfig` must exist in `registry.toml` with `active=true` |
| Model rtn_alpha_id in registry | `rtn_alpha_id` must exist with `alpha_type="rtn"`, `active=true` |
| Production models exist | Every `model_id` in `production.toml` must have `configs/models/<model_id>.toml` |
| alpha_weights keys match | If non-empty, keys must exactly match `alpha_ids` |
| Handler class importable | Must resolve to a subclass of the correct parent (`HighAlpha`/`MidAlpha`/`LowAlpha`/`RtnAlpha`) |
| alpha_type consistency | Per-alpha TOML `alpha_type` must match `registry.toml` entry |
| model_id matches filename | `model_id` field must match TOML file stem |
| required_inputs valid | Each entry must be in `available_data_types` for the alpha's `market_scope` markets |

### 3.10 Config Loading at Runtime

| Runner | Configs loaded |
|---|---|
| `DataDumperRunner` | `SystemConfig` |
| `SignalBuilderRunner` | `SystemConfig`, `RegistryConfig`, typed `AlphaConfig` per alpha |
| `StrategyBacktesterRunner` | `SystemConfig`, `RegistryConfig`, typed `AlphaConfig` per alpha |
| `OptimizerRunner` | `SystemConfig`, `OptimizerConfig`, `RegistryConfig`, `ModelConfig` per model |
| `AlphaAggregatorRunner` | `SystemConfig`, `ProductionConfig`, `RegistryConfig`, `ModelConfig` per active model |
| `PostTradeAnalyzerRunner` | `SystemConfig`, `RegistryConfig`, `ModelConfig` per model |

---

## 4. Core Layer

### 4.1 Global Rule: RIC as Primary Identifier

**`RIC`** is used for all joins, indexing, and storage throughout the entire project.
**`BBL`** (Bloomberg ticker) is used ONLY internally inside `BBGClient`, mapped via `desktool` functions, and NEVER exposed to any caller outside that client.

### 4.2 Data Access Layer

#### Exceptions — `core/data_access/exceptions.py`

```python
class PITViolationError(Exception): ...
class DataNotFoundError(Exception): ...
class DataSourceError(Exception): ...
```

#### KDBClient — `core/data_access/kdb_client.py`

```python
class KDBClient:
    """Wrapper around pykx. All outputs RIC-keyed."""

    def __init__(self, config: KdbConfig) -> None: ...

    def query(
        self, table: str, rics: list[str], fields: list[str],
        start_date: date, as_of_date: date,
    ) -> pd.DataFrame:
        """Returns rows where timestamp <= as_of_date. RIC-keyed."""
        ...

    def query_latest(self, table: str, rics: list[str], fields: list[str]) -> pd.DataFrame:
        """Latest data — production mode, no PIT filter."""
        ...
```

#### BBGClient — `core/data_access/bloomberg_client.py`

```python
class BBGClient:
    """
    Wrapper around xbbg/desktool. BBL↔RIC mapping handled INTERNALLY
    via desktool functions. Callers pass/receive RIC only.
    """

    def __init__(self, config: BloombergConfig) -> None: ...

    def bdh(self, rics: list[str], fields: list[str], start_date: date, end_date: date) -> pd.DataFrame: ...
    def bdp(self, rics: list[str], fields: list[str]) -> pd.DataFrame: ...
```

#### S3Client — `core/data_access/s3_client.py`

```python
class S3Client:
    def __init__(self, config: S3Config) -> None: ...
    def read_csv(self, key: str, as_of_date: date) -> pd.DataFrame: ...
    def read_parquet(self, key: str, as_of_date: date) -> pd.DataFrame: ...
```

#### PITDataManager — `core/data_access/pit_manager.py`

```python
class PITMode:
    RESEARCH   = "research"      # as_of_date is a past date — strict filtering enforced
    PRODUCTION = "production"    # as_of_date = now() — no filtering, minimal latency

class PITDataManager:
    """
    Central data gateway. All data reads anywhere in the system go through this.
    Internally holds KDBClient, BBGClient, S3Client. All outputs RIC-keyed.
    """

    def __init__(
        self, system_config: SystemConfig,
        as_of_date: date | None = None, mode: str = PITMode.PRODUCTION,
    ) -> None:
        self._kdb = KDBClient(system_config.kdb)
        self._bbg = BBGClient(system_config.bloomberg)
        self._s3  = S3Client(system_config.s3)

    @classmethod
    def for_research(cls, system_config: SystemConfig, as_of_date: date) -> "PITDataManager": ...

    @classmethod
    def for_production(cls, system_config: SystemConfig) -> "PITDataManager": ...

    def get_ohlcv(self, rics: list[str], market: str, start_date: date, fields: list[str] | None = None) -> pd.DataFrame: ...
    def get_tick(self, rics: list[str], market: str, start_date: date) -> pd.DataFrame: ...
    def get_depth(self, rics: list[str], market: str, start_date: date, snapshot: str) -> pd.DataFrame: ...
    def get_indic(self, rics: list[str], market: str, start_date: date, snapshot: str) -> pd.DataFrame: ...
    def get_minute_bar(self, rics: list[str], market: str, start_date: date) -> pd.DataFrame: ...
    def get_risk_factors(self, rics: list[str], market: str, start_date: date, model: str = "ASE2S") -> pd.DataFrame: ...
    def get_bbg_reference(self, rics: list[str], fields: list[str]) -> pd.DataFrame: ...
    def get_bbg_history(self, rics: list[str], fields: list[str], start_date: date) -> pd.DataFrame: ...
    def get_s3_data(self, key: str, file_type: str = "csv") -> pd.DataFrame: ...
```

### 4.3 Alpha Base Layer

#### Market Enum

```python
from enum import StrEnum

class Market(StrEnum):
    HK = "HK"; CN = "CN"; SG = "SG"; JP = "JP"; KR = "KR"; AU = "AU"; TW = "TW"
```

#### BaseAlpha — `core/alpha_base/base_alpha.py`

```python
class BaseAlpha(ABC):
    """
    Abstract base. All alphas produce pd.Series indexed by RIC.
    """

    def __init__(self, config: BaseAlphaConfig, data: PITDataManager, universe_dir: str) -> None:
        self.config         = config
        self.data           = data
        self.universe_dir   = universe_dir
        self._working_data  : pd.DataFrame | None = None   # injected by SignalBuilderRunner
        self.validate_config()

    @property
    def alpha_id(self) -> str: return self.config.name

    @property
    def alpha_type(self) -> str: return self.config.alpha_type

    @property
    def market_scope(self) -> list[str]: return self.config.market_scope

    @abstractmethod
    def compute(self, market: str, as_of_date: date) -> pd.Series:
        """
        Uses self._working_data (pre-loaded, RIC-joined by SignalBuilderRunner).
        Returns pd.Series indexed by RIC. Missing-data stocks excluded (not zero-filled here).
        """
        ...

    def normalize(self, raw_signal: pd.Series) -> pd.Series:
        """
        Default: identity (no transformation). Concrete subclasses may override
        with (raw_signal - SHIFT) / SCALE, defined as class-level constants.
        """
        return raw_signal

    def run(self, market: str, as_of_date: date) -> pd.Series:
        """Full pipeline: compute() → normalize()."""
        raw = self.compute(market, as_of_date)
        return self.normalize(raw)

    def get_universe(self, market: str) -> pd.Series:
        """Load universe CSV. Returns pd.Series of RIC strings."""
        ...

    def validate_config(self) -> None:
        """Raises ConfigValidationError on invalid config."""
        ...
```

#### Child Classes

```python
class HighAlpha(BaseAlpha):
    """Tick-level intraday. Type-specific compute contract TBD."""
    ...

class MidAlpha(BaseAlpha):
    """Interval-based intraday. Type-specific compute contract TBD."""
    ...

class LowAlpha(BaseAlpha):
    """Daily / multi-day. Type-specific compute contract TBD."""
    ...

class RtnAlpha(BaseAlpha):
    """
    Return signal — used both as a standalone alpha AND as the realized
    return source for strategy_backtester / post_trade_analyzer.

    Intraday sanity check (horizon=0 only): validates entry/exit
    price_type/session against market session config (SystemConfig.markets).
    Raises ConfigValidationError if inconsistent.
    """

    def __init__(
        self, config: RtnAlphaConfig, data: PITDataManager, universe_dir: str,
        market_config: dict[str, MarketConfig] | None = None,
    ) -> None:
        super().__init__(config, data, universe_dir)
        self.market_config = market_config

    @abstractmethod
    def compute(self, market: str, as_of_date: date) -> pd.Series:
        """Returns pd.Series indexed by RIC. Values = forward return (float)."""
        ...

    def _validate_intraday_session(self, market: str) -> None:
        """Called when horizon=0."""
        ...
```

### 4.4 AlphaRegistry — `core/alpha_base/registry.py`

```python
class AlphaRegistry:
    def __init__(
        self, registry_config: RegistryConfig, model_configs: list[ModelConfig],
        production_config: ProductionConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
    ) -> None:
        """Runs all cross-config validation on construction."""
        ...

    @classmethod
    def from_paths(
        cls, registry_path: str = "configs/registry.toml", models_dir: str = "configs/models/",
        production_path: str | None = None, optimizer_path: str | None = None,
    ) -> "AlphaRegistry": ...

    def get_active(self) -> list[RegistryEntry]: ...
    def get_by_id(self, alpha_id: str) -> RegistryEntry | None: ...
    def get_by_type(self, alpha_type: AlphaType) -> list[RegistryEntry]: ...
    def get_model(self, model_id: str) -> ModelConfig | None: ...
    def get_alphas_for_model(self, model_id: str) -> list[RegistryEntry]: ...
```

### 4.5 AlphaLoader — `core/alpha_base/loader.py`

```python
class AlphaLoader:
    _loader_map: dict[AlphaType, type["TypedAlphaLoader"]] = {
        "high": HighAlphaLoader, "mid": MidAlphaLoader,
        "low": LowAlphaLoader, "rtn": RtnAlphaLoader,
    }

    @classmethod
    def load(cls, entry: RegistryEntry, data: PITDataManager, universe_dir: str) -> BaseAlpha:
        loader = cls._loader_map[entry.alpha_type]()
        return loader.load(entry.config_file, data, universe_dir)


class TypedAlphaLoader(ABC):
    @abstractmethod
    def load_config(self, config_file: str) -> BaseAlphaConfig: ...

    @abstractmethod
    def resolve_class(self, handler_class: str) -> type[BaseAlpha]: ...

    def load(self, config_file: str, data: PITDataManager, universe_dir: str) -> BaseAlpha:
        config = self.load_config(config_file)
        cls    = self.resolve_class(config.handler_class)
        return cls(config=config, data=data, universe_dir=universe_dir)

# HighAlphaLoader, MidAlphaLoader, LowAlphaLoader, RtnAlphaLoader
# each implement load_config() and resolve_class() for their respective type.
```

---

## 5. data_dumper

**Purpose:** Manual backfill tool (not scheduled) — pulls raw data from kdb+, S3, Bloomberg and writes daily CSV files to shared drive, using universe CSV (RIC only) for ticker scope.

### 5.1 Supported Data Types

| Data Type | Source | Notes |
|---|---|---|
| `depth_open`, `depth_close` | kdb+ | Snapshot at session open/close |
| `depth_intraday` | kdb+ | Single daily file, `timestamp` column for all intraday snapshots |
| `indic_open`, `indic_close` | kdb+ | Indicative prices |
| `minute_bar` | kdb+ | Minute-level OHLCV |
| `intropic` | S3 | Intrinsic data |
| `southbound` | S3 | Southbound flow data |
| `bbg` | Bloomberg | **Placeholder — `NotImplementedError`, implemented later via desktool** |

### 5.2 Key Classes

```python
class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, data_type: str, market: str, rics: list[str], date: date) -> pd.DataFrame: ...
    @property
    @abstractmethod
    def supported_data_types(self) -> list[str]: ...


class KDBExtractor(BaseExtractor):
    """Handles: depth_open, depth_close, depth_intraday, indic_open, indic_close, minute_bar."""
    def __init__(self, kdb_client: KDBClient) -> None: ...
    def extract(self, data_type, market, rics, date) -> pd.DataFrame: ...  # dispatches internally


class S3Extractor(BaseExtractor):
    """Handles: intropic, southbound."""
    def __init__(self, s3_client: S3Client) -> None: ...


class BBGExtractor(BaseExtractor):
    """TODO: implement via desktool. Raises NotImplementedError for now."""


class CSVWriter:
    """Writes to raw_data/{market}/{data_type}/YYYYMMDD.csv. RIC primary key."""
    def write(self, df: pd.DataFrame, market: str, data_type: str, date: date) -> None: ...
    def file_exists(self, market: str, data_type: str, date: date) -> bool: ...


class DataDumperRunner:
    def __init__(self, system_config: SystemConfig) -> None: ...

    @classmethod
    def from_config(cls, system_config_path: str = "configs/system.toml") -> "DataDumperRunner": ...

    def run(
        self, market: str, date: str | None = None, start: str | None = None,
        end: str | None = None, source: str | None = None, overwrite: bool = False,
    ) -> None:
        """
        source=None → dump all available_data_types for market (from system.toml).
        Ticker scope loaded from configs/universe/universe_{market}.csv.
        """
        ...
```

### 5.3 Usage

```python
runner = DataDumperRunner.from_config()
runner.run(market="HK", date="20240115")
runner.run(market="HK", start="20240101", end="20240131")
runner.run(market="CN", date="20240115", source="minute_bar")
```

---

## 6. signal_builder

**Purpose:** Computes alpha signal values. Exposes two methods sharing the same core logic.

### 6.1 Universe Resolution (Implicit)

No explicit universe call. For each alpha:
1. Load each `required_input` from `raw_data/{market}/{data_type}/YYYYMMDD.csv`
2. Inner-join all DataFrames **on RIC**
3. Drop rows with nulls in required fields
4. Result set as `alpha._working_data` before `compute()` runs

### 6.2 SignalBuilderRunner

```python
class SignalBuilderRunner:
    def __init__(self, system_config: SystemConfig, registry: AlphaRegistry) -> None: ...

    @classmethod
    def from_config(
        cls, system_config_path: str = "configs/system.toml",
        registry_path: str = "configs/registry.toml", models_dir: str = "configs/models/",
    ) -> "SignalBuilderRunner": ...

    def build_one_day_alpha(
        self, alpha_id: str | list[str], market: str, date: str,
        save: bool = False, overwrite: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        CORE method. Per alpha:
          1. Load required_inputs, inner-join on RIC, drop nulls
          2. Set alpha._working_data
          3. alpha.run(market, date) → compute() → normalize()
          4. Build score DataFrame: RIC, raw_signal, score
          5. If save=True: write parquet (skip if exists & overwrite=False)
        Returns dict[alpha_id, DataFrame(RIC, raw_signal, score)] — no date column.
        """
        ...

    def build_multi_day_alpha(
        self, alpha_id: str | list[str], market: str, dates: list[str],
        save: bool = False, overwrite: bool = False, n_jobs: int = 1,
    ) -> dict[str, pd.DataFrame]:
        """
        Wraps build_one_day_alpha across explicit date list.
        n_jobs > 1 → ProcessPoolExecutor, one process per date.
        Returns dict[alpha_id, DataFrame(date, RIC, raw_signal, score)].
        """
        ...
```

### 6.3 `_working_data` Injection Pattern

```python
# Inside build_one_day_alpha(), per alpha:
working_df          = self._load_and_join(alpha.config.required_inputs, market, date)
alpha._working_data = working_df    # RIC-keyed DataFrame
score               = alpha.run(market, as_of_date)
```

`BaseAlpha._working_data : pd.DataFrame | None = None` declared on base class.

### 6.4 ScoreWriter — `signal_builder/storage/score_writer.py`
```python
class ScoreWriter:
    """Writes to alpha_scores/{market}/{alpha_id}/YYYYMMDD.parquet. Columns: RIC, raw_signal, score."""
    def write(self, scores: pd.DataFrame, market: str, alpha_id: str, date: date) -> None: ...
    def file_exists(self, market: str, alpha_id: str, date: date) -> bool: ...
    def read(self, market: str, alpha_id: str, date: date) -> pd.DataFrame: ...
    def read_range(self, market: str, alpha_id: str, start: date, end: date) -> pd.DataFrame: ...
```

### 6.5 Concrete Alpha Example

```python
class MomentumPriceAlpha(LowAlpha):
    SHIFT : float = 0.0
    SCALE : float | None = None    # None → compute std from data

    def compute(self, market: str, as_of_date: date) -> pd.Series:
        prices = self._working_data.set_index("RIC")["close"]
        # ... compute 21d momentum, skip 1 day ...
        return raw_signal      # pd.Series indexed by RIC

    def normalize(self, raw_signal: pd.Series) -> pd.Series:
        scale = self.SCALE if self.SCALE is not None else raw_signal.std()
        if scale == 0:
            return pd.Series(0.0, index=raw_signal.index)
        return (raw_signal - self.SHIFT) / scale
```

### 6.6 Usage

```python
runner = SignalBuilderRunner.from_config()

# Production — single alpha, single date, no save
result = runner.build_one_day_alpha("momentum_price_21d", "HK", "20240115")

# Research — date range, save to disk, parallel
result = runner.build_multi_day_alpha(
    alpha_id="all", market="CN",
    dates=["20240101", "20240102", "20240103"],
    save=True, overwrite=True, n_jobs=4,
)
```

---

## 7. strategy_backtester

**Purpose:** Evaluate single alpha performance historically. **Reads only** — never computes alpha scores itself.

### 7.1 Inputs

| Input | Source |
|---|---|
| Alpha scores | `alpha_scores/{market}/{alpha_id}/YYYYMMDD.parquet` |
| Return data | `alpha_scores/{market}/{rtn_alpha_id}/YYYYMMDD.parquet` |
| Barra factors | kdb+ via `PITDataManager.get_risk_factors()` |

### 7.2 Metric Definitions

**IC/ICIR:** For each date `t`, compute Spearman correlation of alpha score vs. forward return, cross-sectionally across all stocks on that date → daily IC series. Then: `mean_IC`, `std_IC`, `ICIR = mean_IC/std_IC`.

**Regression:** Pooled cross-sectional OLS over the WHOLE period (all dates+stocks at once): `forward_return ~ alpha_score`. Output: `r_squared`, `t_stat`, `coefficient`, `intercept`, `n_stocks`.

**Performance:** Sharpe (from IC series), max drawdown (cumulative IC), turnover (rank change day-to-day).

**Decile spread:** `mean(D10 return) - mean(D1 return)`.

**Barra exposure:** Score-weighted average factor exposure across stocks, averaged over the period.

### 7.3 Result Table Shape

One table per alpha. **Rows = measures, columns = `rtn_alpha_id`s** (one column per return type tested):

```
measure         | close_to_close | close_to_1d_vwap
----------------+----------------+------------------
mean_IC         | 0.045          | 0.038
std_IC          | 0.055          | 0.054
ICIR            | 0.82           | 0.71
sharpe          | 1.23           | 1.05
turnover        | 0.18           | 0.18
max_drawdown    | -0.12          | -0.09
decile_spread   | 0.032          | 0.028
r_squared       | 0.021          | 0.018
t_stat          | 3.4            | 2.9
coefficient     | 0.15           | 0.12
intercept       | 0.001          | 0.001
n_stocks        | 487            | 487
barra_mkt       | 0.95           | 0.93
barra_size      | -0.12          | -0.11
... (one row per Barra factor)
```

Saved to: `backtest_results/{market}/{alpha_id}_YYYYMMDD-YYYYMMDD.parquet` — **always saved**, `overwrite=False` raises `FileExistsError`.

### 7.4 Key Classes

```python
class ICCalculator:
    def compute_daily_ic(self, alpha_scores: pd.DataFrame, return_scores: pd.DataFrame) -> pd.Series: ...
    def summarize(self, ic_series: pd.Series) -> dict[str, float]: ...   # mean_IC, std_IC, ICIR

class PerformanceMetrics:
    def compute_sharpe(self, ic_series: pd.Series) -> float: ...
    def compute_max_drawdown(self, ic_series: pd.Series) -> float: ...
    def compute_turnover(self, alpha_scores: pd.DataFrame) -> float: ...

class DecileAnalyzer:
    def compute_decile_returns(self, alpha_scores, return_scores) -> pd.DataFrame: ...
    def compute_spread(self, decile_returns: pd.DataFrame) -> float: ...

class CrossSectionalOLS:
    def fit(self, alpha_scores: pd.DataFrame, return_scores: pd.DataFrame) -> dict[str, float]: ...

class BarraExposure:
    def __init__(self, data: PITDataManager) -> None: ...
    def compute(self, alpha_scores: pd.DataFrame, market: str, dates: list[date]) -> dict[str, float]: ...

class BacktestEngine:
    """Orchestrates all metrics for one alpha vs. one rtn_alpha_id."""
    def run(self, alpha_scores, return_scores, market, dates) -> dict[str, float]: ...

class ScoreLoader:
    """Reads pre-computed scores from disk (shared with optimizer)."""
    def load_alpha_scores(self, market, alpha_id, dates) -> pd.DataFrame: ...
    def load_return_scores(self, market, rtn_alpha_id, dates) -> pd.DataFrame: ...

class ReportGenerator:
    def assemble(self, results: dict[str, dict[str, float]]) -> pd.DataFrame: ...
    def save(self, result_df, market, alpha_id, start_date, end_date, overwrite=False) -> None: ...


class StrategyBacktesterRunner:
    def run(
        self, alpha_id: str | list[str], market: str, dates: list[str],
        rtn_alpha_id: str | list[str], overwrite: bool = False, n_jobs: int = 1,
    ) -> dict[str, pd.DataFrame]:
        """Multiple alphas run in parallel when n_jobs > 1."""
        ...
```

### 7.5 Usage

```python
runner = StrategyBacktesterRunner.from_config()
result = runner.run(
    alpha_id="momentum_price_21d", market="HK",
    dates=["20230101", ...], rtn_alpha_id=["open_to_close_1d", "close_to_1d_vwap"],
)
```

---

## 8. optimizer

**Purpose:** Treats each alpha within a model as an independent strategy with a daily return series, allocates weights via **mean-variance optimization**. Correlation feeds the covariance matrix — no hard exclusion of correlated alphas. ML mode was considered and dropped (weights should be stable, not time-varying).

### 8.1 Strategy Return Builder

```python
class StrategyReturnBuilder:
    """
    daily_return(t) = sum_i( score_i(t) * return_i(t) ) across matched stocks.
    No dollar-neutral rescaling — use a hedged rtn_alpha_id if needed.
    """
    def build(self, alpha_scores: pd.DataFrame, return_scores: pd.DataFrame) -> pd.Series: ...
    def build_for_model(self, alpha_ids, market, rtn_alpha_id, dates, score_loader) -> pd.DataFrame:
        """Returns DataFrame: index=date, columns=alpha_id."""
        ...
```

### 8.2 Correlation & Standalone Metrics

```python
class CorrelationAnalyzer:
    def compute(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        return daily_returns.corr()
    def average_correlation(self, corr_matrix, alpha_id) -> float: ...

class StandaloneMetrics:
    def compute(self, daily_returns: pd.Series) -> dict[str, float]:
        """standalone_sharpe, standalone_std, standalone_mean_return."""
        ...
```

### 8.3 Mean-Variance Optimizer

```python
class MeanVarianceOptimizer:
    """
    maximize: w^T * mu - risk_aversion * w^T * Sigma * w
    subject to: min_weight <= w_i <= max_weight, sum(w_i) = 1
    Solved via scipy.optimize (SLSQP).
    """
    def __init__(self, config: MeanVarianceConfig) -> None: ...
    def optimize(self, daily_returns: pd.DataFrame) -> pd.Series: ...    # index=alpha_id
    def compute_portfolio_stats(self, daily_returns, weights) -> dict[str, float]: ...
```

### 8.4 Barra Decomposer

```python
class BarraDecomposer:
    def compute_standalone(self, alpha_scores, market, dates) -> dict[str, float]: ...
    def compute_combined(self, standalone_exposures: dict[str, dict], weights: pd.Series) -> dict[str, float]: ...
```

### 8.5 Storage

```python
class ModelConfigWriter:
    """Updates ONLY [alpha_weights] in configs/models/{model_id}.toml, preserves everything else."""
    def write_weights(self, model_id: str, models_dir: str, weights: pd.Series) -> None: ...

class ResultWriter:
    """One wide table: rows=alpha_id + PORTFOLIO summary row."""
    def build_table(
        self, weights, standalone_metrics, avg_correlations,
        standalone_barra, combined_stats, combined_barra,
    ) -> pd.DataFrame:
        """
        row_id             | weight | sharpe | std   | avg_corr | barra_mkt | ...
        momentum_price_21d | 0.62   | 1.10   | 0.015 | 0.35     | 0.92      |
        earnings_revision  | 0.38   | 0.85   | 0.018 | 0.35     | 0.88      |
        PORTFOLIO          | 1.00   | 1.35   | 0.012 | -        | 0.90      |
        """
        ...
    def save(self, result_df, market, model_id, overwrite=False) -> None:
        """Saves to optimizer_results/{market}/{model_id}.parquet."""
        ...
```

### 8.6 OptimizerRunner

```python
class OptimizerRunner:
    def run(self, model_id: str | list[str], overwrite: bool = False) -> dict[str, pd.DataFrame]:
        """
        Per model_id (all params read from that model's own TOML):
          1. Load ModelConfig
          2. Build daily return series per alpha
          3. Compute correlation matrix + standalone metrics + standalone Barra
          4. Run MeanVarianceOptimizer.optimize() → weights
          5. Compute combined portfolio stats + combined Barra exposure
          6. Write weights back to configs/models/{model_id}.toml
          7. Save audit table to optimizer_results/{market}/{model_id}.parquet
        """
        ...
```

### 8.7 Usage

```python
runner = OptimizerRunner.from_config()
result = runner.run(model_id="hk_momentum")
# configs/models/hk_momentum.toml updated in place with new [alpha_weights]
```

---

## 9. alpha_aggregator

**Purpose:** Production runtime, split into **Producer** (background thread per market, computes individual alpha scores only) and **Consumer** (on-demand, combines scores into a model's composite).

### 9.1 Producer Side

```python
def resolve_market_alpha_ids(market, production_config, registry) -> list[str]:
    """
    Union of all alpha_ids (+ rtn_alpha_id) across active models in this market.
    Computed ONCE at thread startup, not recalculated during run.
    """
    ...

class DailyAlphaCache:
    """Caches LowAlpha/RtnAlpha results in memory for the current trading day."""
    def is_fresh(self, current_date: date) -> bool: ...
    def get(self, alpha_id: str) -> pd.Series | None: ...
    def set(self, alpha_id: str, scores: pd.Series, current_date: date) -> None: ...
    def force_refresh(self) -> None: ...

class MarketProducerThread:
    """
    ONE instance per market. Continuous loop:
      1. If daily cache stale: compute daily alphas once (SignalBuilderRunner, save=False)
      2. Compute intraday alphas fresh every iteration
      3. Write ONLY individual scores to kdb+ (NO combination here)
      4. On failure: retry up to max_restart_attempts, then log CRITICAL +
         send_critical_alert(), stop THIS thread only (other markets unaffected)
    """
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run(self) -> None: ...
    def force_refresh_daily(self) -> None: ...

class KDBWriter:
    """Writes ONLY individual alpha scores. Schema: timestamp, market, alpha_id, RIC, raw_signal, score, frequency."""
    def write_individual_scores(self, market, daily_scores, intraday_scores) -> None: ...
    def query_latest_scores(self, market, alpha_ids) -> dict[str, pd.Series]: ...
```

### 9.2 Consumer Side

```python
class ModelConsumer:
    """
    ON-DEMAND — NOT a background thread. Instantiated per call.
    """

    def get_composite_score(self, model_id: str) -> pd.Series:
        """
        1. Load ModelConfig (alpha_ids, alpha_weights, save_to_parquet)
        2. Query latest individual scores from kdb+ (KDBWriter.query_latest_scores)
        3. Union of RICs; missing alpha score for a RIC → 0
        4. composite(RIC) = sum(weight_i * score_i(RIC)) — NO renormalization
        5. If save_to_parquet: write individual + composite to
           alpha_scores/{model_id}/YYYYMMDD.parquet
        6. Return composite pd.Series (RIC-indexed)
        """
        ...

class ParquetWriter:
    """Writes individual scores + composite to alpha_scores/{model_id}/YYYYMMDD.parquet."""
    def write(self, model_id, individual_scores, composite_score, date) -> None: ...
```

### 9.3 Alerting

```python
def send_critical_alert(subject: str, message: str) -> None:
    """
    Logs CRITICAL via external setup_logger, then delegates to an external
    alert function (email + beep) — exact implementation provided by another
    shared package. This is the integration point only.
    """
    log.critical(f"{subject}: {message}")
    # TODO: call external alert function
```

### 9.4 AlphaAggregatorRunner

```python
class AlphaAggregatorRunner:
    def start_producers(self) -> None:
        """One MarketProducerThread per market with active models. Failures isolated per market."""
        ...
    def stop_producers(self) -> None: ...
    def get_consumer(self) -> ModelConsumer:
        return ModelConsumer(self._config)
```

### 9.5 Usage

```python
runner = AlphaAggregatorRunner.from_config()
runner.start_producers()    # background threads, one per market

# Elsewhere, on-demand:
consumer = runner.get_consumer()
composite = consumer.get_composite_score(model_id="hk_momentum")

runner.stop_producers()
```

---

## 10. post_trade_analyzer

**Purpose:** Compares **individual alpha scores** (predicted) against **realized returns** (from the model's `rtn_alpha_id`), computes the difference, and explains it via performance attribution (PAA) against Barra ASE2S factors. No break detection, no slippage (dropped — no execution data available). Analysis combines all alphas in a model into **one report**.

### 10.1 Core Calculation

```
difference(RIC, t) = realized_return(RIC, t) - alpha_score(RIC, t)
```

Explained via pooled cross-sectional regression:

```
difference ~ barra_factor_1 + barra_factor_2 + ... + barra_industry_1 + ...
```

Barra industry factors serve as the sector-level breakdown — no separate sector lookup needed.

### 10.2 Key Classes

```python
class AlphaScoreLoader:
    def load(self, market: str, alpha_id: str, dates: list[date]) -> pd.DataFrame: ...

class RealizedReturnLoader:
    def load(self, market: str, rtn_alpha_id: str, dates: list[date]) -> pd.DataFrame: ...

class FactorAttribution:
    def compute_difference(self, alpha_scores, realized_returns) -> pd.DataFrame: ...
    def run_regression(self, difference_df, market, dates) -> dict[str, float]:
        """Pools all (RIC, date). Returns {factor}_coefficient per Barra factor + r_squared, n_obs, intercept."""
        ...

class ReportGenerator:
    def assemble(self, per_alpha_results, per_alpha_summary) -> pd.DataFrame:
        """
        alpha_id           | mean_diff | std_diff | r_squared | n_obs | barra_mkt | ... | intercept
        momentum_price_21d | 0.002     | 0.015    | 0.18      | 12400 | 0.05      | ... | 0.001
        earnings_revision  | -0.001    | 0.012    | 0.22      | 12400 | 0.03      | ... | 0.000
        """
        ...
    def save(self, report_df, model_id, date, overwrite=False) -> None:
        """Saves to post_trade_results/{model_id}/YYYYMMDD.parquet."""
        ...


class PostTradeAnalyzerRunner:
    def run(
        self, model_id: str | list[str], dates: list[str], overwrite: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Per model_id:
          1. Load ModelConfig (alpha_ids, rtn_alpha_id, market)
          2. Load realized returns from rtn_alpha_id
          3. For each alpha: load predicted scores, compute difference, run regression
          4. Assemble combined report (one row per alpha)
          5. Save to post_trade_results/{model_id}/YYYYMMDD.parquet
        """
        ...
```

### 10.3 Usage

```python
runner = PostTradeAnalyzerRunner.from_config()
result = runner.run(model_id="hk_momentum", dates=["20240101", ..., "20240131"])
# result["hk_momentum"]: one row per alpha in the model
```

---

## 11. Global Design Principles

1. **PIT correctness is non-negotiable in research mode.** All data access goes through `PITDataManager`. Production mode (`as_of_date=now()`) skips filtering for latency; research mode filters strictly.

2. **RIC is the universal identifier.** `BBL` exists only inside `BBGClient`, mapped via `desktool`, never exposed elsewhere.

3. **Alphas are self-describing.** A TOML config + a Python subclass (of the correct type: `HighAlpha`/`MidAlpha`/`LowAlpha`/`RtnAlpha`) fully defines an alpha. Adding a new alpha = adding these two files + a registry entry.

4. **Normalization is per-alpha, not per-type.** `BaseAlpha.normalize()` defaults to identity; concrete subclasses override with their own `SHIFT`/`SCALE` constants as needed.

5. **Universe is implicit.** Determined by which RICs have complete data across all `required_inputs` — no explicit inclusion/exclusion list beyond the market-level universe CSV.

6. **Return is an alpha type.** `RtnAlpha` computes forward returns and is used both as a standalone predictive signal and as the "ground truth" return source for backtesting and post-trade analysis. Only `RtnAlpha` carries entry/exit timing.

7. **Models are the unit of combination.** A model = market + set of alphas + a shared `rtn_alpha_id` (defining entry/exit) + optimized weights. Individual alphas are computed once; models combine them differently.

8. **Production and research share code.** `signal_builder.build_one_day_alpha()` is called identically by research notebooks and by `alpha_aggregator`'s Producer — the only difference is `save=True/False`.

9. **Producer/Consumer separation avoids redundant computation.** The Producer computes each alpha once per market regardless of how many models use it; the Consumer combines on demand, never persisting composite scores to kdb+.

10. **No CLI.** All packages expose a `{Package}Runner` class called directly from notebooks or from other packages. Logging (`setup_logger`) is imported from an external shared package, not built here.

11. **Config drives behavior.** Optimization parameters, scheduling, storage targets, alpha selection — all configurable via TOML without code changes.

12. **Shared drive is the integration contract.** Parquet/CSV files under `{shared_drive_path}/` are how packages hand off data to each other. No package reads another's in-memory state directly except via well-defined Runner method calls.

---

## Appendix: Open Items for Future Refinement

The following were explicitly deferred during design and should be revisited once implementation begins:

- **HighAlphaConfig / MidAlphaConfig / LowAlphaConfig** field-level specifics (beyond `BaseAlphaConfig`) — to be defined once each type's first concrete alpha is implemented
- **Bloomberg field list** for the `bbg` data_type — to be implemented manually via `desktool`
- **`BBGExtractor`** full implementation — currently a placeholder raising `NotImplementedError`
- **kdb+ table names / schemas** for each raw data type — abstracted behind `KDBClient`, to be filled in against the actual kdb+ schema
- **S3 key/path conventions** for `intropic` and `southbound` — abstracted behind `S3Client`
- **External alert function** (email + beep) — integration point only, actual implementation from another shared package
- **`setup_logger`** — to be imported from an existing external package, not built in this project
