# AlphaFlow

Alpha research, calculation and post-trade reconciliation for APAC equities
(HK, CN, SG, JP, KR, AU, TW).

`RIC` is the universal identifier. `BBL` exists only inside `BBGClient`, mapped via
`apcr_desktool`, and never reaches a caller.

```
data_dumper → signal_builder → strategy_backtester → optimizer
                     ↓                                    ↓
              alpha_aggregator (production) ←—————— model weights
                     ↓
           post_trade_analyzer
```

## Install

```bash
uv sync
uv run pytest
```

`apcr_desktool` is the on-prem integration package (kdb+, Bloomberg, BBL↔RIC symbology, auth).
It is not on PyPI, so it is deliberately **not** a declared dependency — that would make the
lock file unresolvable off-prem. On-prem:

```bash
uv pip install -e ../apcr_desktool
export ALPHAFLOW_DESKTOOL=real           # real | stub | auto (default)
export ALPHAFLOW_SHARED_DRIVE=/mnt/shared/alphaflow
```

Off-prem the adapter falls back to `StubDeskTool`, whose symbology works and whose **data
calls raise**. That is deliberate: a forgotten mock fails loudly instead of inventing numbers.

## Packages

Each exposes exactly one `Runner` with `from_config()`. No CLI.

| Package | Runner | Role |
|---|---|---|
| `data_dumper` | `DataDumperRunner` | Manual backfill of raw data → `raw_data/*.csv` |
| `signal_builder` | `SignalBuilderRunner` | Computes alpha scores → `alpha_scores/*.parquet` |
| `strategy_backtester` | `StrategyBacktesterRunner` | Single-alpha IC / regression / decile / Barra |
| `optimizer` | `OptimizerRunner` | Mean-variance weights across a model's alphas |
| `alpha_aggregator` | `AlphaAggregatorRunner` | Production producer threads + on-demand consumer |
| `post_trade_analyzer` | `PostTradeAnalyzerRunner` | Predicted vs realised, attributed to Barra factors |

```python
from alphaflow import SignalBuilderRunner
SignalBuilderRunner.from_config().build_one_day_alpha("momentum_price_21d", "HK", "20240115")
```

`from_config()` paths resolve relative to the **project root**, not the CWD, so notebooks in
any subdirectory work with no arguments.

## Adding an alpha

Three files, no code changes anywhere else:

1. `configs/alphas/<type>/<alpha_id>.toml` — copy `_template.toml`
2. `src/alphaflow/signal_builder/alphas/<type>/<alpha_id>.py` — copy `_template.py`, implement `compute()`
3. a `[[alphas]]` entry in `configs/registry.toml`

`AlphaRegistry` runs all eight cross-config checks on construction, so a mismatch between
these three fails at startup rather than mid-run.

Alpha types: `HighAlpha` (tick intraday), `MidAlpha` (fixed interval), `LowAlpha` (daily),
`RtnAlpha` (forward return — the only type carrying entry/exit timing, and the ground truth
for the backtester and post-trade analyzer).

## Configuration

| File | Scope |
|---|---|
| `configs/system.toml` | Storage, connections, per-market sessions and data types |
| `configs/production.toml` | Which models run live, threading, scheduling |
| `configs/optimizer.toml` | Mean-variance parameters (all models) |
| `configs/registry.toml` | Master alpha list |
| `configs/models/<model_id>.toml` | Market + alphas + rtn_alpha_id + dates + weights |
| `configs/alphas/<type>/*.toml` | Per-alpha definitions |
| `configs/universe/universe_<MKT>.csv` | RIC universe per market |

All validated with Pydantic v2; `ConfigValidationError` on anything invalid.

Two TOML layout notes, both departures from the spec's examples, which are not valid TOML as
printed: a bare key written after a `[table]` header belongs to that table, so `risk_model`
sits **above** `[mean_variance]` in `optimizer.toml`, and an rtn alpha's `horizon` /
`return_type` live in an explicit `[timing]` section rather than trailing `[exit]`.

## Shared drive

The integration contract between packages (spec §12) — nothing reads another package's
in-memory state.

```
{shared_drive_path}/
├── raw_data/<MARKET>/<data_type>/YYYYMMDD.csv
├── alpha_scores/<MARKET>/<alpha_id>/YYYYMMDD.parquet
├── alpha_scores/<model_id>/YYYYMMDD.parquet
├── backtest_results/<MARKET>/<alpha_id>_YYYYMMDD-YYYYMMDD.parquet
├── optimizer_results/<MARKET>/<model_id>.parquet
└── post_trade_results/<model_id>/YYYYMMDD.parquet
```

## Point-in-time correctness

Every read goes through `PITDataManager`.

- **Research** (`for_research(as_of_date)`) — any row stamped after `as_of_date` raises
  `PITViolationError`. The cut is re-applied locally on top of whatever the source did.
- **Production** (`for_production()`) — no filtering; `now` is the as-of date by definition.

One deliberate exception: an `RtnAlpha` with `horizon > 0` gets its PIT cut extended past the
horizon. A forward return signalled on `t` is only observable once `t+horizon` has traded —
it is the label, not a predictor, and the backtester lines it up against scores from `t`.

## Development

```bash
uv run pytest            # 388 tests
uv run ruff check src tests
```

Tests are mock-based (`pytest-mock`); no kdb+, Bloomberg or S3 connection is needed or made.
An autouse fixture redirects the shared drive to a temp directory, so no test can write into
`data/`.

See `PLAN.md` for the design decisions behind this implementation and what remains open.
