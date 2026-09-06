# Notebooks

Research entry points. Every package exposes exactly one `Runner` class called directly from
here — there is no CLI.

All `from_config()` defaults resolve relative to the **project root**, not the working
directory, so a notebook in any subdirectory can call them with no arguments.

## Typical sequence

```python
# 1. Backfill raw data (manual, not scheduled)
from alphaflow import DataDumperRunner
DataDumperRunner.from_config().run(market="HK", start="20230101", end="20231231")

# 2. Compute alpha scores to the shared drive
from alphaflow import SignalBuilderRunner
builder = SignalBuilderRunner.from_config()
builder.build_multi_day_alpha("all", market="HK", dates=dates, save=True, n_jobs=4)

# 3. Evaluate one alpha against one or more return definitions
from alphaflow import StrategyBacktesterRunner
StrategyBacktesterRunner.from_config().run(
    alpha_id="momentum_price_21d", market="HK", dates=dates,
    rtn_alpha_id=["open_to_close_1d", "close_to_1d_vwap"])

# 4. Allocate weights across the alphas in a model
from alphaflow import OptimizerRunner
OptimizerRunner.from_config().run(model_id="hk_momentum")   # writes weights back into the model TOML

# 5. Reconcile predictions against what actually happened
from alphaflow import PostTradeAnalyzerRunner
PostTradeAnalyzerRunner.from_config().run(model_id="hk_momentum", dates=dates)
```

## Directories

| Directory | Purpose |
|---|---|
| `alpha_research/` | Building and evaluating a single alpha: `SignalBuilderRunner` → `StrategyBacktesterRunner` |
| `portfolio_optimization/` | Combining alphas into a model: `OptimizerRunner` |
| `post_trade/` | Predicted vs realised, attributed to Barra factors: `PostTradeAnalyzerRunner` |

## Before anything runs

These runners reach real data sources. Off-prem they will fail at the first read, by design —
see `src/alphaflow/core/integrations/desktool_adapter.py`. On-prem:

```bash
uv pip install -e ../apcr_desktool
export ALPHAFLOW_DESKTOOL=real
export ALPHAFLOW_SHARED_DRIVE=/mnt/shared/alphaflow
```
