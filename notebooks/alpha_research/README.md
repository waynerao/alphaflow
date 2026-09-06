# Alpha Research

Building a new alpha and judging whether it works.

**Adding an alpha is three files:**
1. `configs/alphas/<type>/<alpha_id>.toml` — copy `_template.toml` from the same directory
2. `src/alphaflow/signal_builder/alphas/<type>/<alpha_id>.py` — copy `_template.py`, implement `compute()`
3. a `[[alphas]]` entry in `configs/registry.toml`

`AlphaRegistry` validates all three against each other on construction, so a mismatch fails
at startup rather than mid-run.

```python
from alphaflow import SignalBuilderRunner, StrategyBacktesterRunner

builder = SignalBuilderRunner.from_config()
scores = builder.build_one_day_alpha("my_alpha", "HK", "20240115")     # inspect before committing to a range
builder.build_multi_day_alpha("my_alpha", "HK", dates, save=True, n_jobs=4)

StrategyBacktesterRunner.from_config().run("my_alpha", "HK", dates, rtn_alpha_id="open_to_close_1d")
```

The backtest result is a table of **measures as rows, `rtn_alpha_id`s as columns** — pass a
list of return definitions to compare entry/exit conventions side by side. It is always saved
to `backtest_results/{market}/`; re-running the same window needs `overwrite=True`.
