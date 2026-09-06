# Portfolio Optimization

Turning a set of alphas into a weighted model.

Each alpha is treated as a standalone strategy with its own daily return series
(`sum(score * forward_return)` per day). Correlation enters through the covariance matrix —
correlated alphas are down-weighted, never excluded by a threshold.

```python
from alphaflow import OptimizerRunner
result = OptimizerRunner.from_config().run(model_id="hk_momentum")
```

Two side effects, both intentional:
- `configs/models/hk_momentum.toml` is updated **in place** — only the `[alpha_weights]`
  section changes; comments and every other key survive byte-for-byte.
- an audit table lands in `optimizer_results/{market}/{model_id}.parquet`: one row per alpha
  plus a `PORTFOLIO` row, with weight, standalone Sharpe/std, average correlation, and Barra
  exposures.

Parameters live in `configs/optimizer.toml` and apply to **all** models. Which models to
optimize is an argument to `run()`, never configuration.
