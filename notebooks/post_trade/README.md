# Post-Trade Analysis

Where the prediction and the outcome diverged, and which risk factors explain it.

```
difference(RIC, t) = realized_return(RIC, t) - alpha_score(RIC, t)
```

pooled across every `(RIC, date)` and regressed on the Barra ASE2S factors. Industry factors
are the sector breakdown — there is no separate sector mapping.

```python
from alphaflow import PostTradeAnalyzerRunner
report = PostTradeAnalyzerRunner.from_config().run(model_id="hk_momentum", dates=dates)
```

One report per model, one row per alpha, saved to `post_trade_results/{model_id}/YYYYMMDD.parquet`.

Factor columns are whatever the risk model returns — nothing here hardcodes a factor list.
If the universe is too small relative to the factor count the regression is rank-deficient,
and the analyzer reports NaN rather than coefficients that are not identified.
