# AlphaFlow — Implementation Plan

> **Reference spec:** `AlphaFlow_Master_Technical_Specification update.md` (v1.0, 1708 lines)
> **Created:** 2026-09-06
> **Status:** Awaiting approval — no code written yet

---

## 1. Environment Findings

| Item | Finding |
|---|---|
| Working dir | `/mnt/c/Users/朱念湘/workspace/alphaflow` — contains only the spec |
| Git | Not a repo yet |
| Python | 3.12.3 local; `uv` 0.11.2; PyPI reachable |
| `../apcr_desktool` | **Absent** — spec pins it as an editable local dependency |
| kdb+ / Bloomberg / S3 | Not reachable from this machine |
| `alphaflow_old/` | Earlier, superseded design (empty scaffolding) — **not reused** |
| Conventions | Root `CLAUDE.md` + `python-CLAUDE.md` apply |

---

## 2. Design Decisions (confirmed with user)

| # | Open question | Decision |
|---|---|---|
| **D1** | `apcr_desktool` missing | Thin adapter boundary `core/integrations/` — a `Protocol` for `ric_to_bbl` / `bbl_to_ric` / auth plus a local `setup_logger` fallback, backed by a working stub. Real package swaps in via one config flag. |
| **D2** | Runnability without live data sources | Real client code per spec. **Unit tests use `pytest-mock` only.** No synthetic-data generator, no fake backends. Code will not execute end-to-end until connected on-prem. |
| **D3** | Spec style vs `python-CLAUDE.md` | Keep spec's type hints on all public signatures + Pydantic v2 models. Adopt `python-CLAUDE.md` formatting: **line-length 149**, compact multi-arg lines, f-strings, minimal docstrings, inline comments. |
| **D4** | Build scope | All 7 packages, phased in dependency order. |
| **D5** | kdb+ tables / S3 keys (spec-deferred) | Config-driven: `[kdb.tables]` and `[s3.keys]` maps in `system.toml`, data_type → table name / key template, with documented placeholders. Filling in real values is a TOML edit. |
| **D13** | kdb access mechanism (**user directive, 2026-09-06**) | **`pykx` is removed entirely.** All kdb+ data is fetched through the `apcr_desktool` API. `KDBClient` becomes a thin wrapper over the desktool adapter (D1) rather than a pykx wrapper. Bloomberg likewise routes through the adapter. Only S3 keeps a direct third-party client (`boto3`), per spec. |
| **D6** | Concrete alphas | `momentum_price_21d` (Low), `open_to_close_1d` (Rtn), `order_imbalance_1430` (High) + `_template.toml` and a template `.py` for all four types. |
| **D7** | Barra ASE2S factor list (spec-deferred) | **Runtime discovery** — every non-key column of the risk-factor frame is a factor; result tables emit one `barra_<factor>` entry per discovered factor. No hardcoded list. |
| **D8** | Bootstrap | `git init` in `alphaflow/`; `.python-version` = **3.12**. |
| **D9** | Intraday freshness gap (§6.1 CSV vs §9.1 "fresh every iteration") | Add `source="raw_data" \| "live"` to `SignalBuilderRunner`. Research reads CSV dumps; the Producer passes `source="live"` to pull straight from `PITDataManager`. Identical `compute()`/`normalize()` both ways — preserves principle #8. |
| **D10** | Path resolution | `find_project_root()` (walks up to `pyproject.toml`) anchors all `configs/...` defaults so notebooks work from any subdir. `shared_drive_path` overridable by `$ALPHAFLOW_SHARED_DRIVE`; committed default `<repo>/data/` (gitignored). |
| **D11** | `HighAlphaConfig`/`MidAlphaConfig`/`LowAlphaConfig` (spec-deferred) | Minimal fields driven by the real alphas: High → `snapshot_time`, `depth_levels`; Low → `lookback_days`, `skip_days`; Mid → `interval_minutes`. Documented as extensible. |
| **D12** | `notebooks/` | README stubs only, documenting Runner call patterns per workflow. |

### Assumptions I am making (not separately asked — flag if any is wrong)

- **A1** `ConfigValidationError` (referenced but never located in the spec) lives in `core/config/exceptions.py`.
- **A2** Data type `intropic` is kept spelled exactly as the spec writes it.
- **A3** APScheduler drives the **daily refresh trigger** inside `MarketProducerThread`; the intraday loop is a plain thread loop on `intraday_update_interval_sec`.
- **A4** ~~pykx~~ — superseded by **D13**: not a dependency at all. Because `apcr_desktool` is absent (D1), the adapter `Protocol` must also cover the kdb query surface; its method signatures are my best-guess narrow contract and **will need one editing pass against the real desktool API**. This is the single place that has to change.
- **A5** `[tool.ruff] line-length = 149` (overrides the spec's 100, per D3).
- **A6** Dates are `datetime.date` internally; `YYYYMMDD` strings only at API and filename boundaries. A shared `core/utils/dates.py` handles conversion.
- **A7** Every `{Package}Runner` gets `from_config()` with project-root-anchored defaults, per §2.5.

---

## 3. Phase Breakdown

```
Phase 0  Bootstrap                 ← no dependencies
Phase 1  Config Layer              ← Phase 0
Phase 2  Core / Integrations       ← Phase 1
Phase 3  Core / Data Access        ← Phase 2
Phase 4  Core / Alpha Base         ← Phase 3
Phase 5  data_dumper               ← Phase 3
Phase 6  signal_builder            ← Phase 4, 5
Phase 7  strategy_backtester       ← Phase 6
Phase 8  optimizer                 ← Phase 7
Phase 9  alpha_aggregator          ← Phase 6
Phase 10 post_trade_analyzer       ← Phase 6
Phase 11 Integration & polish      ← all
```

---

### Phase 0 — Bootstrap

**Files**
- `.git/` (init), `.gitignore` (`data/`, `.venv/`, `__pycache__/`, `*.parquet`, `.ipynb_checkpoints/`)
- `.python-version` → `3.12`
- `pyproject.toml` — per §2.2, minus `apcr-desktool` from `[tool.uv.sources]` (D1), **minus `pykx`** (D13); `ruff line-length = 149` (A5)
- `README.md` — purpose, install, package map, "how to add an alpha"
- `context/` — spec copy + this session's decision record (root `CLAUDE.md`: *always save down the context / input output locally*)
- Full directory tree from §2.1 with every `__init__.py`

**Definition of done**
- `uv sync` completes; `uv run python -c "import alphaflow"` succeeds
- `uv run pytest` collects 0 tests without error

---

### Phase 1 — Config Layer

**Files** — `src/alphaflow/core/config/`
- `exceptions.py` → `ConfigValidationError` (A1)
- `paths.py` → `find_project_root()`, `resolve_config_path()`, `resolve_shared_drive()` (D10)
- `system_config.py` → `SessionConfig`, `MarketConfig`, `StorageConfig`, `KdbConfig` (+ `tables` map, D5), `BloombergConfig`, `S3Config` (+ `keys` map, D5), `SchedulerConfig`, `SystemConfig`
- `production_config.py` → `ProductionModelConfig`, `ProductionSchedulerConfig` (+ `HH:MM` validator), `ThreadingConfig`, `ProductionOutputConfig`, `ProductionConfig`
- `optimizer_config.py` → `MeanVarianceConfig`, `OptimizerConfig`
- `registry_config.py` → `AlphaType`, `RegistryEntry`, `RegistryConfig` (+ `get_active` / `get_by_id` / `get_by_type`)
- `model_config.py` → `ModelConfig` (+ filename-match and weight-key validators)
- `alpha_config/` → `base_config.py`, `high_config.py`, `mid_config.py`, `low_config.py` (D11), `rtn_config.py` (`PricePoint` + `RtnAlphaConfig`, fully spec'd §3.7)
- `loader.py` → `tomllib` readers returning validated models, raising `ConfigValidationError`

**Config files** — `configs/system.toml`, `production.toml`, `optimizer.toml`, `registry.toml`, `models/hk_momentum.toml`, `alphas/{high,mid,low,rtn}/_template.toml`, the 3 real alpha TOMLs, `universe/universe_{HK,CN,SG,JP,KR,AU,TW}.csv` (small representative RIC lists)

**Tests** — `tests/unit/core/config/` — valid parse per model, missing-field → `ConfigValidationError`, bad time format, weight-key mismatch, `model_id` vs filename mismatch, `PricePoint` vwap/session validators.

**DoD** — every shipped TOML loads clean; each validator has a passing negative test.

---

### Phase 2 — Core / Integrations (D1)

**Files** — `src/alphaflow/core/integrations/`
- `desktool_adapter.py` — `DeskToolProtocol` (`ric_to_bbl`, `bbl_to_ric`, `authenticate`, `query_kdb`, `query_kdb_latest`, `bdh`, `bdp`), `StubDeskTool`, `get_desktool()` resolving real vs stub from config/env. **Sole integration seam for kdb+ and Bloomberg** (D13).
- `logging_setup.py` — `setup_logger(name, level)`; delegates to the external package when importable, else a local rotating-file + console fallback writing to `StorageConfig.log_path`

**Tests** — adapter selection logic; stub round-trips RIC↔BBL; logger returns a configured `Logger` and does not duplicate handlers.

---

### Phase 3 — Core / Data Access

**Files** — `src/alphaflow/core/data_access/`
- `exceptions.py` → `PITViolationError`, `DataNotFoundError`, `DataSourceError`
- `kdb_client.py` → `KDBClient.query()` / `query_latest()`; **all reads via the desktool adapter, no pykx** (D13); table names from `[kdb.tables]` (D5); all output RIC-keyed
- `bloomberg_client.py` → `BBGClient.bdh()` / `bdp()` via the desktool adapter; **BBL confined inside the class** (§4.1) — converts RIC→BBL on entry, BBL→RIC on exit
- `s3_client.py` → `S3Client.read_csv()` / `read_parquet()`; key templates from `[s3.keys]`
- `pit_manager.py` → `PITMode`, `PITDataManager` with `for_research()` / `for_production()` and all 9 `get_*` methods from §4.2

**Tests** — `tests/unit/core/test_pit_manager.py` and per-client tests, all `pytest-mock` (D2): research mode filters rows > `as_of_date` and raises `PITViolationError`; production mode skips filtering; every output frame is RIC-keyed; no BBL column ever escapes `BBGClient`; empty result → `DataNotFoundError`.

---

### Phase 4 — Core / Alpha Base

**Files** — `src/alphaflow/core/alpha_base/`
- `market.py` → `Market(StrEnum)`
- `base_alpha.py` → `BaseAlpha(ABC)` — `compute()` (abstract), `normalize()` (identity default), `run()`, `get_universe()`, `validate_config()`, `_working_data` slot
- `high_alpha.py` / `mid_alpha.py` / `low_alpha.py` — typed bases
- `rtn_alpha.py` → `RtnAlpha` with `market_config` and `_validate_intraday_session()` for `horizon == 0`
- `registry.py` → `AlphaRegistry` + `from_paths()`, running **all 8 cross-config checks of §3.9** on construction
- `loader.py` → `AlphaLoader`, `TypedAlphaLoader(ABC)`, four typed loaders (config load + `handler_class` import + parent-class assertion)

**Tests** — one file per alpha type (as in §2.1 tree) + registry and loader: identity normalize, `run()` ordering, universe CSV load, each §3.9 check fails loudly, loader rejects a handler of the wrong parent type, session validation for `horizon=0`.

---

### Phase 5 — data_dumper

**Files** — `src/alphaflow/data_dumper/`
- `extractors/base_extractor.py`, `kdb_extractor.py` (6 kdb types, internal dispatch), `s3_extractor.py` (`intropic`, `southbound`), `bbg_extractor.py` (`NotImplementedError` placeholder per spec)
- `writers/csv_writer.py` → `raw_data/{market}/{data_type}/YYYYMMDD.csv`, RIC primary key, `depth_intraday` single file with `timestamp` column
- `runner.py` → `DataDumperRunner` with `run(market, date|start/end, source, overwrite)`; `source=None` → all `available_data_types`; universe from `universe_{market}.csv`

**Tests** — extractor dispatch, `supported_data_types`, writer path/skip-on-exists, runner date-range expansion, BBG placeholder raises.

---

### Phase 6 — signal_builder

**Files** — `src/alphaflow/signal_builder/`
- `runner.py` → `SignalBuilderRunner.build_one_day_alpha()` (the core method: load inputs → inner-join on RIC → drop nulls → inject `_working_data` → `run()` → `DataFrame(RIC, raw_signal, score)` → optional parquet) and `build_multi_day_alpha()` (`ProcessPoolExecutor` when `n_jobs > 1`). Both carry `source="raw_data"|"live"` (**D9**)
- `storage/score_writer.py` → `ScoreWriter` write / `file_exists` / `read` / `read_range`
- `alphas/low/momentum_price_21d.py`, `alphas/rtn/open_to_close_1d.py`, `alphas/high/order_imbalance_1430.py`
- `alphas/{high,mid,low,rtn}/_template.py` — documented copy-paste starting points

**Tests** — join/null-drop produces the implicit universe; `_working_data` injected before `compute()`; `save=False` writes nothing; `overwrite=False` skips existing; multi-day shape includes `date`; per-alpha compute and normalize correctness on hand-built frames.

**DoD** — adding an alpha = 1 TOML + 1 `.py` + 1 registry entry, demonstrated by the three real alphas.

---

### Phase 7 — strategy_backtester

**Files** — `src/alphaflow/strategy_backtester/`
- `score_loader.py` → `ScoreLoader` (shared with optimizer)
- `metrics/ic_calculator.py` (daily Spearman IC → `mean_IC`, `std_IC`, `ICIR`), `performance_metrics.py` (Sharpe from IC series, max drawdown of cumulative IC, rank-change turnover), `decile_analyzer.py` (D10−D1 spread)
- `regression/cross_sectional_ols.py` (pooled OLS over all dates+stocks → `r_squared`, `t_stat`, `coefficient`, `intercept`, `n_stocks`), `regression_report.py`
- `risk/barra_exposure.py` — score-weighted mean factor exposure, period-averaged, factors discovered at runtime (**D7**)
- `engine/backtest_engine.py` — orchestrates one alpha × one `rtn_alpha_id`
- `reporting/report_generator.py` — assemble the §7.3 **measures × rtn_alpha_ids** table; save to `backtest_results/{market}/{alpha_id}_YYYYMMDD-YYYYMMDD.parquet`; **always saved**; `overwrite=False` → `FileExistsError`
- `runner.py` → `StrategyBacktesterRunner.run(alpha_id, market, dates, rtn_alpha_id, overwrite, n_jobs)`

**Tests** — IC against a known-correlation fixture, ICIR arithmetic, drawdown on a crafted series, decile spread, OLS coefficients vs `statsmodels` ground truth, table orientation (rows = measures), `FileExistsError` path.

---

### Phase 8 — optimizer

**Files** — `src/alphaflow/optimizer/`
- `returns/strategy_return_builder.py` → `daily_return(t) = Σ score_i(t) · return_i(t)`; `build_for_model()` → date × alpha_id frame
- `analysis/correlation.py`, `analysis/standalone_metrics.py`
- `optimizers/mean_variance_optimizer.py` → SLSQP maximize `wᵀμ − λ·wᵀΣw`, bounds `[min,max]`, `Σw = 1`
- `risk/barra_decomposer.py` → standalone + weight-combined exposures
- `storage/model_config_writer.py` → rewrites **only** `[alpha_weights]` in `configs/models/{model_id}.toml`, preserving comments and every other key
- `storage/result_writer.py` → wide table, one row per alpha + a `PORTFOLIO` row → `optimizer_results/{market}/{model_id}.parquet`
- `runner.py` → `OptimizerRunner.run()`, the 7 steps of §8.6

**Tests** — return-series arithmetic on matched/unmatched RICs, weights sum to 1 and respect bounds, a dominant-alpha case converges as expected, config writer round-trip leaves all other TOML content byte-identical, `PORTFOLIO` row present.

---

### Phase 9 — alpha_aggregator

**Files** — `src/alphaflow/alpha_aggregator/`
- `producer/daily_cache.py` → `DailyAlphaCache` (`is_fresh` / `get` / `set` / `force_refresh`)
- `producer/market_producer.py` → `resolve_market_alpha_ids()` (union across active models, computed **once** at startup) and `MarketProducerThread`: daily alphas once per day (APScheduler trigger, A3), intraday every `intraday_update_interval_sec` via `source="live"` (D9), writes **individual scores only** to kdb+, retry → `max_restart_attempts` → CRITICAL + `send_critical_alert()` → stop **this thread only**
- `consumer/model_consumer.py` → `ModelConsumer.get_composite_score()`: on-demand, RIC union, missing → 0, `Σ wᵢ·scoreᵢ` with **no renormalization**, optional parquet archive
- `storage/kdb_writer.py` → schema `timestamp, market, alpha_id, RIC, raw_signal, score, frequency`; `query_latest_scores()`
- `storage/parquet_writer.py` → `alpha_scores/{model_id}/YYYYMMDD.parquet`, individual + composite
- `alerting/alert.py` → `send_critical_alert()` integration point
- `runner.py` → `AlphaAggregatorRunner.start_producers()` / `stop_producers()` / `get_consumer()`

**Tests** — alpha-id union dedupes across models; cache staleness across a date boundary; failure in one market's thread leaves others running and fires exactly one alert; composite maths incl. missing-RIC zero-fill and no renormalization; **composite never written to kdb+**.

---

### Phase 10 — post_trade_analyzer

**Files** — `src/alphaflow/post_trade_analyzer/`
- `loaders/alpha_score_loader.py`, `loaders/realized_return_loader.py`
- `attribution/factor_attribution.py` → `difference = realized_return − alpha_score`; pooled cross-sectional OLS of difference on all Barra style + industry factors (D7) → per-factor coefficient, `r_squared`, `n_obs`, `intercept`
- `reporting/report_generator.py` → one row per alpha → `post_trade_results/{model_id}/YYYYMMDD.parquet`
- `runner.py` → `PostTradeAnalyzerRunner.run(model_id, dates, overwrite)`

**Tests** — difference alignment on partially overlapping RIC sets, regression recovers planted factor loadings, one report row per alpha in the model, industry factors present as the sector breakdown.

---

### Phase 11 — Integration & Polish

- `tests/conftest.py` — shared fixtures: tmp shared drive, sample configs, mocked `PITDataManager`, synthetic score frames
- `tests/integration/` — the six pipeline tests from §2.1, wiring runners together with mocked data sources
- `notebooks/README.md` + per-directory READMEs (D12)
- Each package `__init__.py` re-exports its Runner (§2.4)
- Final `uv run pytest` green; `uv run ruff check` clean
- `README.md` finalized; `context/` decision record saved

---

## 4. Progress Tracker

| Phase | Status | Tests |
|---|---|---|
| 0 — Bootstrap | ☑ Done | — |
| 1 — Config Layer | ☑ Done | 45 |
| 2 — Core / Integrations | ☑ Done | 17 |
| 3 — Core / Data Access | ☑ Done | 18 |
| 4 — Core / Alpha Base | ☑ Done | 72 |
| 5 — data_dumper | ☑ Done | 26 |
| 6 — signal_builder | ☑ Done | 35 |
| 7 — strategy_backtester | ☑ Done | 47 |
| 8 — optimizer | ☑ Done | 43 |
| 9 — alpha_aggregator | ☑ Done | 34 |
| 10 — post_trade_analyzer | ☑ Done | 16 |
| 11 — Integration & Polish | ☑ Done | 35 |
| **Total** | **Complete** | **388 passing, ruff clean** |

### Deviations found during implementation

Four things the spec could not have anticipated, each resolved and covered by a test:

1. **The spec's TOML examples are not valid TOML.** A bare key written after a `[table]`
   header is absorbed into that table, so `optimizer.toml`'s `risk_model` and an rtn alpha's
   `horizon`/`return_type` would have landed in the wrong place. `risk_model` now sits above
   `[mean_variance]`; rtn timing moved to an explicit `[timing]` section.
2. **A degenerate IC series produced an absurd ICIR.** A constant series has
   `std ≈ 3e-17`, not `0.0`, so `mean/std` returned ~6e15 — which reads as a spectacular
   alpha. `core/utils/stats.py` now guards every std-ratio in the codebase.
3. **`feasible_for()` rejected single-alpha models.** With the default 0.8 cap, one alpha
   cannot sum to 1. The cap is a diversification constraint that only bites across several
   alphas, so `n == 1` short-circuits to weight 1.0.
4. **Post-trade attribution fitted rank-deficient designs.** More Barra factors than stocks
   (very plausible with industry factors on a small universe) gave unidentified coefficients
   with no error. It now requires 5 observations per factor plus a matrix-rank check, and
   reports NaN rather than noise.

---

## 5. Known Deferrals Carried Forward

Inherited from the spec's own appendix, and unblocked by config rather than code:

- Real kdb+ table names and column schemas → fill `[kdb.tables]` in `system.toml`
- Real S3 key conventions for `intropic` / `southbound` → fill `[s3.keys]`
- Bloomberg field list and `BBGExtractor` implementation → stays `NotImplementedError`
- External alert function (email + beep) → `alerting/alert.py` is the integration point
- `MidAlpha` has no concrete implementation (spec provides no example or field spec)
- Real `apcr_desktool` → drop into `../apcr_desktool` and flip the adapter flag; **reconcile `DeskToolProtocol` method signatures with the real API** (D13)
