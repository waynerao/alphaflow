# AlphaFlow — Session Context: Design Decisions
**Date:** 2026-09-06
**Input spec:** `AlphaFlow_Master_Technical_Specification update.md` (v1.0)
**Output:** `PLAN.md`

## Environment as found
- `/mnt/c/Users/朱念湘/workspace/alphaflow` contained only the spec file
- Not a git repo; Python 3.12.3; uv 0.11.2; PyPI reachable
- `../apcr_desktool` absent
- No kdb+, Bloomberg, or S3 connectivity
- `alphaflow_old/` = earlier superseded design, not reused

## Questions asked and answers given

**Round 1 — scope & environment**
1. `apcr_desktool` missing → **Adapter + local stub** in `core/integrations/`
2. Runnability without live data → **Interfaces + mocked unit tests only** (no synthetic data generator, no fake backends)
3. Spec style vs `python-CLAUDE.md` → **Spec signatures (type hints + Pydantic) + user formatting (line-length 149, compact args, inline comments)**
4. Build scope → **All 7 packages, phased in one plan**

**Round 2 — deferred technical details**
5. kdb+ tables / S3 keys → **Config-driven placeholders** (`[kdb.tables]`, `[s3.keys]` in system.toml)
6. Concrete alphas → **momentum_price_21d (low), open_to_close_1d (rtn), order_imbalance_1430 (high) + templates for all 4 types**
7. Barra ASE2S factor list → **Runtime discovery from risk-factor frame columns**
8. Bootstrap → **git init + pin Python 3.12**

**Round 3 — design gaps found while reading**
9. §6.1 CSV vs §9.1 "fresh every iteration" contradiction → **Add `source="raw_data"|"live"` to SignalBuilderRunner**
10. Path resolution → **`find_project_root()` + `$ALPHAFLOW_SHARED_DRIVE` override, default `<repo>/data/`**
11. Empty High/Mid/LowAlphaConfig → **Minimal fields driven by the real alphas**
12. `notebooks/` contents → **README stubs only**

## Assumptions stated without asking
- A1 `ConfigValidationError` → `core/config/exceptions.py`
- A2 `intropic` spelling kept verbatim from spec
- A3 APScheduler drives daily refresh; intraday is a plain thread loop
- A4 `pykx` declared as spec'd; falls back to an optional dep group with lazy import if it won't install
- A5 ruff `line-length = 149` (overrides spec's 100)
- A6 `datetime.date` internally, `YYYYMMDD` strings only at boundaries
- A7 Every Runner gets `from_config()` with project-root-anchored defaults

## Post-implementation note (same session)

**D13 (user directive):** `pykx` removed entirely. All kdb+ access goes through the
`apcr_desktool` API via the adapter; Bloomberg likewise. Only S3 keeps a direct third-party
client (boto3). `DeskToolProtocol` therefore covers kdb and Bloomberg as well as symbology,
and its method signatures are a best-guess contract needing one reconciliation pass against
the real package — `RealDeskTool` is the only class that should need editing.

**Outcome:** all 12 phases complete, 388 tests passing, ruff clean. Four spec-level defects
found and fixed during implementation — see the "Deviations" section of PLAN.md.
