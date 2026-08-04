# Vendored artefact — FrontierView Almgren–Chriss goldens

| | |
| --- | --- |
| **Artefact** | `tests/golden/vendor/frontierview_goldens.json` |
| **Source** | FrontierView, `api/market_impact.py` + `api/parameters.py` |
| **Source commit** | `f87795f670591d2a0ffa63ba02ab95abe02a65cc` (clean working tree) |
| **Generated** | 2026-08-04, Python 3.12.2 |
| **Exporter** | `tools/export_frontierview_goldens.py` (lives here, runs there) |
| **Consumed by** | `tests/test_oracle_golden.py` |

## Why this exists

`temper/oracle` is Temper's reference engine (constitution invariant 2). Its
authority comes from agreeing with an implementation Temper did not write, so
the fixture must be produced by FrontierView's compute core and nothing else.
Goldens synthesised from Temper's own closed forms would make every downstream
claim circular — the M0 brief's one hard stop.

## Regenerating

Zero upstream changes (constitution §7): the exporter is read-only with respect
to FrontierView. It imports `api.market_impact`, writes only into Temper, and
records the source commit and whether that checkout was dirty. The provenance
test fails on a dirty export, because then the recorded sha does not identify
the code that produced the numbers.

```bash
make goldens FRONTIERVIEW=/path/to/FrontierView
```

Re-exporting after an upstream change is expected to turn `test_oracle_golden`
red. That is the differential doing its job: reconcile the oracle with the new
numbers, or establish that upstream regressed. Never edit the fixture by hand.

## What the fixture pins

16 cases and a 17-point frontier. Nine core cases are three symbols × three risk
aversions on the canonical full-day grid; the rest exist to pin the guarded
branches, which are exactly where two implementations of the same formula
usually part company:

| Tag | Pins |
| --- | --- |
| `core` | AAPL / MSFT / JPM × λ ∈ {1e-7, 1e-5, 1e-3}, T = 6.5 h, N = 13 |
| `kappa-floor` | λ small enough that the `max(…, 1e-12)` clamp on κ² binds |
| `near-twap-lambda` | λ = 1e-12: κT ≈ 6.5e-4, the flat-schedule limit |
| `sinh-overflow-asymptote` | κT ≈ 6500, so the `exp(−κt)` branch replaces `sinh` |
| `two-bin-horizon` | T = 1 h → the N = 2 minimum |
| `participation-floor` | X = 1 share, so `p₀` hits the 1e-4 linearisation floor |
| `high-participation` | 5 % of SPY's ADV — the far end of the power law |
| `half-bin-rounding` | T = 2.25 h → `round(4.5) = 4`, banker's rounding |

Each case carries its full parameter set, the derived intermediates
(`v_hourly`, `sigma_bin`, `eta_tilde`, `kappa`) so a differential failure
localises to one formula, and for both the AC and TWAP schedules: inventory
trajectory, trade list, participation rates, expected cost, shortfall variance
and the temporary/permanent/spread decomposition.

The `eta_tilde` floor in `_linearised_eta` is *not* pinned: it is unreachable
for any parameter set the model would accept.

## Observed upstream quirks (recorded, not corrected)

Read-only observations from the export. None of these are defects Temper may
fix — FrontierView's behaviour is the numeric spec, and the goldens encode
whichever way it actually goes.

1. **Temporary-impact docstring vs code.** `model_assumptions.md` §2 and the
   docstring of `temporary_impact()` both write the power law over
   `|v| / (6.5 · V_hourly)`; the code divides by `V_hourly` alone, with an
   inline comment asserting that is correct (both quantities are shares/hour).
   The code is what the goldens capture and what `temper/oracle` reproduces.
2. **κ is a continuum-limit expression.** `κ² = λ σ_bin² / η̃` is not
   dimensionally the stationary point of the discrete objective the same module
   evaluates. It is a perfectly good risk-aversion dial for a UI that sweeps λ;
   it is not the argmin at that λ. Temper carries both conventions — see
   `ARCHITECTURE.md` §9 (2026-08-04) and `temper/oracle/schedules.py`.
3. **The asymptotic branch does not fully liquidate.** In the `exp(−κt)` branch
   terminal inventory is `X·e^{−κT}` rather than zero (≈ 1e-218 of X at the
   threshold). Reproduced, not rounded away.
