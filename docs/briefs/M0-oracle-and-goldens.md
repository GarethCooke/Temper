# M0 — Oracle + goldens

**Track:** agentic · **Size:** one evening · **Reads first:** `ARCHITECTURE.md` (the
constitution; this brief must not violate it, and invariants 1–3 bite directly here).

## Objective

Stand the repo up and land `temper/oracle`: the Almgren–Chriss closed forms, proven
equivalent to FrontierView's compute core by matching vendored golden fixtures within the
tolerances pre-stated below. When M0 is done, the reference answer every later milestone
is graded against exists, is tested, and regenerates green from a clean clone.

## Context

Temper's whole claim structure rests on the oracle being *independently trustworthy* —
it is the `ref_engine.py` of this project. FrontierView's `api/market_impact.py` +
`api/parameters.py` are the source of truth for parameters and schedules; Temper
reimplements the closed forms and must agree with fixtures exported from that repo. Zero
FrontierView changes (constitution §7): the export is a read-only script run there once.

## Owner input (before or at session start)

Run a small export in the FrontierView repo and commit the output to Temper at
`tests/golden/vendor/frontierview_goldens.json`. The **schema below is the contract**; the
snippet is indicative — adapt call names to `market_impact.py`'s actual API:

```python
# run from the FrontierView repo root; stdlib + the repo's own deps only
import json, subprocess, datetime
from api.parameters import SYMBOL_PARAMS          # + calibrated ALMGREN_ETA / GAMMA
from api import market_impact as mi               # AC-optimal + TWAP schedules, E/V, decomposition
# for ~3 symbols × ~3 risk aversions: emit trajectory, trade list, E[cost], V[cost],
# temporary/permanent/spread decomposition, plus TWAP moments, with full params inline
```

```json
{ "provenance": { "source": "FrontierView", "commit": "<sha>", "dirty": false,
                  "generated": "<iso8601>", "exporter": "tools/export_frontierview_goldens.py",
                  "modules": [], "python": "" },
  "conventions": { "cost_units": "bps of notional", "variance_units": "bps^2",
                   "trading_hours_per_day": 6.5, "temp_exponent": 0.6,
                   "n_bins_rule": "max(2, round(horizon_hours * 2))",
                   "sigma_bin": "sigma_daily * sqrt(dt_hours / trading_hours_per_day)" },
  "grid": { "horizon_hours": 6.5, "n_bins": 13, "dt_hours": 0.5 },
  "cases": [ {
    "case_id": "", "tag": "core", "symbol": "AAPL", "X": 100000, "lambda": 1e-6,
    "horizon_hours": 6.5, "n_bins": 13, "dt_hours": 0.5,
    "params":  { "adv": 0.0, "sigma": 0.0, "half_spread": 0.0, "eta": 0.0, "gamma": 0.0 },
    "derived": { "v_hourly": 0.0, "sigma_bin": 0.0, "eta_tilde": 0.0,
                 "kappa": 0.0, "kappa_T": 0.0 },
    "ac":   { "trajectory": [], "trades": [], "participation": [],
              "expected_cost": 0.0, "variance": 0.0,
              "decomposition": { "temporary": 0.0, "permanent": 0.0, "spread": 0.0 } },
    "twap": { "trajectory": [], "trades": [], "participation": [],
              "expected_cost": 0.0, "variance": 0.0,
              "decomposition": { "temporary": 0.0, "permanent": 0.0, "spread": 0.0 } } } ],
  "frontier": { "symbol": "AAPL", "X": 100000, "horizon_hours": 6.5, "n_bins": 13,
                "points": [ { "lambda": 1e-6, "expected_cost": 0.0, "variance": 0.0,
                              "expected_cost_rounded_4dp": 0.0,
                              "variance_rounded_4dp": 0.0 } ] } }
```

Whatever units/grid FrontierView emits are canonical — the goldens, not this brief, are
the numeric spec (constitution §4). If the export surfaces a mismatch between this schema
and what the compute core naturally produces, adjust the schema *in this brief, before
oracle work starts*, and note it below.

### Schema adjustments made before oracle work (2026-08-04)

The indicative schema assumed a textbook AC parameterisation in abstract units. The
compute core is parameterised differently, so the schema above was amended first:

| Adjustment | Why |
| ---------- | --- |
| `grid` is `{horizon_hours, n_bins, dt_hours}`, canonically 6.5 h / 13 bins, not `{T: 1.0, N: 50}`. | FrontierView bins a horizon *in trading hours* into half-hour slots by `max(2, round(2T))`. There is no grid where N = 50 is natural. |
| `params` is `{adv, sigma, half_spread, eta, gamma}`, not `{sigma, eta, gamma, spread}`. | `adv` sets `v_hourly`, without which no impact term can be evaluated; `sigma` is *daily* fractional vol; the spread parameter is a bps half-spread. |
| Added a `conventions` block. | Costs are bps of notional and variance bps², not currency. Making the units contract machine-readable lets the golden loader assert the oracle compiles against the same one. |
| Added `derived` per case (`v_hourly`, `sigma_bin`, `eta_tilde`, `kappa`, `kappa_T`). | A single E-mismatch localises to one formula instead of "somewhere in the model". Cheap to emit, and it caught the κ question below during implementation. |
| Added `participation` and per-case `horizon_hours` / `n_bins` / `dt_hours`. | Participation rate is the compute core's native schedule representation; per-case grid fields let edge cases use non-default horizons. |
| TWAP carries the full block (trajectory, trades, decomposition), not just E and V. | Strictly more pinning at no cost; TWAP is a baseline on every later chart. |
| Added a `frontier` block over FrontierView's own 17-point λ grid, at full precision plus its 4-dp published values. | Task 2 requires the (E, V) frontier point at a given λ, so the locus should be pinned rather than one point per case. `generate_frontier` rounds to 4 dp, far coarser than the 1e-6 tolerance, hence the recomputation *and* the rounded cross-check. |
| Added a `tag` per case and six edge cases beyond the 3 × 3 core. | Guarded branches — κ floor, `sinh` overflow asymptote, N = 2 minimum, participation floor, high participation, banker's rounding at N = 4.5 — are exactly where two implementations of one formula diverge. |

The export runs from `tools/export_frontierview_goldens.py`, which lives in Temper and is
read-only with respect to FrontierView: it imports `api.market_impact`, writes only into
Temper, and records the upstream sha and whether that tree was dirty. Keeping the script
here rather than there keeps the export reproducible without touching upstream (§7).

## Tasks

1. **Scaffold.** Package layout per constitution §3; `requirements.txt` pinned
   (numpy, torch CPU, gymnasium, pytest, pyyaml — plotting stack is a later brief);
   `Makefile` with `test`; Python ≥ 3.11; `.gitignore` for venv/results scratch.
2. **`temper/oracle`.** Discrete AC: κ solve, sinh trajectory + trade list, E[cost],
   V[cost], the (E, V) frontier point at given λ, and TWAP moments under the same
   dynamics. Pure numpy, no torch, no I/O beyond loading fixtures in tests.
3. **Golden tests.** Every vendored case: trajectory, trade list, E, V, and decomposition
   compared at the tolerances below; provenance block asserted present.
4. **Seed utilities.** A tiny `temper/seeding.py` (spawn disjoint train/eval pools from a
   root seed) + determinism test — M1's env depends on it.
5. **README.** Flip the build section from "intended shape" to real commands.

## Tolerances (pre-stated — constitution invariant 3)

| Quantity                         | Tolerance                        |
| -------------------------------- | -------------------------------- |
| Trajectory / trade list          | ≤ 1e-6 relative (vs X)           |
| E[cost], V[cost], decomposition  | ≤ 1e-6 relative                  |

Loosen only by amending this brief *before* implementation, with the reason recorded here.
**Not loosened.** Observed worst case, across all 16 cases and 17 frontier points:

| Quantity                        | Tolerance | Worst observed | Where |
| ------------------------------- | --------- | -------------- | ----- |
| Trajectory / trade list         | 1e-6 of X | **3.5e-16**    | `core` MSFT, TWAP trajectory |
| E, V, decomposition             | 1e-6 rel  | **9.5e-16**    | `core` MSFT, TWAP variance |
| Frontier (E, V) over 17 λ       | 1e-6 rel  | **4.3e-16**    | — |

Agreement is at float round-off, so the tolerance is doing no work: any future failure
will be a real divergence, not tolerance creep.

## Definition of done

- [x] Clean clone → `python -m venv … && pip install -r requirements.txt && make test` green.
      292 tests, 0.4 s, on the pinned requirements.
- [x] Oracle matches **every** vendored case within the stated tolerances (see table above).
- [x] Determinism test green (identical root seed ⇒ identical pools; train/eval disjoint).
      `tests/test_seeding.py` also pins order-independence and prefix stability, which
      committed results depend on.
- [x] No network access, no GPU, anywhere in the test path. Enforced, not just observed:
      `tests/test_repo_invariants.py` statically rejects networking imports anywhere under
      `temper/`, rejects torch under `temper/oracle/`, and runs the oracle with
      `socket.socket` replaced by a raising stub.
- [x] `ROADMAP.md` status flipped; three structural findings recorded in
      `ARCHITECTURE.md` §9 (see session notes).

## Out of scope (resist)

The environment, any agent code, figures, plotting deps, config schema beyond what the
golden loader needs. M1 owns the env; M2 owns PPO.

## Session notes

If the vendored fixture file is missing, stop and request it — do not synthesise goldens
from this session's own AC implementation; that would collapse the differential into a
tautology, which is the one failure mode this milestone exists to prevent.

### Closed 2026-08-04

The fixture was exported from FrontierView at `f87795f6` (clean tree) before any oracle
code was written, so the differential is genuine. Three findings, all in
`ARCHITECTURE.md` §9; the first is the one M1 and M2 must read.

**1. FrontierView's κ is not the argmin at that λ — the oracle carries both.**
`κ² = λ σ_bin²/η̃` is a continuum-limit expression that drops a factor of `X·τ/10⁴`.
Writing the linearised objective as `A Σ(Δx)² + B Σx²`, the discrete stationarity
condition is `cosh(κτ) = 1 + μ/2` with `μ = λ σ_bin² · 10⁴ · τ / (X η̃)`; the two rates
differ by ~2.2× for AAPL at 100 k shares. Both trajectories are sinh, so the difference is
invisible by eye and would have surfaced in M2 as an agent apparently *beating* the
reference by up to ~18 % on the frozen objective (λ = 1e-4) — which §1.1 names a red flag,
not a result. `oracle.ac_kappa` / `ac_trajectory` reproduce the vendored convention and
are what the goldens pin; `oracle.optimal_kappa` / `optimal_trajectory` solve the discrete
condition and are what M2 onward grades against. `tests/test_oracle_properties.py` pins
the exact λ-rescaling that maps one onto the other, so the pair cannot silently drift
apart, and a guard test fails if anyone collapses them.

**2. Temporary impact is a 0.6-power law; §4's "linear η" is the tangent η̃.**
The power law admits no sinh closed form — FrontierView linearises only to *derive* the
schedule, then charges the power law to *cost* it. The oracle keeps both explicit:
`cost_moments` (power law, golden-pinned) and `linear_cost_moments` (tangent, the frozen
objective of §6.7). ~~**M1 must encode the env reward from `linear_cost_moments` and the
eval metric from `cost_moments`**~~ — one objective, encoded once, per invariant 7.

> **⚠ Superseded — do not implement this sentence.** The split would violate invariant 7:
> training reward and eval metric would be different functionals. Ratified wrong at M0's
> closing review; resolved by M1 task 1. Phase 1 is the linearised world end-to-end —
> dynamics at tangent η̃, reward, eval metric and oracle all one encoding — and
> `cost_moments` is quarantined to reporting context, enforced by refusal, behaviour and
> static checks (`tests/test_objective_registry.py`). The two encodings differ by 12 %–54 %
> of expected cost on the Phase-1 golden sets, so this was a real defect rather than a
> stylistic one. See the invariant-7 entry in `ARCHITECTURE.md` §9 (2026-08-04) and
> `docs/briefs/M1-env-and-analytic-differential.md` task 1.

**3. Scope: two things beyond the literal task list, both load-bearing.**
`optimal_kappa` / `optimal_trajectory` / `optimal_trajectory_by_solve` exist because of
finding 1 — without them the M2 rediscovery claim is not well-posed, and the brief's
"κ solve" is the discrete solve. `tests/test_repo_invariants.py` exists because the "no
network, no GPU" line in the definition of done is a property nothing else would check.
Neither touches the env, agents, figures or config schema. `configs/` and `results/` are
placeholders only.

**Not done, deliberately.** No commit was made — the repo is `git init`-ed with everything
untracked, for the owner to review and commit. `temper/env/`, `temper/agents/` and
`temper/eval/` were not created: empty packages that claim territory M1 and M2 own are
worse than absent ones.

**Watch item for M1.** The `sinh-overflow-asymptote` branch leaves terminal inventory at
`X·e^{−κT}` rather than exactly zero. Harmless in the oracle (≈1e-218 of X), but the env's
`x_N = 0` force-liquidation must not be written to assume the oracle's trajectories
already end at a hard zero.

> **Closed by M1a — unreachable, not merely untriggered.** M1a task 4,
> `tests/test_sinh_asymptote_guard.py` (2026-08-04). The caution
> was right and the code honours it: the env does not assume a hard-zero tail and
> `SchedulePolicy` does not repair one. But the residual can never reach the
> force-liquidation, so the item is closed rather than carried into M2. Taking the branch
> requires `κT > 500`, so on the canonical 13-bin grid the per-bin decay is at most
> `e^{−500/13}` = 2.0e-17 — below half an ulp at `X`. The first bin's planned trade
> `X − x₁` therefore rounds to exactly `X`, the env holds zero from bin 1 onward, and the
> terminal residual is annihilated some 200 orders of magnitude before anything could
> charge it. No cell of the 3 × 3 grid reaches the branch either (largest `κT` = 20.6
> vendored, 9.0 exact); a dedicated guard case in `configs/m1_differential.yaml` — a
> guard, **not** a golden — runs the branch through the env anyway, where it pins that the
> differential costs the schedule the env *realised* rather than the one the policy
> planned. The vendored `sinh-overflow-asymptote` golden could not have served as that
> guard case either: at its λ = 100 the residual underflows to exactly 0.0, so it takes the
> branch without ever producing the residual this watch item is about. M1a's guard uses
> λ = 1.0 and is deliberately not a golden.
