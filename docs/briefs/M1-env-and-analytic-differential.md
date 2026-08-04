# M1 — Env + analytic differential

**Track:** agentic · **Size:** one evening (+ one ≤30 min acceptance run) · **Reads
first:** `ARCHITECTURE.md` (invariants 1, 3, 6, 7, 9 bite directly here), then
`docs/briefs/M0-oracle-and-goldens.md` for what already exists.

## Objective

Land `ExecutionEnv` and prove it correct by differential: TWAP and both AC schedules run
*as policies* through the real env, and their simulated cost moments must match the
linear closed forms within the bands pre-stated below. This is the load-bearing
correctness milestone — env bugs die here, not in M2's training curves. Alongside it,
land the two guards ratified at M0 acceptance: the variational optimality certificate for
`optimal_*` (task 0) and the invariant-7 resolution (task 1).

## Context

M0 established that the oracle carries two κ conventions: `ac_*` (vendored FrontierView
convention, golden-pinned at `f87795f6`) and `optimal_*` (discrete stationarity of the
frozen objective — what M2 grades against). The λ-rescaling test proves the two are one
sinh family; it does **not** prove `optimal_*` optimal. Task 0 closes that gap with no κ
formula anywhere in the check. Separately: FrontierView's impact is a 0.6-power law and
§4's "linear temporary (η)" is its tangent η̃ — task 1 resolves how that fact enters
Phase 1 without splitting the objective. **This brief supersedes any earlier note
sourcing the env reward from `linear_cost_moments` and the eval metric from
`cost_moments` — that split would violate invariant 7.**

## Preflight (idempotent — apply if not already done)

Handover items, if the repo docs don't already show them: §9 amendment rewording per the
invariant-7 resolution below; M4 roadmap row led by the calibrated 0.6-power break;
κ-convention item cross-filed on FrontierView's backlog; M0 row flipped to house format.

## Tasks

0. **Variational optimality certificate for `optimal_*`.** In `tests/` (a check, not
   product API): assemble the frozen objective as a quadratic in the interior holdings
   `x_1..x_{N−1}` (boundaries `x_0 = X`, `x_N = 0`), with the tridiagonal Hessian —
   second-difference in η̃/τ plus λσ²τ on the diagonal, units per `oracle/model.py`.
   Then: (a) Cholesky succeeds — positive-definite certificate, so the stationary point
   is the unique global minimum; (b) generic linear solve of the stationarity system
   matches `optimal_trajectory` to ≤ 1e-12 relative (vs X) on the 3 × 3 golden grid;
   (c) perturbation test — 200 random interior directions at ‖δ‖ ∈ {1e-3, 1e-6}·X,
   require `U(x*+δ) − U(x*) ≥ −1e-9·|U(x*)|`; (d) assert x* is monotone so the spread
   term `ε·Σ|n_k|` is the constant `εX` and drops out of the gradient — state this in
   the test, don't assume it silently. No κ formula, no import of the κ solver.
1. **Invariant-7 resolution (normative).** Evaluate the power-law and linear moment paths
   on the Phase-1 golden parameter sets. If they agree ≤ 1e-12 relative across all
   cases, pin that with an equality test and name `linear_cost_moments` the canonical
   encoding. If they do not reduce (expected — exponent 0.6): Phase 1 is the linearized
   world end-to-end — dynamics at tangent η̃, training reward, eval metric, and oracle
   all one encoding — and `cost_moments` is quarantined to *reporting context only*:
   the eval harness keeps two registries, `graded` and `context`; `cost_moments` may
   register only under `context`; a test asserts this. Record the outcome in §9.
2. **`ExecutionEnv` per §4.** Gymnasium API, numpy float64 core (agents cast at their
   boundary, later). Episode/action/dynamics/observation/reward exactly as the
   constitution's contract: shares-this-interval clipped to `[0, remaining]`; forced
   terminal liquidation charged normally; ABM + linear permanent/temporary + spread with
   parameters from the golden cases; observation `(time remaining fraction, inventory
   remaining fraction)`; per-step reward `−shortfall_k − λσ²τx_k²`. Seeding via
   `temper/seeding` spawn keys; all M1 Monte-Carlo draws from a dedicated
   `"m1/differential"` pool — train and eval pools remain untouched for M2.
3. **Baselines as policies.** A wrapper turning any deterministic schedule into a policy;
   named baselines: TWAP, `ac_trajectory(λ)`, `optimal_trajectory(λ)`. Oracle gains
   `schedule_moments(x)` — E[cost]/V[cost] of an *arbitrary* deterministic schedule —
   which is what the differential compares against (the named-schedule moments remain
   golden-pinned from M0).
4. **Monte-Carlo differential.** For each (case, schedule) pair below, run N_sim episodes
   through the real `step` loop (no vectorised side-channel — the env is the thing under
   test), standardise costs `z_i = (C_i − E)/√V` against `schedule_moments`, and require
   the bands in the table. Rationale for exact bands: under Phase-1 dynamics a
   deterministic schedule's shortfall is *exactly* Gaussian, so the null distributions of
   both statistics are known, not asymptotic.
5. **Exact identity tests (non-statistical, per episode).** (a) Per-step shortfalls
   telescope to total IS, ≤ 1e-10 relative; (b) for a fixed schedule the summed
   inventory penalty equals λ·V[cost] exactly (≤ 1e-12 relative) — this mechanically
   pins invariant 7: the env pays out the same functional the oracle computes;
   (c) summed reward equals −(realized IS + λV); (d) an under-trading policy triggers
   forced terminal liquidation and the realized schedule's cost matches
   `schedule_moments` of that realized schedule; (e) determinism — identical
   `(config, seed)` ⇒ bitwise-identical trajectory arrays.
6. **Config + enforcement.** `configs/m1_differential.yaml` names every case, N_sim,
   band, and seed — the tests read the config, not literals. Extend
   `tests/test_repo_invariants.py`: no torch imports under `temper/env/`.

## Pre-stated numbers (invariant 3 — loosen only by amending this brief before work)

| Item | Value |
| --- | --- |
| Fast tier (in `make test`) | 3 cases (one per symbol, middle λ) × 3 schedules, N_sim = 20,000 |
| Deep tier (`make differential`, marker `deep`) | full 3 × 3 golden grid × 3 schedules, N_sim = 100,000 |
| Mean band (standardised) | \|mean(z)\| ≤ 4/√N_sim |
| Variance band (standardised) | \|var(z) − 1\| ≤ 4·√(2/N_sim) |
| Identity tests | ≤ 1e-10 relative (1e-12 for the penalty ≡ λV pin) |
| Variational match / PD / perturbation | ≤ 1e-12 rel · Cholesky succeeds · ≥ −1e-9·\|U\| |
| Runtime budget, reference box | fast tier ≤ 90 s added to `make test` (suite total ≤ 3 min); deep tier ≤ 30 min |

Resolution these bands buy (state in the test docstring, it's the point): fast tier
detects mean shifts ≳ 2.8% of σ_C and variance mis-scalings ≳ 4%; deep tier ≳ 0.9% and
1.3%. The κ-class bug (~18% on the objective) and off-by-one-in-`Σx_k²` class (~2–5%)
both die in the deep tier, which is the acceptance gate.

## Definition of done

- [ ] Clean clone → `make test` green (mingw32-make / `python -m pytest` per README),
      suite ≤ 3 min on the reference box.
- [ ] `make differential` green, ≤ 30 min, run at least once at acceptance.
- [ ] Task-0 certificate green: PD + solve-match + perturbation + monotonicity assert.
- [ ] Invariant-7 outcome recorded in §9; registry test enforcing the quarantine (or the
      equality pin) green.
- [ ] All five identity tests green; diagnostic seed pool used; train/eval pools untouched.
- [ ] No new dependencies beyond M0's pins; repo-invariants extension green.
- [ ] `ROADMAP.md` M1 row flipped; anything structural discovered → §9, not code comments.

## Out of scope (resist)

PPO and anything that trains; figures and plotting deps; power-law *dynamics* in the env
(that is M4's assumption break — M1 only touches the power-law *moments* in task 1's
resolution); Phase-2 market models; `client/`; oracle API changes beyond
`schedule_moments`.

## Session notes

- The Gaussian-exactness argument is why the bands need no scipy: standardised-cost
  bounds with z = 4 are exact-null; do not add a dependency for χ² quantiles.
- If the fast tier blows its 90 s budget, cut fast-tier *cases* (never N_sim below
  20,000), and amend the table here first — invariant-3 process applies to runtime
  trades too.
- At small λ all three schedules converge toward TWAP; the differential is env-vs-formula
  per schedule, not schedule-vs-schedule, so near-degenerate cases are fine.
- If any moment mismatch survives the deep tier, stop and report — do not tune the env
  toward the bands. A differential failure is the milestone's product, same as M0's κ
  finding.
