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
| Deep tier (`make differential`, marker `deep`) | full 3 × 3 golden grid × 3 schedules, N_sim = **200,000** (amended 2026-08-04, see below) |
| Mean band (standardised) | \|mean(z)\| ≤ 4/√N_sim |
| Variance band (standardised) | \|var(z) − 1\| ≤ 4·√(2/N_sim) |
| Identity tests | ≤ 1e-10 relative (1e-12 for the penalty ≡ λV pin), **relative to the summed absolute terms** — see amendment 3 |
| Variational match / PD / perturbation | ≤ 1e-12 rel · Cholesky succeeds · ≥ −1e-9·\|U\| |
| Runtime budget, reference box | fast tier ≤ 90 s added to `make test` (suite total ≤ 3 min); deep tier ≤ 30 min |

Resolution these bands buy (state in the test docstring, it's the point): fast tier
detects mean shifts ≳ 2.8% of σ_C and variance mis-scalings ≳ 4%; deep tier ≳ 0.89% and
1.26%. The κ-class bug (~18% on the objective) and off-by-one-in-`Σx_k²` class (~2–5%,
26% for TWAP at N = 13) both die in the deep tier, which is the acceptance gate.

### Amendments (invariant 3 — recorded before the work they licensed, 2026-08-04)

Three pre-stated numbers moved. Each is recorded here, with its reason, per
`docs/briefs/M1a-acceptance-hardening.md`'s preflight.

1. **Deep-tier N_sim: 100,000 → 200,000.** Reason: it restores the resolution this
   brief's own prose claimed, and the margin on the weakest named bug class. At
   N = 100,000 the low end of the off-by-one-in-`Σx_k²` class (~2%) sits 4.47σ against a
   4σ gate — roughly two detections in three. At N = 200,000 it is 6.3σ (~99%). Both the
   table above and `configs/m1_differential.yaml` carry 200,000; the config is what the
   tests read.
2. **Deep-tier resolution prose: ≳ 1.3% / 1.8% → ≳ 0.89% / 1.26%.** These are the
   N = 200,000 figures (4/√N = 0.894%, 4·√(2/N) = 1.265%). The original 0.9%/1.3% prose
   in this brief was *already* the N = 200,000 pair while the table pre-stated 100,000;
   amendment 1 makes the table match the prose rather than the other way round.
3. **The identity-test tolerance denominator.** "≤ 1e-10 relative" now reads *relative to
   the summed absolute terms*, not to the surviving total. Reason: `Σr = −(IS + λV)`
   cancels ~1e2 bps per-bin terms down to totals that are occasionally ~1e-2 bps, so a
   total-relative verdict is seed-dependent while round-off is ~1e-14 absolute either
   way. The bar itself is unchanged; what it is relative to is now stated. Worst observed
   use of that budget: 8.8e-12.

Amendment 3 generalises past Temper: *scale-relative tolerances for cancelling
identities* is now a house convention, cross-filed alongside the Anvil/Crucible notes.

## Definition of done

- [x] Clean clone → `make test` green (mingw32-make / `python -m pytest` per README),
      suite ≤ 3 min on the reference box. **681 tests, 14 s** (M1a added 179), of which
      the fast differential tier is ~11 s across its 9 cells.
- [x] `make differential` green, ≤ 30 min, run at least once at acceptance.
      Re-run at N_sim = 200,000 for M1a: **27 cells, 317 s**, 70,200,000 steps through the
      real loop at 221 k steps/s — the step count asserted, not inferred.
      No cell used more than 61% of its band; worst mean-band use 61% (fast, MSFT `ac`),
      worst variance-band use 50% (deep, MSFT λ=1e-3 `ac`).
- [x] Task-0 certificate green: PD + solve-match + perturbation + monotonicity assert.
      `tests/test_variational_certificate.py`, 9 cases × 5 checks; the generic solve
      matches `optimal_trajectory` and the quadratic is pinned to the oracle's own
      objective so the certificate cannot certify an invention of its own.
- [x] Invariant-7 outcome recorded in §9; registry test enforcing the quarantine green.
      `tests/test_objective_registry.py` — refusal, behavioural and static checks.
- [x] All five identity tests green; diagnostic seed pool used; train/eval pools untouched.
      Worst observed: telescope 1.6e-13, penalty ≡ λV **5.8e-16** (band 1e-12), the exact
      expectation identity 1.1e-12, reward 8.8e-12 (bands 1e-10).
- [x] No new dependencies beyond M0's pins; repo-invariants extension green.
      `requirements.txt` unchanged — gymnasium and pyyaml were already pinned there.
- [x] `ROADMAP.md` M1 row flipped; anything structural discovered → §9, not code comments.

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

### Closed 2026-08-04

No moment mismatch. The env matched the closed forms on the first run, before any of the
identity tests were written, which is the outcome the milestone was shaped to make
falsifiable rather than the one it was shaped to produce.

**1. Task 1 resolved to the quarantine branch, and the M0 handover was wrong.**
The two encodings differ by 12%–54% of expected cost on the Phase-1 golden sets — a power
law against its own tangent agrees only where they touch, and at TWAP participation the
tangent charges exactly β = 0.6 of the power law. So M0's instruction to source the env
reward from `linear_cost_moments` and the eval metric from `cost_moments` would have
trained the agent on one functional and graded it on another, differing by up to half the
number being reported: invariant 7 violated in the one way it was written to prevent.
Phase 1 is now the linearised world end-to-end, `cost_moments` is quarantined to
`temper.eval.metrics.CONTEXT`, and the resolution is in `ARCHITECTURE.md` §9. M0's brief
carries a supersession note so the wrong instruction cannot be followed twice.

**2. An exact per-episode expectation identity, beyond the brief's five.**
The env publishes the cumulative price shock each bin executed against, so a test can
subtract the noise off a *single* episode and compare what remains against the oracle's
`E[cost]` — no averaging, no CI. It holds to ~1e-12 for every schedule including the
force-liquidated one. This is a strictly stronger statement than the Monte-Carlo tiers can
make about the mean, and it is what makes task 5(d) a per-episode identity rather than
another statistical test. The tiers still earn their place: they are the only check on the
*distribution*, and they are what a variance bug dies to.

**3. Two identities cancel, so their tolerance is relative to the terms, not the total.**
`Σ r_k = −(IS + λV)` sums per-bin quantities of ~1e2 bps into a total that is occasionally
~1e-2 bps, when an episode's price path happens to offset its impact charges. Round-off is
~1e-14 absolute either way, so a tolerance relative to the surviving total makes the
verdict depend on which Gaussian draws came up. The comparison is against the magnitude of
the summed terms; the pre-stated 1e-10 is unchanged and what it is relative to is now
stated. Worst observed use of that budget: 8.8e-12.

**4. `.gitignore` was excluding the entire env package.** The virtualenv entry was a bare
`env/`, and a bare directory pattern matches at any depth, so `temper/env/` was invisible
to git from the moment it was created — `make test` green locally, a missing module from a
clean clone, which is the gate every milestone is measured by. The patterns are now
anchored to the repo root, and `tests/test_repo_invariants.py` asks git directly whether
any package source is excluded (skipped when git is unavailable). Worth the guard: this
class of bug is invisible to every test that runs in the working tree.

**5. Deliberately absent.** No timing assertion. The runtime budgets are measured and
reported by the differential module (`make differential` prints tier wall time against the
config's budget) but never asserted: a test that goes red because the box was busy teaches
a session to rerun until green, which is the opposite of what the suite is for.

### Amended by M1a (2026-08-04) — what the Monte-Carlo tiers are now *for*

`docs/briefs/M1a-acceptance-hardening.md` task 1 landed the exact per-episode noise
identity: the realised cost minus the oracle's `E[cost]` equals
`−σ_bin · Σ_{k=0}^{N−1} (x_k/X)·ξ_k` to ~1e-13 relative, on all 27 deep cells, the
force-liquidated under-trader and the asymptote guard case. **That changes what the
tiers certify.** The variance of a known linear form in iid standard normals is
arithmetic, so `V = σ²τ Σ x_k²` now holds *by construction*, not by sampling — the tiers
no longer gate the cost assembly. What they still and only certify is the **draws**: that
they are iid standard normal and uncorrelated across bins. Belt and braces, and a smaller
claim than it was.

M2 should read the guarantee list this way:

| Statement | How it is held |
| --- | --- |
| The env pays out the oracle's functional (invariant 7) | **Exact**, per episode — penalty ≡ λV to 5.8e-16 |
| Realised cost less the price path is the oracle's `E[cost]` | **Exact**, per episode |
| Realised noise is the right linear form in the right draws | **Exact**, per episode (M1a task 1) |
| Every episode ran through the real `step` loop, bin by bin | **Exact** — step counter, 70,200,000 deep / 2,340,000 fast |
| The shocks are iid standard normal, uncorrelated across bins | **Sampled** — the 4σ bands, and nothing else |

**Watch items for M2.**

- The env takes seeds only by *pool address*, never as a raw integer — `reset(seed=i)`
  means "stream `i` of this env's pool". PPO wrappers that pass entropy-derived seeds will
  need adapting, deliberately: it makes invariant 5 unbreakable rather than observed.
- `EPISODE_KEY` is `"episode_summary"`, not `"episode"`, because gymnasium's
  `RecordEpisodeStatistics` owns the latter and would overwrite it.
- Baselines read their bin index out of the observation clock, so a vectorised or
  frame-skipping rollout that perturbs the observation sequence will fail loudly instead of
  silently replaying bin 0.
