# M3 — Antithetic validation and the risk–cost frontier

**Track:** agentic · **Size:** one unattended night for task 1, then a sweep sized by task
1's measured result · **Reads first:** `ARCHITECTURE.md` §4, §6.9 and invariants 1, 3, 4, 5,
7; then `docs/briefs/M2-ppo-rediscovery.md` — its amendment 1 and its reproducibility
finding are what this brief is built on.

## Objective

Establish antithetic pairing as a reward regime that gives M2's control-variate precision
without M2's weakened claim, then sweep λ to produce the risk–cost frontier: agent, TWAP,
`ac_trajectory` and `optimal_trajectory` on one figure, with the agent tracking the optimal
frontier across the grid.

This brief is **gated**. Task 1 either validates the pairing or it does not, and the sweep's
grid, seed count and reward regime are all decided from task 1's measured numbers rather
than from the estimates in this brief. Do not start task 2 before task 1 is reported and
accepted.

## Context — why antithetic, and why validation first

M2 measured the problem: at 1:70 SNR, sampled-reward training is a lottery — median gap
0.098 against ε = 0.05, one seed in five with a trajectory deviation exceeding the parent
order. The control variate fixed it to 0.0002 but subtracts the analytic noise form, which
weakens the claim to "the optimiser finds AC's minimum" and — the part that matters —
**does not transfer to Phase 2**, where cost stops being affine in the shocks and no closed
form exists to subtract.

Antithetic pairing runs each episode as (ξ, −ξ) and averages the two rewards. It works here
for a structural reason established in M1: the observation carries no price, so the agent
takes **identical actions in both halves**. Same schedule, mirrored shocks, and because
Phase-1 cost is affine in the shocks the noise cancels exactly on the average. It never
touches the analytic noise form — it needs only the ability to replay with negated draws —
so in Phase 2 it degrades to partial cancellation rather than disappearing.

Validation is first because the sweep is expensive and the argument above is an argument.
M2's control variate gives a known answer to four decimals at a known λ. If antithetic
reproduces it, the mechanism is confirmed. If it does not, one night has been spent finding
out instead of six days.

**Cost, from M2's measured per-seed times (~1600 s at 512 envs, 8 threads, 1802 updates).**
The reference box has no headroom — 512 envs × 8 threads saturates 8 physical cores, and
M2's first discarded run of the night was two concurrent sweeps truncating each other. The
schedule is therefore strictly serial. At a naive 2× for pairing, 17 λ × 10 seeds is ~150 h.
That is why task 1 measures the real multiplier and task 2's grid is sized from it.

## Tasks

1. **Validate the pairing at M2's λ. Gate for everything else.**

   At λ = 10^−3.5, 10 seeds from the `train` pool, antithetic reward regime, everything else
   identical to M2's committed configuration. Grade through the existing `GRADED` registry —
   analytic, against `optimal_trajectory`, unchanged.

   Acceptance is against **M2's control-variate result, not against ε**: median gap
   ≤ 0.002, i.e. within an order of magnitude of the CV's 0.0002. The point is not that
   antithetic is good, it is that antithetic reproduces the answer a zero-variance reward
   already established. A median in the CV's neighbourhood confirms the cancellation; a
   median near the sampled regime's 0.098 means the pairing is not cancelling and the
   argument is wrong.

   Report, all of them: median and IQR of the gap fraction; per-seed values; median ‖δ‖₂
   against the derived band; **measured per-seed wall-clock**; and the realised reward
   variance per update against the sampled regime's, which is the direct evidence that
   cancellation happened rather than being inferred from the outcome.

   Two structural checks, because they are cheap now and expensive later:
   - **Action identity across the pair.** Assert the two halves of a pair produce
     bitwise-identical action sequences. This is the assumption the whole method rests on,
     and it fails silently and instantly the moment an observation carries price. Make it a
     permanent test, not a one-off diagnostic.
   - **Shock negation is exact.** Assert the mirror's draws are the exact negation of the
     first half's, elementwise, not a fresh draw from a mirrored distribution.

2. **Measure the real pairing multiplier, and report it before sizing the sweep.**

   A pair shares one schedule and one set of network forward/backward passes; only the
   environment stepping and the reward assembly differ. If the mirror can be carried as 512
   additional envs inside the same batch, the cost is extra env-steps, not a second training
   run — plausibly ~1.3× rather than 2×. Implement whichever way is correct, and **report
   the measured multiplier**. The sweep's grid depends on it and this brief deliberately
   does not assume it.

   While the M2 traces are open: report the update at which the CV runs' objective went flat.
   1802 updates was M2's blind budget. If convergence is at 1200, the last third is heat, and
   cutting it is an evidence-based reduction rather than a guess. Any cut is recorded as an
   amendment here before the sweep runs.

3. **Size and fix the grid — from task 1 and 2's numbers, then amend this brief.**

   Ten seeds per point is not negotiable: M2's IQR of 0.081 against a 0.05 tolerance is what
   makes five insufficient, and a frontier whose error bars exceed the effect is
   uninterpretable. **Thin the λ grid instead.** The frontier is a smooth curve; nine
   well-spaced log-uniform points draw the same shape as seventeen.

   Record the chosen grid, the seed count, the measured per-seed cost and the total
   wall-clock estimate in this brief before the sweep starts. If the estimate exceeds ~24 h,
   thin further. Grid points must include M2's λ = 10^−3.5 so the sweep contains a run
   directly comparable to a committed result.

4. **The sweep.**

   Each λ point: 10 seeds, antithetic (or the regime task 1 established), graded
   analytically. Per point report median and IQR of the gap fraction, median ‖δ‖₂ against
   that λ's derived band, and the three baselines through the identical grader.

   The red-flag test — `J_agent ≥ J_optimal − 1e-9·|J_optimal|` — is a **hard failure on
   every seed at every λ**, unchanged from M2. `optimal_trajectory` is certified as the
   unique global minimum, so a negative gap is a bug, never a win.

5. **The frontier figure.**

   `results/m3_frontier.*`: expected cost against variance, agent with IQR band plus TWAP,
   `ac_trajectory` and `optimal_trajectory`, over the λ grid.

   **Plot the excess over the floor, not total V.** §9 (M1a): `V` floors at `σ_bin²X²` —
   every schedule pays one bin of volatility on the full position, roughly 20 % of TWAP's V
   at N = 13 and more for front-loaded schedules. Plotting total V compresses the separation
   exactly where the schedules differ most, at the high-λ end. Report both axes' definitions
   in the caption so the choice is visible rather than silent.

   **Draw every seed trace, not only the band.** M2's committed figure hid a failed seed
   behind an IQR band that looked tight. At n = 10 the IQR spans seeds 3–8 and structurally
   cannot show the extremes. Individual traces at any n below ~10, and here regardless.

6. **README honesty ladder.** Write it now, while the limits are fresh:
   - **Phase 1** — the pipeline works: the agent recovers AC in a world where AC is provably
     optimal.
   - **Phase 2** — the agent beats AC where AC's formula breaks, **inside an AC-shaped
     market** with realistic impact curvature.
   - **Phase 3** — it runs on a wire against synthetic data. Plumbing evidence, not
     execution-quality evidence.

   State plainly that none of this establishes real-market performance, which would need
   real fills or historical order-book data, and that neither is in the portfolio. This is
   also the answer to the obvious interview question, and it is better volunteered than
   extracted.

7. **Two house notes, from M2 and M1a. Place them wherever Anvil/Crucible keep theirs.**
   - **Thread count is a reproducibility axis.** The same seed scored 0.165 vs 0.066 of the
     TWAP gap on four threads vs eight — torch's reduction order changes the trained weights.
     Every RL result in the portfolio that does not pin thread count is suspect by this
     argument. Pin it in committed config, and pin `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
     before torch imports.
   - **Below n ≈ 10, draw every trace.** An IQR band cannot display extremes at small n, so
     a band plot at n = 5 is structurally misleading rather than merely incomplete.

## Pre-stated numbers (invariant 3 — loosen only by amending this brief before work)

| Item | Value |
| --- | --- |
| Task 1 λ | 10^−3.5, M2's committed point |
| Task 1 acceptance | median gap ≤ **0.002** — against M2's CV result (0.0002), not against ε |
| Task 1 failure signal | median in the sampled regime's neighbourhood (~0.098) ⇒ pairing is not cancelling; stop and report |
| Task 1 seeds | 10, `train` pool; eval on `eval` pool |
| Reward-variance evidence | realised per-update reward variance reported against the sampled regime's |
| Action identity | pair halves bitwise-identical action sequences — permanent test |
| Shock negation | mirror draws elementwise-exact negation |
| Seeds per λ, sweep | **10**, not negotiable — M2's IQR 0.081 against a 0.05 tolerance |
| λ grid | **decided in task 3** from measured cost; must include 10^−3.5; thin the grid, never the seeds |
| Per-λ tolerance | ε = 5 % of that λ's `(J_twap − J_optimal)/J_optimal`, median across seeds; per-seed floor 10 % |
| Red flag (hard fail) | any seed, any λ, with `J_agent < J_optimal − 1e-9·\|J_optimal\|` |
| Trajectory band | derived per λ from `‖δ‖₂ ≤ √(2·ΔU/λ_min(H))`; report implied bound beside observed |
| Frontier axes | expected cost vs **excess over the `σ_bin²X²` floor**; both defined in the caption |
| Torch threads | 8, pinned in committed config, physical cores — changing it changes results, not just speed |
| Concurrency | **serial**. Two concurrent 8-thread sweeps truncate seeds (M2's first discarded run) |
| Sweep wall-clock ceiling | ~24 h; over that, thin the grid and re-record |
| Artifact size | per the ROADMAP policy M2 pre-stated; confirm the sweep's projected size against it in task 3 |
| Suite impact | `make test` ≤ 3 min; sweeps behind a marker |

## Definition of done

- [ ] Task 1 green: median ≤ 0.002 on 10 seeds, with reward-variance evidence, not outcome
      inference.
- [ ] Action-identity and shock-negation tests permanent and green.
- [ ] Measured pairing multiplier reported; any update-budget cut recorded as an amendment
      **before** the sweep.
- [ ] Grid, seeds and wall-clock estimate recorded here before the sweep starts.
- [ ] Sweep complete; per-λ median + IQR; three baselines through the identical grader at
      every point.
- [ ] Red-flag test green on every seed at every λ.
- [ ] Frontier figure on excess-over-floor axes, every seed traced, config hash + git rev.
- [ ] README ladder written.
- [ ] House notes placed.
- [ ] Clean clone through the documented interface: `make help`, `make test`, `make
      reference`, sweep configs dry-run, figure redraws byte-identical.
- [ ] Artifacts stamped `git_dirty: false` at a committed rev.
- [ ] `ROADMAP.md` M3 row flipped; anything structural → §9.

## Out of scope (resist)

Phase-2 dynamics — the 0.6 power law stays out of `temper/env/` until M4; observation
enrichment of any kind, which breaks the action-identity property task 1 depends on; any
change to `temper/env/`; hyperparameter search beyond the update-budget question in task 2;
the Anvil wire.

## Session notes

- Task 1 is a gate, not a formality. If the median lands near 0.098, the pairing argument is
  wrong and that is the milestone's finding — report it and stop. The fallback is the control
  variate with the weakened claim carried explicitly, and that is a decision for the next
  session, not a rescue to attempt at 3 a.m.
- The action-identity test is the load-bearing assumption made checkable. It holds only
  because the observation carries no price. When Phase 2 enriches observations, this test
  goes red and that is correct — it will be the signal that the pairing's exactness has
  lapsed, which is information M4 needs.
- M2's pattern of preserving pre-run reasoning verbatim with retractions beneath it is house
  practice now. Keep it. Task 2's update-budget decision is exactly the shape of thing that
  benefits.
- Serial scheduling is not a preference. It is the finding from M2's first discarded run.
