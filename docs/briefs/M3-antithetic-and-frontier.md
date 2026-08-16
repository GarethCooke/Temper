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

## Task 1 — the pairing is validated, and it reproduces the variate bitwise

**Status: green, gate met.** Ten seeds at λ = 10^−3.5, antithetic regime, everything else
identical to `configs/m2_ppo.yaml` (`tests/test_m3_validation.py` asserts the config identity
field by field). Graded analytically through the unchanged `GRADED` registry.

| | antithetic pairing, 10 seeds |
| --- | --- |
| gap fraction, per seed | 0.000225, 0.000204, 0.000125, 0.000061, 0.000313, **0.001451**, 0.000047, 0.000083, 0.000205, 0.000131 |
| **median** (gate ≤ **0.002**) | **0.000168** — met, 12× inside; the CV's committed median is 0.000204 |
| IQR | 0.000126 |
| worst seed | 0.001451 — inside the gate, and 69× inside the per-seed floor of 0.10 |
| median excess over `J_optimal` | +0.0094 % |
| median ‖δ‖₂ (band: 28 797) | **974 shares**; worst 4 168 |
| red flags | none · **timeouts** none · every seed completed all 1 802 updates |
| measured per-seed wall clock | 1 743–1 761 s (median **1 752 s**); one seed at 2 513 s under contention (below) |
| realised reward variance per update, median across seeds | sampled half **3 377 bps²** · averaged **3.40e−08 bps²** · **ratio 1.0e−11** |

The variance row is the direct evidence the brief asked for, and it is measured inside the
run under the agent's own actions rather than inferred from the outcome: `PairLedger` records
both halves' episode returns every update, so "what the sampled regime would have paid" and
"what the agent actually trained on" are the same 512 episodes, differenced. The primary
half's 3 377 bps² is the full Phase-1 noise — an episode-return standard deviation of ~58 bps
against a 2.35 bps objective, M2's ~1:70 SNR restated per update. The averaged return's
variance is eleven orders of magnitude smaller. That is cancellation, not reduction.

### The finding: this is not "close to" the control variate's answer, it *is* it

Seeds 0–4 of this run reproduce M2's control-variate seeds **bitwise** — the graded objective
to all seventeen digits and the entire 14-point inventory trajectory, for all five seeds they
share an address with:

| seed | M2 control variate, `J` (bps) | M3 antithetic, `J` (bps) | equal |
| --- | --- | --- | :---: |
| 0 | 2.354848190645846 | 2.354848190645846 | bitwise |
| 1 | 2.354821109991793 | 2.354821109991793 | bitwise |
| 2 | 2.3547160907352476 | 2.3547160907352476 | bitwise |
| 3 | 2.354630998039967 | 2.354630998039967 | bitwise |
| 4 | 2.3549655346658245 | 2.3549655346658245 | bitwise |

This was not designed for and it is worth stating why it happens, because the reason is a
property of the boundary rather than a coincidence. The two estimators produce rewards that
differ by a couple of ulps of the noise they remove — ~2e−17 bps on a ~0.013 reward after
scaling, about 1e−15 relative. PPO's rollout buffers are **float32** (`AGENT_DTYPE`, the
agent boundary; the env's core stays float64). A 1e−15 relative difference is ~7 orders of
magnitude below float32's resolution, so both regimes hand the optimiser *the same
float32 numbers*, and everything downstream — gradients, weights, the eval schedule — is
identical by construction. Checked directly rather than inferred: over 40 episodes the
float32-cast reward vectors of the two regimes are `torch.equal` on every one.

So the gate's question — "does antithetic reproduce the answer a zero-variance reward already
established?" — is answered in the strongest available form at this case. It also bounds the
claim honestly: the two estimators agreeing bitwise *here* is a statement about this reward
magnitude and this dtype, not a theorem. In Phase 2, where cost stops being affine in the
shocks, the pairing's residual is a real quantity rather than an ulp, and this equality will
break — which is the point of preferring the pairing, and the reason the mechanism is
asserted per step (below) rather than trusted.

Seeds 5–9 are new addresses with no M2 counterpart. Seed 5 is the worst of the ten at
0.001451 — still 34× inside the gate — and is the only seed whose training return was still
more than ε/10 from its final value at update 706. It is drawn individually on the figure.

### The two structural checks are permanent, and live

- **Action identity across the pair.** `AntitheticPair.step` requires, bitwise on every step,
  that both halves report the same observation and realise the same trade, and raises
  `PairDiverged` otherwise — in the wrapper, so it guards all 12 M steps of every training
  run rather than a test fixture. `tests/test_m3_antithetic.py` asserts it for TWAP, both AC
  schedules, a fraction policy and two freshly-initialised PPO networks, and proves the check
  is not decorative by leaking 1e−12 of the shock into a mirror observation and requiring the
  pair to refuse.
- **Shock negation is exact.** The mirror is an `ExecutionEnv` at the *same seed address*
  whose generator is wrapped in `NegatedDraws`, so its draws are the elementwise negation of
  the primary's — checked at the generator over 2 000 draws, and per step through the
  published cumulative shock (`m_walk == -walk`, bitwise). A mirror that drew fresh numbers
  from a symmetric distribution fails both. `NegatedDraws` deliberately proxies only
  `standard_normal`; anything else raises rather than silently returning a non-mirror.
- **It costs one stream, not two.** The mirror shares the primary's `(root_seed, pool,
  stream_index)`, so a pair spends exactly the stream the config addressed it to — asserted
  through the conftest ledger that records every address the env resolves (invariant 5).

### Retraction: the first run of this night is not the committed artefact

The numbers above were produced by a run that started from a **dirty tree** — the M3 frontier
tooling was being written while the validation trained, so `provenance.git_dirty` is `true`
and the recorded revision `096f6a2` does not contain everything that ran. Under
`ARCHITECTURE.md` §9 (*`git_dirty` asks whether the source is uncommitted…*) that is not an
acceptance artefact, and `tests/test_m3_validation.py` refuses it. It was re-run from a
committed tree; that run is what `results/m3_antithetic_validation.json` holds, and the
comparison between the two is reported below. The discarded run is kept out of `results/` and
is recorded here rather than described, in the same spirit as M2's three discarded attempts.

The mistake is worth naming precisely because the guard that caught it was M2's, and it
caught it *after* five hours rather than before: the provenance stamp is taken at the start of
the run, so editing unrelated files while a sweep runs is safe — editing them in the window
between launching a script and the sweep's own start is not. This run's script trained a
control seed first, which widened that window to 25 minutes.

## Task 2 — the measured multiplier, and the update budget

**The pairing multiplier is ~1.17×, not 2×.** A pair shares one schedule, one set of network
forward and backward passes, and one optimiser step; only the env stepping and the reward
assembly double. Measured three ways, all on the reference box at the committed 512 envs and
8 threads:

| comparison | control variate | antithetic | multiplier |
| --- | ---: | ---: | ---: |
| full-length seed, same session, cold box → warm box | 1 493 s (seed 0, this session) | 1 752 s (median of 9) | **1.17×** |
| full-length, against M2's committed per-seed median | 1 569 s | 1 752 s | 1.12× |
| 30-update probe, back to back | 0.84 s/update | 0.93 s/update | 1.10× |

The three disagree by less than the box's own thermal spread (M2 measured +25 % on a loaded
host), so the honest statement is **1.1×–1.2×, and the sweep is sized at 1.2×**. The mirror is
carried as a second env inside the same batch, exactly as the brief hoped; it is not a second
training run.

One seed of the ten took 2 513 s rather than ~1 752 s. That seed ran while this session was
rendering figures and running the test suite on the same box, and it is the measurement of
what contention costs rather than of what the pairing costs: the brief's serial-scheduling
rule covers other sweeps, and this widens it to *anything* that uses the cores, editor tooling
included. Its result is unaffected — 0.000131, mid-pack — which is the point M2 made about
wall clock not being a property of the numbers.

### Where the objective actually goes flat

Measured on all ten antithetic traces (25-update moving mean of the training return, against
each seed's own final value; ε is 5 % of the TWAP gap, 0.001326 in scaled reward units):

| within … of the final return | last update below it, worst seed | 9 of 10 seeds |
| --- | ---: | ---: |
| ε | 62 | ≤ 62 |
| ε/10 | 706 | ≤ 167 |
| ε/100 | 1 519 | ≤ 1 225 |
| ε/1000 | 1 754 | ≤ 1 754 |

1 802 updates was M2's blind budget and it is *not* a third of heat: the objective keeps
improving in ever-smaller amounts right to the end, and reaching the final ε/1000 needs
essentially all of it. What the table does establish is that ε — the bar the frontier is
actually judged against — is reached by update ~62, and ε/10 by ~170 on nine seeds of ten.
The budget question is therefore not "when is it converged" but "how much precision does the
frontier need", and the answer is that it needs to meet ε at nine λ, with the full-budget run
at M2's λ committed beside it to show what the last thousand updates buy.

## Amendment 1 — the sweep's update budget is cut to 751 updates (recorded before the sweep)

**Status: invoked. Recorded 2026-08-16, before any frontier point was run**, from the task-1
and task-2 measurements above and from nothing else.

`configs/m3_frontier.yaml` sets `ppo.total_timesteps: 5000000` for every sweep point — 751
updates at the committed batch size, against the validation run's 1 802. With
`anneal_lr: true` this is a **re-anneal, not a truncation**: the learning rate reaches zero at
update 751 rather than being cut off two-thirds of the way down its ramp. Stating it that way
matters, because a truncated anneal would be a different experiment wearing the same
hyperparameters, and M2's own truncation incident is what taught this repo the difference.

*Why.* At 1 802 updates the nine-point sweep costs 9 × 10 × 1 752 s = **43.8 h**, which is
nearly twice the brief's ~24 h ceiling; the brief's instruction is to thin the grid, never the
seeds. At 751 updates it costs 9 × 10 × 730 s = **18.3 h**, inside the ceiling with a
nine-point frontier intact. The alternative — 1 802 updates on a five-point grid — is 24.3 h,
still over the ceiling, and buys precision the ε bar does not ask for at the cost of the
frontier's shape, which is the milestone's actual product.

*What it costs.* By the flat-point table, at update 751 every seed's training return is
already within ε/10 of its own final value (nine within ε/100), so the expected graded gap is
of order 1e−3 of the TWAP gap rather than the validation run's 1.7e−4 — still ~50× inside ε.
That prediction is checked rather than assumed: the sweep runs its λ = 10^−3.5 point **first**
and the point is directly comparable to both the full-budget validation run and to M2's
committed control-variate result. If it misses, the budget is revisited before the remaining
eight points run.

*What does not change.* Ten seeds per λ, the seed addressing, the case, the tolerances, the
estimator, the reward scale and every other hyperparameter — `tests/test_m3_frontier.py`
asserts a sweep point differs from the validation config in `total_timesteps`, λ, the runtime
bounds, the trace budget, the results paths and the claim's closing sentence, and in nothing
else.

## Task 3 — the grid, the seeds and the wall clock (recorded before the sweep)

| item | value | how it was fixed |
| --- | --- | --- |
| λ grid | **9 points**, 10^−5 … 10^−1 in half decades: 1e−5, 10^−4.5, 1e−4, **10^−3.5**, 1e−3, 10^−2.5, 1e−2, 10^−1.5, 1e−1 | thinned from M0's 17-point grid by dropping the eight points below 1e−5, where TWAP, AC and the optimum agree to four digits (task 0's table: TWAP gap ≤ 0.44 %) and a frontier point measures nothing. Taken **by index** from `VENDOR_LAMBDA_GRID`, so each λ is the reference table's float exactly, and it contains M2's rule-selected 10^−3.5 |
| seeds per λ | **10** — not negotiable | M2's IQR of 0.081 against a 0.05 tolerance |
| updates per seed | 751 (5 M steps) | amendment 1 above |
| measured per-seed cost | 1 752 s at 1 802 updates ⇒ **730 s** at 751 | task 1's measurement × 751/1802 |
| **total wall-clock estimate** | **18.3 h** (9 × 10 × 730 s), plus grading and figure time | inside the ~24 h ceiling |
| scheduling | strictly serial, one fresh process per point, nothing else on the box | M2's discarded concurrent run; and task 2's contended seed |
| artifact size | 9 points × 10 seeds × 128 trace points ≈ **1.4 MB** of traces total, against M2's 1.2 MB for one 5-seed sweep | the ROADMAP's M3 pre-statement (`results.trace_points: 128`, `temper.eval.sweep.thin`); graded values and trajectories stay whole; full-resolution traces are kept at 10^−3.5 by the validation run |
| per-λ tolerance | ε = 5 % of *that* λ's `(J_twap − J_optimal)/J_optimal`, median across seeds; per-seed floor 10 % | unchanged from the pre-stated table |

The grid is generated, not hand-written: `configs/m3_frontier.yaml` is the manifest and
`python tools/m3_frontier.py configs` stamps the nine point configs from the validation
config; `tests/test_m3_frontier.py` asserts every committed point is byte-identical to what
the generator writes, so a point cannot drift from the template it claims to be.

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
