# M4b — Stochastic liquidity

**Track:** agentic · **Size:** one evening — the oracle is minutes, the differential ~10 min,
ten seeds ~2–3 h, plus ~40 min of bitwise regression across two worlds · **Reads first:**
`ARCHITECTURE.md` §4 including *Amended by M4a*, §5's *Amended by M4a*, invariants 3, 4, 5,
6, 7; then four §9 entries by title — *A deterministic policy on a price-free observation is
graded analytically, not by Monte Carlo* (M4b is the first milestone that cannot obey it as
written), *The power law's break is in a shock-free term, so the noise identity and the
antithetic pairing survive it exactly* (which names M4b as the place exactness ends, and is
half right), *A metric grades the world that charges it*, and *The tolerance's denominator is
the available advantage, not the TWAP gap, wherever the closed form is the thing being
beaten*. Also the house note *No code path may be reachable only at the end of a long run* — M4b writes a new artefact shape and will re-earn
that lesson otherwise. (It was titled *The artefact writer is tested on
fabricated data, not on the run* when this brief was written; M4b is why it is
not any more.)

## Objective

Liquidity becomes a second, independent noise source, and the agent gets to see it. This is
the first advantage in the project that **no static schedule can capture at all**: under the
power law alone (M4a) the closed form was merely solving the wrong problem, and a different
fixed schedule fixed it. Here the best possible fixed schedule is provably beaten by a policy
that reacts, and the milestone measures by how much and how much of it the agent gets.

Three things break together and that is why they are here rather than in M4a: analytic
grading (the schedule is no longer open-loop), the exactness of the antithetic pairing (a
second noise source), and the "no Monte-Carlo interval" property (M4a's dispersion was across
seeds only). The brief's job is to replace each with something equally checkable rather than
to weaken all three at once and call the result a result.

## Context — five facts that fix this milestone's shape

### 1. The invented model, owned as invented

FrontierView has no liquidity process, so §7's "vendored, not invented" cover does not extend
to M4b and the README must not imply it does. The model is therefore the smallest one that
makes the question well-posed: a **per-bin i.i.d. lognormal multiplier** `L_k` on `v_hourly`,
`E[L] = 1`, one parameter `σ_L`. Participation becomes `p_k = n_k / (dt · v_hourly · L_k)`;
nothing else in the world changes.

I.i.d. is a choice with three consequences worth stating rather than discovering:

- **All of the measured advantage is adaptivity.** The best static schedule already absorbs
  the level shift `E[L^−0.6] = 1.1275`, by re-solving M4a's Newton system at an inflated
  coefficient. Nothing about the *shape* of the liquidity law is capturable statically, so
  numerator and denominator are clean. A U-shaped intraday profile would not have this
  property — the profile is deterministic and a static schedule eats it — which is why the
  realistic-looking model is the harder one to report honestly, and why it is backlog.
- **`(k, x_k, L_k)` is a sufficient statistic**, so the dynamic-programming optimum over that
  state *is* the optimum over all adapted policies. Past liquidity carries no information
  about future liquidity. That makes an observation-minimality guard available (task 1(e)),
  and it means "the agent could have done better with a richer observation" is answerable
  rather than arguable.
- **`σ_L` is invented, so the result is a curve, not a point.** Train at one `σ_L`; report the
  oracle's value-of-sight at three. A single invented parameter with a single number beside it
  reads as calibration, and it is not.

### 2. The size of the thing, and how it splits

At λ = 10^−3.5 (M4a's rule-selected point), σ_L = 0.5, per the oracle:

Every static rung below is a **closed form**, not a simulation — `E[cost]` of a fixed
schedule under i.i.d. liquidity is `A·E[L^−β]·Σ w^{1+β} + λB·Σ r² + const`, and
`E[L^−β] = exp(σ_L²·β(1+β)/2)` exactly. Only `J_DP` is numerical.

| quantity | bps | reading |
| --- | ---: | --- |
| M4a's power-law schedule (knows no liquidity at all) | 2.49895 | where M4a leaves off |
| `J_static*` — best fixed schedule that knows the liquidity *law* | 2.49661 | the denominator's top |
| `J_DP` — adaptive optimum (dynamic programming) | 2.43449 | the denominator's floor |
| **adaptive advantage** `J_static* − J_DP` | **0.06212** | **2.55 % of the objective** |
| the level shift `J_M4a − J_static*` | 0.00234 | **3.8 % of the naive gap, and not the agent's** |

The effect is **1.7× M4a's** 0.0367 bps, which is the good news. The trap is the last row: an
agent measured against *M4a's* schedule appears to gain 0.06446 bps, and 0.00234 of that is a
constant any static solver picks up for free. **The denominator is `J_static* − J_DP`.**
Report the level shift separately, as its own line, every time.

Value of sight against the invented parameter:

| σ_L | `E[L^−0.6]` | `J_static*` | `J_DP` | advantage | % of objective | level shift / advantage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 1.03045 | 2.41072 | 2.39598 | 0.01474 | 0.62 % | 0.9 % |
| **0.50** | **1.12750** | **2.49661** | **2.43449** | **0.06212** | **2.55 %** | **3.8 %** |
| 0.75 | 1.30996 | 2.65172 | 2.49949 | 0.15223 | 6.09 % | 8.5 % |

The last column is why task 0's level-shift gate is not a formality: the constant grows faster
in σ_L than the adaptivity does, and by σ_L = 0.75 it is 8.5 % of the advantage against a gate
of 10 %.

### 3. Grading stops being exact and becomes cheap anyway

The observation carries `L_k`, so the schedule is closed-loop and
`deterministic_schedule`'s bitwise check cannot hold as written. But the price shocks still
enter cost only through M1a's affine term, and the policy still never sees a price — so
**`E[cost | L-path]` is a closed form**: `cost_moments` evaluated at the realised trajectory
and the realised per-bin liquidity. Grade by averaging that over sampled liquidity paths.
There is **no price sampling anywhere**, and the residual error is liquidity dispersion only:

- SD of `E[cost | L]` across paths: **0.179 bps** — against the ~95 bps per-episode SD that
  made sampled grading hopeless in M2, a factor of 530.
- Common random numbers across policies cut the *variance* a further **8.6×** (2.9× in
  standard deviation): paired SD of *(reference − agent)* is **0.0612 bps**.
- So a 95 % half-width of **1.36 % of the effect at M = 20 000 paired draws** (4.32 % at
  M = 2 000). Seconds of oracle work.

The successor assertion to `deterministic_schedule` is one axis wider and just as mechanical:
**hold the liquidity stream fixed, roll out on two unrelated price streams, require the
trajectories bitwise equal.** That is still "the policy is price-free", which is still exactly
what makes the conditional expectation legitimate — and it still fails loudly the moment
price reaches the observation.

### 4. The pairing survives, because the pair shares its liquidity

§9's M4a entry names "a second, independent noise source" as what ends the pairing's
exactness. That is half right, and the half that is wrong is the useful half. If the mirror
is handed **the same liquidity stream** and only the price draws are negated, then both halves
see identical observations, take identical actions, and their price noise cancels *exactly* —
because cost is still affine in the price draws given `(x, L)`. The action-identity assertion
stays green. What the pair no longer does is remove the **liquidity** noise, which is the
reward variance the agent must now train through.

That variance is affordable: per-update standard error at 512 envs is **0.0079 bps, 12.7 % of
the effect**, against M4a's exactly zero and M2's sampled-reward regime that was a lottery at
1:70. Do **not** antithetically mirror the liquidity draws as well — `u → 1−u` on the mirror's
uniform would make the two halves disagree about `L`, hence about their actions, and would
trade the pairing's one exact property for a partial second one.

### 5. The reference is converged and bracketed, not certified

M4a could certify its optimum: convex, Cholesky PD, KKT residual 1.2e-15. A stochastic DP has
no such certificate, and pretending otherwise would be the first dishonest number in the repo.
What is available is stronger than "it converged", and it is two-sided:

- **A feasible upper bound.** The DP's own greedy policy, simulated, is a real policy, so its
  Monte-Carlo mean is an unbiased estimate of an attainable value — hence an upper bound on
  the optimum. Measured: 2.434684 against the DP's 2.434492, a gap of **0.31 % of the
  advantage** when the stage problem is solved by 1-D convex search against the interpolated
  value function. With naive nearest-quadrature-node action snapping the same gap is **3.7 %**
  — an order of magnitude worse and entirely an artefact of the action map. Interpolate.
- **A rigorous lower bound, from M4a's own solver.** Give the optimiser the whole liquidity
  path in advance and solve the deterministic convex problem per path — M4a's Newton system
  with per-bin coefficients `A·L_k^{−0.6}`, batched. More information cannot cost more, so the
  average is a **perfect-information lower bound** on the adaptive optimum. Measured
  2.42938 ± 0.00246, bracketing the optimum to **8.5 % of the advantage**.

The bracket is too loose to grade against and exactly tight enough for the thing that matters:
**the red-flag test becomes rigorous.** No adapted policy can beat perfect information, so an
agent below the clairvoyant bound is a defect with a proof, not a discovery — where M4a's red
flag rested on an algebraic certificate, M4b's rests on a relaxation. Everything between the
clairvoyant bound and the DP value is numerical uncertainty in the reference, and is reported
as such.

## Tasks

### 0. The liquidity reference table, and the three worlds' λ — *gate*

Extend M4a's table to the liquidity world: at every λ on `VENDOR_LAMBDA_GRID`, `J_static*`,
`J_DP`, the clairvoyant bound, the adaptive advantage, the level shift, and the max bin
fraction of the DP's *mean* schedule. Oracle only.

`Experiment.verify_lambda_rule_agrees_across_worlds` currently loops `ENCODINGS`. Liquidity is
not a new encoding (context §3 — the functional is unchanged, the *market* is random), so
decide and record how the rule is applied here: predicted, it is M4a's power-law table with
the coefficient inflated by `E[L^−0.6]`, which is a monotone rescaling and should select
10^−3.5 again. **Assert the selection, do not assume it.**

**Gate conditions:**

- the rule selects the same λ as M4a and M3, so all three milestones' points are comparable;
- the adaptive advantage at that λ is ≥ 1 % of `J_DP`;
- the level shift is ≤ 10 % of the adaptive advantage — if the constant dominates, the
  headline is a re-solve and not adaptivity, and the milestone is re-shaped before training;
- the clairvoyant bracket is ≤ 15 % of the advantage, so the red-flag test has teeth.

### 1. The adaptive oracle

`temper/oracle/` gains the liquidity world's reference, in the shape M4a established — solved,
independently checked, no scipy.

- **(a) The DP.** Backward value iteration on an inventory grid with Gauss–Hermite quadrature
  over `log L`. Report grid convergence: predicted 2.434715 / 2.434544 / 2.434502 / 2.434492
  at 201 / 401 / 801 / 1601 points — second-order, so Richardson-extrapolate and state the
  residual. Quadrature converges by **7 nodes** (2.434503 against 2.434502 at 15); use more
  and say why not fewer.
- **(b) The σ_L → 0 limit is the differential check.** At σ_L = 0 the DP must return M4a's
  certified `power_law_optimum` value, 2.383215 bps. Predicted 2.383218 at 1601 grid points —
  agreement to 3e-6 bps, which is grid discretisation and not a disagreement, and which also
  calibrates what "converged" is worth here: 5e-5 of the advantage. This is the single most
  valuable test in the task, because it ties the new machinery to a number that *was*
  certified.
- **(c) The feasible upper bound.** Simulate the DP's greedy policy with the stage problem
  solved by 1-D convex search against the interpolated value function, not by snapping to a
  quadrature node. Pre-stated band: the simulated value must sit within **2 % of the
  advantage** of the DP value. Predicted 0.31 %; the snapping implementation misses at 3.7 %,
  which is the failure this bar exists to catch.
- **(d) The clairvoyant lower bound.** M4a's `optimum_for_charge` generalised to per-bin
  coefficients, batched over paths. Report mean and CI. Assert `lower ≤ J_DP ≤ upper` with the
  CIs carried, and commit the bracket width as a fraction of the advantage.
- **(e) Sufficiency, checked not asserted.** Re-run the DP on an augmented state carrying
  `L_{k−1}` as well and require the same value to the grid tolerance. I.i.d. liquidity makes
  the extra coordinate uninformative; a value that *improves* means the process implementation
  is not i.i.d., which is a bug in the env and not a discovery about markets.

### 2. The second seam, and the second stream

**The world.** `temper/env/liquidity.py`, mirroring `impact.py` exactly:
`DeterministicLiquidity` (returns 1.0 — the default, everywhere) and
`LognormalLiquidity(sigma_log)`, each exposing its closed-form moments (`mean_multiplier`,
`inverse_power_moment(exponent)`) because the analytic reference needs them and a
distribution moment re-derived at three call sites is the same pattern that put a
`Market`-style derived-quantities object on *FrontierView's* backlog for `v_hourly`, `dt` and
`sigma_bin` — an analogy, not that item. `ExecutionEnv` takes one; **a config must name it** (`world.liquidity`), by the same
rule and the same `tests/test_repo_invariants.py` check that stops a config inheriting a
Phase-2 encoding by omission. The bound temporary charge gains a liquidity argument defaulting
to 1.0.

**Liquidity draws from their own stream.** Non-negotiable and easy to get wrong: if the
liquidity variate comes out of the price generator, every downstream price draw shifts and
Phase 1 and M4a stop reproducing. A separate seed address, `POOLS` extended the way M4a
extended it, and the acceptance is arithmetic — **one M3 seed and one M4a seed retrained,
bitwise identical**, plus every committed figure redrawing byte-identically.

**The observation grows to three.** `(time left, inventory left, log L_k)`, with `L_k` drawn
and published *before* bin `k` executes — so `reset` draws `L_0` and each `step` draws the
next bin's multiplier after executing its own. Record the ordering in the docstring the way
M1a's shock ordering is recorded; it is invisible in code and load-bearing for the DP's state
definition.

**Every env the estimator builds gets both models.** This is M4a's §9 lesson verbatim — *when
a per-episode property is injected into the env, every env the estimator constructs has to be
handed it* — and M4b hands out two. `mirror_of` must pass the impact model, the liquidity
model **and the same liquidity stream**; the pair's per-step assertions gain a third: the two
halves saw the same `L`. The mirror charging a different liquidity path would look exactly
like the M4a bug looked: rewards still rewards, schedules still plausible, estimator silently
not the one the config names.

### 3. Conditional grading, and the assertion that licenses it

The graded route becomes: roll the policy out on a liquidity path, take the realised
trajectory *and* the realised path, evaluate `cost_moments` at the realised participations —
that is `E[cost | L]` exactly — and average over paths from the `eval` pool. Common random
numbers across every policy on the chart; report the **paired** difference and its bootstrap
CI, never the difference of two independently-sampled levels.

`cost_moments` gains a per-bin liquidity argument defaulting to `None`; the encoding stays
`power_law` and *A metric grades the world that charges it* is untouched. Pin that the
`liquidity=None` path is bit-identical to today, so no M4a or earlier number moves.

`deterministic_schedule` generalises rather than retires: same liquidity stream, two unrelated
price streams, trajectories bitwise equal. Keep the name, keep the exception
(`ScheduleNotDeterministic`), widen the docstring. A policy that fails it has still not been
scored — the conditional expectation is only valid because the price never entered the
decision.

### 4. The differential — invariant 6, two new models

Two expectation tests, because there are two new pieces:

- **The process.** From the env's own draws: `E[L] = 1`, the variance, and `E[L^−0.6]` against
  the closed form; per-bin independence (lag-1 autocorrelation inside its CI); the draws
  reproducible from the seed address.
- **The world.** Fixed schedules — TWAP, `ac`, `tangent`, M4a's `optimal`, `static*` — through
  the stochastic-liquidity env at M1's tiers, with the analytic reference
  `A·E[L^−β]·Σ w^{1+β} + …`. Keep the `step_count` assertion.

**State plainly what did *not* change.** The frozen objective still penalises *price*
shortfall variance: `V = σ_bin²·Σ(x_k/X)²`, untouched, because liquidity dispersion enters
`E[cost]` through Jensen and not `λV`. So invariant 7 holds with no amendment — one
functional, still encoded once — and the realised-cost variance the differential measures now
has two sources while the graded `V` has one. Write that down in the brief's own results
section; it is precisely the kind of distinction that drifts silently. Penalising total cost
variance instead is a different functional, a §9 amendment and a loss of comparability with
M0–M4a: backlog, not tonight.

### 5. The training point, and the control that makes it mean something

Ten seeds at the rule-selected λ, σ_L = 0.5, `train` pool, antithetic regime with common
liquidity, evaluated on `eval`-pool liquidity streams — disjoint by construction, which is
invariant 5 doing the out-of-sample work the ROADMAP row asks for.

Headline: **capture fraction** `c = (J_static* − J_agent) / (J_static* − J_DP)`, median, IQR,
per seed — with the absolute excess over `J_DP` in bps beside it everywhere, per §9's
denominator entry, and the level shift reported as its own line so nobody credits the agent
with a constant.

**The liquidity-shuffled control is part of the milestone, not an extra.** Re-grade each
trained policy with the observed `L` drawn independently of the `L` that is charged. If the
advantage survives, the agent is not using the signal and the headline is measuring something
else. Pre-stated: shuffled capture fraction ≤ 0.15, and the gap between real and shuffled is
the actual claim. This is M5's overfit-check pattern arriving one milestone early because it
costs a re-grade rather than a re-train.

### 6. Figure, README, and the σ_L curve

`results/m4b_adaptivity.*`: the three rungs (M4a's schedule → `J_static*` → `J_DP`) with the
clairvoyant bound drawn as the floor and all ten seeds individually (*Below n ≈ 10, draw every
trace*), plus the value-of-sight curve against σ_L with the trained point marked. The caption
names the denominator, states that σ_L is invented, and gives the bracket width.

README's Phase-2 rung gains its second half: the advantage is now against the best *static*
schedule in a world with a noise source the schedule cannot see; it is worth ~2.6 % of the
objective at σ_L = 0.5 and that parameter is Temper's own, not FrontierView's. Say that the
liquidity model is invented in the same sentence that reports what it bought.

## Pre-stated numbers (invariant 3 — loosen only by amending this brief before work starts)

| Item | Value |
| --- | --- |
| Case | AAPL, X = 100 000, T = 6.5 h, N = 13 — unchanged since M2 |
| World | M4a's power law, plus i.i.d. lognormal liquidity on `v_hourly`, `E[L] = 1` |
| σ_L | **0.5** trained; 0.25 / 0.5 / 0.75 reported from the oracle. Invented, and labelled so |
| λ | the rule-selected point; required to agree with M4a's and M3's |
| Reference | `J_DP`, converged and bracketed — **not** certified, and the report says which |
| Denominator | `J_static* − J_DP` (the *adaptive* advantage). **Not** `J_M4a − J_DP`: 3.8 % of that is a level shift any static solver gets free |
| Tolerance, median | excess over `J_DP` ≤ **10 %** of the adaptive advantage ⇒ `c ≥ 0.90`. Loosened from M4a's 5 % **for a stated reason**: the reward now carries uncancellable liquidity noise at 12.7 % of the effect per update against M4a's zero, and grading carries a 1.36 % measurement floor. The bar sits 7.4× above that floor |
| Tolerance, per seed | ≤ **25 %** ⇒ `c ≥ 0.75` per seed |
| Red flag (hard fail) | `J_agent < J_clairvoyant − 1.96·SE`. Rigorous: no adapted policy beats perfect information |
| Red flag (soft, investigate) | `J_agent < J_DP` by more than the DP's own numerical uncertainty — possible without being a defect, because the reference is numerical. Report, do not auto-fail |
| Shuffled control | capture fraction ≤ **0.15** on the liquidity-shuffled re-grade |
| Eval draws | **M = 20 000** paired liquidity paths, common random numbers across every policy; report the achieved half-width, predicted 1.36 % of the effect |
| DP grid | ≥ 1601 inventory points, Richardson residual reported; ≥ 7 quadrature nodes |
| DP action map | interpolated, 1-D convex stage solve. Simulated-vs-DP gap ≤ **2 %** of the advantage (predicted 0.31 %; snapping gives 3.7 %) |
| Bracket | clairvoyant lower ≤ `J_DP` ≤ feasible upper, committed as a fraction of the advantage; predicted 8.5 % |
| σ_L → 0 | DP returns M4a's certified `power_law_optimum` value 2.383215 bps; predicted 2.383218 at 1601 points, i.e. 5e-5 of the advantage |
| Pairing | liquidity **common** across the pair, price negated. Action identity must stay green; a third per-step assertion covers the shared `L` |
| Objective | unchanged. `V` is price-shortfall variance only; liquidity enters `E[cost]`. No §9 amendment to invariant 7 |
| Regression | one M3 seed **and** one M4a seed bitwise identical; all committed figures byte-identical |
| Seeds | 10, `train`; eval on `eval`; liquidity on its own stream in both |
| Torch threads | 8, pinned (*Thread count is a reproducibility axis*) |
| Concurrency | serial |
| Dependencies | none new |
| Wall-clock ceiling | ~4.5 h including both regression seeds |
| Suite impact | `make test` ≤ 3 min; DP convergence and the deep differential behind markers |

## Definition of done

- [x] Liquidity reference table committed; all four task-0 gates recorded green, or the
      milestone re-shaped here with the reason.
- [x] DP oracle landed: grid and quadrature convergence reported, σ_L → 0 returning M4a's
      certified value, feasible upper bound inside its 2 % band, clairvoyant lower bound
      committed, `lower ≤ J_DP ≤ upper` asserted with CIs.
- [x] Sufficiency check green — augmented state does not improve the value.
- [x] Liquidity seam landed with its own seed stream; config must name its liquidity model;
      one `step` loop still.
- [x] **Bitwise regression across two worlds**: one M3 seed and one M4a seed identical; every
      committed figure byte-identical.
- [x] `mirror_of` hands over impact model, liquidity model and liquidity stream; the pair's
      third per-step assertion live and shown to be live.
- [x] Conditional grading landed; `deterministic_schedule` generalised and still refusing a
      price-bearing policy; `cost_moments(liquidity=None)` bit-identical to today.
- [x] Differential green: the process's moments and the world's E[cost] at M1's tiers, with
      `step_count` asserted; the "graded `V` is price-only" distinction written down.
- [x] Ten seeds; median and per-seed capture fraction against the pre-stated bars; absolute
      excess in bps beside every fraction; level shift reported separately.
- [x] Paired CI reported with its achieved half-width; red-flag test green on every seed.
- [x] Liquidity-shuffled control run and inside its bar.
- [x] `results/m4b_adaptivity.*` with every seed drawn and the σ_L curve; `git_dirty: false`.
- [x] New artefact keys covered by `tests/test_sweep_document.py` on fabricated data
      *before* the training run (the house note exists for exactly this).
- [x] README Phase-2 rung completed, naming the liquidity model as Temper's own.
- [x] Clean clone through the documented interface; `ROADMAP.md` M4b row flipped; structural
      findings → §9.

### §9 amendments this milestone is expected to yield

Predicted so the session recognises them, and records what it actually found instead:

1. **A liquidity-observing policy is graded by conditional expectation, not by sampling
   realised cost.** Successor to *A deterministic policy on a price-free observation is graded
   analytically, not by Monte Carlo*: the open-loop shortcut retires, the price-free assertion
   does not, and the interval that arrives is over liquidity alone.
2. **A second noise source gets its own stream, or the first one moves.** The seeding decision
   that keeps Phase 1 and M4a bitwise through a second seam.
3. **The antithetic pair holds liquidity common, so action identity survives a richer
   observation.** Narrows the M4a entry a second time: what ends action identity is an
   observation the two halves *disagree* about — a price-bearing one — not a richer one.
4. **A numerical reference is bracketed, not certified, and the red-flag test moves to the
   bound that is rigorous.** Possibly the most portable of the four.

## Out of scope (resist)

Alpha — M5, and the observation should grow once per milestone. Persistent or AR(1) liquidity,
and the U-shaped intraday profile: both backlog, and the U-shape specifically because its
deterministic part contaminates the adaptivity claim. Transient impact with decay. A λ sweep
in the liquidity world — the power-law frontier is already backlogged and this would be a
third. Penalising liquidity-driven cost variance in the objective. Hyperparameter search: the
configuration is M4a's, and if it fails to train through 12.7 %-per-update reward noise that
is the finding. New dependencies. The Anvil wire.

## Session notes

- **The gate that matters most is the level shift**, not the advantage. If `J_M4a − J_static*`
  is a large fraction of `J_static* − J_DP`, then most of what looks like adaptivity is a
  constant, and the milestone's headline has to be restated before any training rather than
  caveated afterwards. Predicted 3.8 % at the trained σ_L — comfortable, but it is 8.5 % at
  σ_L = 0.75, so check it rather than assuming the gate is slack. Compute both rungs from the
  closed form, not from a simulation: they have one, and two simulated levels differenced is
  how a 0.002 bps quantity turns into noise.
- **Interpolate the DP's action map.** The difference between snapping to a quadrature node
  and solving the stage problem properly is 3.7 % versus 0.31 % of the whole effect, which is
  the difference between a reference that can carry a 10 % tolerance and one that cannot.
- **The bracket is the honest form of "certified" here.** Do not describe `J_DP` as certified;
  M4a's word is earned and this one is not the same word. "Converged, and bracketed to 8.5 %
  of the advantage by a perfect-information relaxation" is both true and stronger-sounding
  than it looks, because the relaxation also bounds what any smarter agent could ever gain.
- **The numbers in this brief are predictions from a cloud container on unpinned numpy.**
  Nothing here is a committed artefact. Task 0 regenerates all of them on the box; a material
  disagreement is the first thing to understand and means this brief is wrong before the code
  is.
- **Two seams now default to Phase 1, and every env the estimator builds must be handed
  both.** M4a's mirror bug cost minutes because a per-step identity caught it. M4b doubles the
  surface for that class of defect and the same class of check is what will catch it — run the
  inherited-guarantee suite before training, not after.
- **The claim is narrower than it will be tempting to write.** It is "with a one-parameter
  invented liquidity process, seeing liquidity is worth 2.6 % of the objective and the agent
  captured most of it" — not "the agent adapts to real market liquidity". The invented
  parameter belongs in the same sentence as the result, every time it is stated.


---

# What happened

Written after the fact, in the shape M4a's brief records its own. Everything above
this line is the plan; everything below is the measurement.

## Task 0 — the gate, and the one decision the brief left open

**All four gates green, and every number the brief predicted reproduced on the
reference box.** The brief's own warning — that its numbers were predictions made
in a cloud container on unpinned numpy and none of them was a committed artefact —
turned out to be a warning about a risk that did not materialise.

| quantity | predicted | measured | gate |
| --- | ---: | ---: | --- |
| `J_M4a` re-priced | 2.49895 | 2.49895 | — |
| `J_static*` | 2.49661 | 2.49661 | — |
| `J_DP` | 2.43449 | 2.43449 | — |
| adaptive advantage | 0.06212 | 0.06212 (2.55%) | ≥ 1 % ✓ |
| level shift | 3.8 % of advantage | 3.8% | ≤ 10 % ✓ |
| clairvoyant bracket | 8.5 % | 9.7% | ≤ 15 % ✓ |
| σ_L → 0 vs M4a's certified value | 2.383218 | 2.383217 (+1.8e-06) | — |
| λ selected | 10^−3.5 | 10^−3.5, agreeing with the linear and power-law tables and therefore with M3 and M4a | must agree ✓ |

Two words in that table are load-bearing and easy to blur. The λ gate is about
the three **tables** — linear, power-law, and the liquidity world's static one —
selecting the same point, which is what makes M4b's result comparable to M3's
and M4a's. It is *not* a claim that every way of **reading** the rule agrees,
because they do not: that is the subject of the next paragraph, and the two
senses of the word must not be allowed to borrow each other's authority.

The clairvoyant bracket is a **sampled** quantity, so the figure above is the
one the *gate* was decided on — `results/m4b_reference.json`, 9.7487% of the
advantage. The trained sweep re-estimates the same bracket on its own liquidity
stream and gets 9.7883% (`results/m4b_liquidity.json`); the two differ by
2.46e-05 bps and straddle a rounding boundary at one decimal place. Every
*deterministic* quantity in the table — `J_DP`, the advantage, the level shift —
is identical between the two artefacts to the last bit.

**Gate 1 needed a decision, and the two candidate readings disagree.** The brief
asked task 0 to *decide and record* how the selection rule applies in a world that
is not a new encoding, and predicted the answer. The prediction was right and the
alternative was not merely different — it was **knife-edge**:

| reading | selects | margin on the 20 % TWAP-gap bar at 10^−4 |
| --- | :---: | ---: |
| static (`J_static*`, the schedule) | **10^−3.5** | −3.94 points — misses, and it is a closed form |
| adaptive (`J_DP` and the DP's mean schedule) | 10^−4.0 | **+0.011 points** — clears, on the fifth digit of a numerical value function |

The static reading was taken. The deciding argument is not the margin: both of the
rule's conditions are properties of a **schedule** — condition (ii) asks for the
largest single-bin fraction — and a policy has no single schedule. The DP's *mean*
schedule is an average no episode ever executes. The rejected reading is recorded
in the driver's gate-1 output, the config's comment, `LiquidityReferenceRow`'s
docstring and a test that pins its margin, because a session that had quietly
taken the agreeing reading would have made the choice unauditable.

## Task 1 — the adaptive oracle, converged and bracketed

The word *certified* is absent everywhere this reference is reported, and the
results file says so in the field a reader would look for it in
(`reference_kind: converged and bracketed, not certified`). M4a earned that word
with a Cholesky factorisation and a 1.2e-15 KKT residual; a stochastic dynamic
program has no such object.

| check | result |
| --- | --- |
| grid convergence | second order, monotone from above; Richardson residual **1.96e-06 bps** |
| quadrature | converged by **5** nodes; pinned at 15 |
| σ_L → 0 vs M4a's *certified* optimum | **+1.8e-06 bps** — the single most valuable check in the milestone |
| sufficiency (augmented state carrying `L_{k−1}`) | agrees to 4.4e-16; continuation spread across previous multipliers 4.4e-16 |
| feasible upper bound (interpolated stage solve) | inside the 2 %-of-advantage band |
| clairvoyant lower bound | `lower ≤ J_DP ≤ upper` with the CIs carried ✓ |

**The 2 % band is a test of the action map, not of convergence**, and at any
affordable path count it cannot be anything else: the estimate's own half-width
dwarfs the gap it would have to resolve. So the comparison that resolves is made
against a *snapped* stage solve on the same paths, where interpolating is worth
**11 % of the advantage** — five times the band, resolved to 0.15 % because the
comparison shares its liquidity.

**The red-flag test came out sharper than the brief predicted.** Perfect
information beats any policy on **every path**, not merely on average, because the
clairvoyant solve is the per-path minimum over all schedules and the agent's
realised schedule is one of them. So the hard failure is a *count* with no
confidence interval in it at all.

## Task 4's precondition — the guarantees, before a seed was spent

`make m4b-guarantees`, run and recorded **before** task 5 started.

| guarantee | worst observed | bar |
| --- | ---: | ---: |
| exact per-episode noise identity | 2.397e-14 relative | ≤ 1e-12 |
| antithetic cancellation | 3.416e-16 bps per step | ≤ 1e-12 |
| action identity across the pair | exact, through a **three**-coordinate observation | bitwise |
| the two halves saw the same `L` | exact, and shown live | bitwise |
| discriminative: the deterministic reference | misses by 1.4e-1 relative | ≫ band |

**§9's M4a entry was half right, and the wrong half is the useful one.** It named
"a second, independent noise source **or** a price-bearing observation" as what
ends the pairing's exactness. The disjunction is too wide: what action identity
needs is not a *poor* observation but one the two halves **agree about**. They
share their liquidity, so they see the same three-vector, take the same action,
and the price noise still cancels exactly given `(x, L)`.

**Where the third identity had to be fired from is the interesting part.** A
mirror on a wholly different liquidity path is refused at `reset` by the
*pre-existing* observation check — in this world the multiplier is in the
observation, so that check does new work for free. The dedicated one is therefore
fired by perturbing only the **last** bin's multiplier: every observation still
agrees (the terminal entry is 0.0 for both halves), the schedules are identical,
and nothing but M4b's third identity is looking at the charge.

## Task 5 — the training point

Ten seeds, σ_L = 0.5 (**invented**), M4a's configuration verbatim, no
hyperparameter search.

| quantity | measured | bar |
| --- | ---: | ---: |
| capture fraction, median | **0.9896** | ≥ 0.90 ✓ |
| capture fraction, IQR / worst seed | 0.0125 / 0.9628 | worst ≥ 0.75 ✓ |
| median excess over `J_DP` | **+0.00064 bps** | ≤ 0.00621 ✓ |
| liquidity-shuffled control, median capture | **-1.0086** (worst -0.8932) | ≤ 0.15 ✓ |
| paths below perfect information | **zero**, on every seed | zero |
| soft flags (below `J_DP`) | none | reported, not failed |
| sweep wall-clock | 8,576 s | ≤ 14 000 ✓ |

Per seed: 0.9777, 0.9919, 0.9906, 0.9905, 0.9947, 0.9741, 0.9888, 0.9628, 0.9830, 0.9947.

**The control is the claim.** Re-graded with the observed liquidity drawn
independently of the liquidity charged, the same policies capture
-1.01 — they do *worse than not reacting at all*. The gap
between 0.99 and -1.01 is what says the agent
is using the signal rather than having found a better fixed schedule.

**The reward noise the brief predicted was there, and the agent trained through
it.** Averaged reward variance 3.269e-02
bps² per update against M4a's 2.63e-07 — five orders larger, and 12.9 % of the
effect per update at 512 envs against the predicted 12.7 %. The configuration was
M4a's, unchanged; that it trains through this is the finding the brief asked for
either way.

## Task 2 — the seam's acceptance, across two worlds

`make m4b-regression`, **both green**. One committed M3 seed (the Phase-1 world, `configs/m3_frontier/lambda_1e-3.5.yaml`) and one committed M4a seed (the power-law world) retrained through the second seam and reproduced their committed grades **bitwise** — objective to seventeen digits and every point of the trajectory — in ~40 min for the pair.

Bitwise rather than `allclose`, for M4a's reason: `allclose` would pass on a seam that changed the order of a float addition, which is exactly the failure worth catching, because PPO compounds it over ~750 updates and M2 measured the same seed address landing at 0.165 and 0.066 of the TWAP gap under nothing worse than a different thread count.

Three fast checks run in `make test` alongside them, because each names a way this could have gone wrong silently: neither committed config acquires a liquidity world by omission, the observation is still two-dimensional wherever liquidity is deterministic, and the two noise sources are addressed in **different pools** with the same root seed and index. That last one is the whole argument — a liquidity draw taken out of the price generator would shift every downstream shock, and every committed result would still regenerate perfectly from its own config against a different market.

Every committed figure redraws byte-identically: `m2_trajectory_overlay`,
`m3_antithetic_overlay`, `m3_frontier` (via `figure --redraw`; the plain `figure`
subcommand re-aggregates and re-stamps by design), `m4a_degradation`,
`m4a_trajectory_overlay`.

## What the value of sight is worth, against the invented parameter

| σ_L | E[L^−0.6] | adaptive advantage | % of `J_DP` | level shift / advantage |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 1.03045 | 0.01474 bps | 0.62% | 0.9% |
| 0.5 | 1.12750 | 0.06212 bps | 2.55% | 3.8% |
| 0.75 | 1.30996 | 0.15223 bps | 6.09% | 8.5% |

σ_L is **Temper's own invention**. FrontierView has no liquidity process, so the
result is this curve and not any one row of it, and the claim is stated in the
same sentence as the parameter every time it appears: *with a one-parameter
invented liquidity process, seeing liquidity is worth 2.6%
of the objective and the agent captured 99% of it.*

## The house note, four times

`docs/house-notes.md`'s note — *No code path may be reachable only at the end of a long run*, and it was titled *The artefact writer
is tested on fabricated data, not on the run* at the time — was cited by this
brief as a thing to obey. It was obeyed for
`build_document` — the new artefact keys were covered on fabricated data before
the training run, exactly as asked — and the same defect class then arrived
**four more times** in code the brief had not named:

1. `_on_seed` read `Grade` fields off a `LiquidityGrade`. Caught by watching the
   first launch; cost 20 minutes instead of a night.
2. `--dry-run` printed M4a's tangent advantage as M4b's bar — understating the
   pre-stated bar by 1.7× in the *flattering* direction. Found at the same time.
3. `tools/m4b_adaptivity.py`'s `main` died reporting where it had written the
   figure. The figure existed; the process did not survive saying so.
4. The closing summary read `summary['relative_excess']` and died **after** all
   ten seeds were graded and both artefacts written. Nothing was lost only
   because `write_outputs` runs before the printing does.

All four are pure functions of data that were reachable only behind a producer.
All four are now extracted and tested on fabricated inputs in milliseconds. The
portable lesson the note already states is right; what M4b adds is that the
*reporting* path is as much an artefact writer as the JSON assembly is, and a
driver that dies while printing a grade has thrown away the run just as
completely.
