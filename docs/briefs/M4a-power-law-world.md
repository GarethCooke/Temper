# M4a — The power-law world

**Track:** agentic · **Size:** one evening — task 0 and the certificate are minutes of
oracle arithmetic, the differential is a ~10 min run, and the training point is ten seeds
at one λ (~2.0–2.8 h serial, from M3's measured per-seed times) · **Reads first:**
`ARCHITECTURE.md` §4 including both *Amended by* bullets, §5's *Amended by M3*, and
invariants 2, 3, 6, 7; then three §9 entries by title — *Phase 1 is the linearised world
end-to-end; `cost_moments` is reporting context only*, *Antithetic pairing is the Phase-1
variance-reduction regime, and at this reward magnitude it is bitwise the control
variate*, and *The per-λ tolerance is meaningful only where the testbed is discriminative,
and the frontier measures where that stops being true*. The third one is what re-bases
this milestone's ε, and a session that skips it will pre-state a tolerance twice the size
of the effect.

**M4 is split.** M4a is the vendored power law with the observation untouched; M4b is
stochastic liquidity, which enriches the observation and therefore breaks analytic
grading, the antithetic action-identity check, and the "no Monte-Carlo interval" property
all at once. Bundled, a red result could not be attributed to a break. Split, M4a inherits
the entire M3 apparatus unchanged and M4b changes one thing at a time. `ROADMAP.md`'s M4
row becomes M4a and M4b; M5 and M6 do not renumber.

## Objective

The project's first *earned* advantage. The env's temporary impact becomes FrontierView's
calibrated 0.6-power law — the vendored mis-specification, not an invented one — and the
agent is required to find that world's own optimum while the Almgren–Chriss schedule,
derived at the tangent η̃, does not.

The claim is bounded on **both** sides, and that is the point of the milestone. The
numerator is what the agent beat the AC schedule by. The denominator is what was there to
be beaten: a *certified* power-law optimum, not a best-so-far. Without the denominator,
"the agent beats AC" is a number with no scale, and this repo has spent three milestones
refusing to report those.

## Context — three facts that fix this milestone's shape

### 1. The mis-specification is real, small, and invisible to M3's tolerance

The power-law objective evaluated on the AC schedule, against the power-law world's own
optimum, at the reference case (AAPL, X = 100 000, T = 6.5 h, N = 13):

| λ | J_twap | J_ac (vendored κ) | J_opt (tangent) | J_pow\* | TWAP gap | available advantage | avail / TWAP gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10^−5 | 1.1271 | 1.1443 | 1.1222 | 1.1220 | 0.005 | 0.018 % | 0.039 |
| 10^−4.5 | 1.3208 | 1.3499 | 1.2788 | 1.2772 | 0.034 | 0.125 % | 0.037 |
| 10^−4 | 1.9332 | 1.7578 | 1.6479 | 1.6387 | 0.180 | 0.565 % | 0.031 |
| **10^−3.5** | **3.8697** | **2.5144** | **2.4200** | **2.3832** | **0.624** | **1.542 %** | **0.025** |
| 10^−3 | 9.9936 | 4.1533 | 4.1149 | 4.0212 | 1.485 | 2.330 % | 0.016 |
| 10^−2.5 | 29.359 | 8.4514 | 8.4588 | 8.3170 | 2.530 | 1.704 % | 0.007 |
| 10^−2 | 90.598 | 21.231 | 21.245 | 21.128 | 3.288 | 0.552 % | 0.002 |

Two readings, both load-bearing.

**The advantage exists and peaks in the middle of the grid.** It is 1.54 % of the
objective at the rule-selected λ and 2.33 % at 10^−3. In trajectory space it is much
larger than that sounds: the tangent-derived schedule sits **16 878 shares — 16.9 % of the
parent order — from the power-law optimum**, front-loading 32.6 % of the order in the
first bin where the power-law optimum front-loads 42.3 %. The power law charges
`Σ n^1.6` where the tangent charges `Σ n²`, so concentrating is cheaper than the closed
form believes, and the correct schedule is faster. That is a schedule difference a chart
will show.

**M3's ε cannot see any of it.** ε was 5 % of *that λ's TWAP gap*. At 10^−3.5 that is
0.0663 bps as M3 actually computed it (on the linear encoding it graded), or 0.0743 bps
re-derived in the power-law encoding — either way **1.8–2.0× the entire available
advantage of 0.0367 bps**. An agent graded to M3's tolerance in the power-law world would
pass while capturing none of the mis-specification. The §9 entry *The per-λ tolerance is
meaningful only where the testbed is discriminative…* said the denominator is the thing to
watch; this is that entry arriving as a hard constraint rather than a caution. **M4a's
tolerance is a fraction of the available advantage, not of the TWAP gap** (task 0, and the
pre-stated table).

### 2. Nothing about the estimator or the grading route needs to change — and that is a prediction, not an assumption

§9's antithetic entry says the pairing's exactness "does not exist in Phase 2, where cost
stops being affine in the shocks". That sentence is about a Phase 2 that has not been
built yet, and it is **wrong about this half of it**. The power law replaces the temporary
term, which is a function of the schedule and carries no shock. Realised cost stays

```
C = f(x)  −  σ_bin · Σ_{k=0}^{N−1} (x_k / X) ξ_k
```

with only `f` changing. So M1a's exact per-episode noise identity should still hold, the
antithetic pair should still cancel exactly, the observation is untouched so the
action-identity assertion should stay green, and `deterministic_schedule` should still
certify an open-loop schedule. Every one of those is a **test that already exists**, so
task 4 checks all four in minutes rather than reasoning about them. If any goes red, that
is the milestone's finding and training does not start that night.

The corollary matters for M4b: what actually breaks the pairing is a second, *independent*
noise source or a price-bearing observation — not curvature in the cost.

### 3. The power-law optimum is a convex program, so it can be certified rather than trusted

On the reachable set — sell-only, fully liquidating, which `ExecutionEnv`'s clip to
`[0, remaining]` makes the *only* set (M2's band derivation already leans on this) —
the power-law objective in the trade weights is

```
J(w) = A · Σ w_i^1.6  +  λ B · Σ_{k<N} (1 − Σ_{i<k} w_i)²  +  const,     Σ w = 1, w ≥ 0
```

`A = η σ · BPS · (X / (dt·v_hourly))^0.6`, `B = (σ_bin·BPS)²`. Permanent cost telescopes
to `γσ·BPS·X / (2 v_hourly)` and the spread to `half_spread`; both are schedule-invariant
for a monotone full liquidation, exactly as `permanent_cost_bps` documents. `w^1.6` is
strictly convex on `w ≥ 0` and the variance term is a convex quadratic, so `J` is strictly
convex on the simplex and its stationary point is the **unique global minimum**. There is
no sinh and there does not need to be one.

It also does not need scipy. An equality-constrained Newton iteration on the KKT system
in dense numpy reaches a KKT residual of ~1e-15 in well under twenty iterations at every λ
on the grid, which is the same "generic linear algebra, no formula" discipline M1 task 0
used. **Do not add a dependency for this.**

## Tasks

### 0. The power-law reference table, and the λ it fixes — *gate*

Build the power-law analogue of M2's task-0 table: for every λ on `VENDOR_LAMBDA_GRID`,
the power-law objective of TWAP, `ac_trajectory`, `optimal_trajectory` and the certified
power-law optimum, plus the TWAP gap, the available advantage `J_opt(tangent) − J_pow*`,
and the maximum bin fraction of the power-law optimum. Oracle only; no agent exists yet.

Apply **M2's selection rule unchanged** — smallest λ with TWAP gap ≥ 0.20 and the
optimum's max bin fraction ≤ 0.50 — to this table. Predicted: it selects 10^−3.5, the same
λ M2 and M3 committed, so M4a's point is directly comparable to two committed results and
`Experiment.verify_lambda_rule` needs an encoding parameter rather than a bypass. **Assert
that both encodings' tables select the same λ.** If they do not, stop: a Phase-2 milestone
whose λ was chosen by a different rule than the Phase-1 results it is compared against is
not comparable to them, and the resolution is a decision, not a rescue.

**Gate conditions, all three:**

- the rule selects the same λ under both encodings;
- the available advantage at that λ is ≥ 1 % of `J_pow*` — below that the training point
  is not worth an evening and the milestone leads with M4b instead;
- the trajectory band implied by the median tolerance is comfortably inside the
  AC-schedule's distance from the optimum, so the test discriminates in trajectory space
  as well as in objective space. Predicted: band 4 739 shares vs a 16 878-share separation,
  a factor of 3.6.

### 1. The certified power-law optimum

`temper/oracle/` gains the power-law optimum as first-class oracle surface — it is what
M4a grades against, and invariant 2 says the oracle is normative. Pure numpy.

The certificate, in `tests/`, is M1 task 0's shape adapted to a solve rather than a
formula:

- **(a) Convexity, stated and checked.** The Hessian in the interior holdings at `x*` is
  PD (Cholesky succeeds); record `λ_min` and the condition number. Predicted at 10^−3.5:
  `λ_min = 1.636e-10` bps/share², cond ≈ 34.6 — against Phase-1's `1.5985e-10` from
  `objective_curvature_floor`, i.e. the power-law optimum sits in a *slightly* sharper
  bowl. Assembling the Hessian and recovering M3's committed 28 797-share band from the
  Phase-1 branch is the cheap cross-check that the new assembly is right.
- **(b) KKT.** At the solution the marginal cost is equal across every bin with `w_i > 0`
  and no lower bound is active (predicted: min weight 0.0059 at 10^−3.5, interior at
  every λ up to 10^−2). Require a relative KKT residual ≤ 1e-12.
- **(c) Perturbation.** 200 random feasible interior directions at `‖δ‖ ∈ {1e-3, 1e-6}·X`,
  require `J(x*+δ) − J(x*) ≥ −1e-9·|J(x*)|`. Same test as M1 task 0, same reason.
- **(d) Independence.** A second solver — projected gradient, or a coarse grid search
  refined by bisection on the equal-marginal-cost condition — reproduces `x*` to ≤ 1e-10
  relative of X. Slow and inelegant on purpose: it is the differential check on the Newton
  solve, as `optimal_trajectory_by_solve` is on the closed form.
- **(e) The Phase-1 limit.** Replacing the power law with its tangent in the same solver
  must return `optimal_trajectory` to ≤ 1e-12 relative of X. One test, and the two worlds
  are demonstrably the same machinery at different exponents.

The band is derived, not chosen, exactly as in M2 — but state honestly that it is now a
**local** bound: the Hessian is no longer constant in `x`, so `‖δ‖₂ ≤ √(2Δ/λ_min(H(x*)))`
is a statement at the optimum. Validate it by direct evaluation on random directions at
the band radius rather than asserting the quadratic inequality.

### 2. The env seam, and the registry becomes world-aware

**The env.** Temporary impact becomes an injected model rather than a precomputed
per-share constant: `temper/env/impact.py` with `LinearTemporary(eta_tilde)` and
`PowerLawTemporary(eta, sigma, beta)`, each declaring its `encoding` and returning the
per-share concession in bps. `ExecutionEnv` takes one, defaults to the linear model built
exactly as today, and exposes `cost_encoding`. §4's "additive alternatives behind the same
interface, never silent modifications of Phase 1" is honoured by the config having to
*name* the world: no config may inherit a Phase-2 world by default, and
`tests/test_repo_invariants.py` gains that check.

One env, one `step` loop. Not a subclass with a duplicated loop — the differential's
`step_count` claim (invariant 6, M1) is a claim about *the* loop, and two of them make it
unfalsifiable again.

**Bitwise regression is the seam's acceptance.** `make test` green is necessary and not
sufficient. Re-run **one M3 seed at 10^−3.5** through the new code and require the trained
objective and the whole trajectory to be **bitwise identical** to the committed value
(~20 min; M2 and M3 both established that this repo reproduces bitwise on a fixed thread
count, so anything less than bitwise is the seam changing float order in Phase 1). Redraw
`results/m3_frontier.png` and require byte-identity.

**The registry.** `temper/eval/metrics.py`'s quarantine does not get deleted; it gets
*generalised*. Today `GRADEABLE_ENCODINGS = {LINEAR}`, which is the right rule stated in
the only way that was checkable when one world existed. The rule M4a needs is **a metric
grades the world that charges it**: registries keyed by encoding, and the grader asserting
`metric.encoding == env.cost_encoding` before it computes anything. That is strictly
stronger than what it replaces — the flat rule could not have caught a *linear* metric
grading a power-law env, which is now the live failure mode — and it is the same shape as
M2's replacement of the flat seed-pool ban with the per-module allow-list.

`tests/test_objective_registry.py` extends to check the labels are true rather than
honest, on both worlds: every graded metric evaluated against both encodings on the golden
cases, and a mismatch between env and metric refused by construction.

### 3. The differential, re-run against the power-law env — invariant 6

Non-negotiable and cheap. TWAP, `ac_trajectory`, `optimal_trajectory` and the new
power-law optimum run **as policies** through the power-law env; simulated E[cost] and
V[cost] must match `oracle.cost_moments` within the pre-stated CIs. Reuse M1's tiers and
harness with the analytic reference swapped; the variance reference is unchanged, because
`shortfall_variance_bps2` does not depend on the impact model — which is itself worth
asserting rather than assuming, since it is the reason only one of the two moments needs a
new reference.

Keep M1a's `step_count` assertion. A vectorised shortcut round the loop is exactly as
tempting here as it was there.

### 4. The four inherited guarantees, checked before training — *minutes, not a night*

Run these four existing tests against the power-law env and record the results **before**
task 5 starts. Predicted green, for the reason in context §2:

| Guarantee | Test | What red would mean |
| --- | --- | --- |
| Exact per-episode noise identity | `tests/test_noise_identity.py` | cost stopped being affine in the shocks — news, and M4b's design changes |
| Antithetic cancellation is exact | `tests/test_m3_antithetic.py` | the pairing degraded a milestone earlier than predicted |
| Action identity across the pair | same | something reached the observation |
| Schedule is open-loop | `deterministic_schedule` | analytic grading is invalid and nothing downstream is a number |

If they are green, say so plainly and note that the §9 antithetic entry's prediction was
right about the mechanism and wrong about the milestone — the break that ends exactness is
a second noise source, not curvature. Do not weaken a red test here to proceed.

### 5. The training point

Ten seeds at the rule-selected λ, `train` pool, antithetic regime, everything else
identical to M3's committed configuration except the world and the graded encoding.
Evaluate on the `eval` pool, graded analytically through the world-keyed registry.

Report, all of them: the **capture fraction**
`c = (J_opt(tangent) − J_agent) / (J_opt(tangent) − J_pow*)` — median, IQR, per seed; the
absolute excess over `J_pow*` in bps (§9: never report only a fraction); the gap fraction
against TWAP for continuity with M3; median `‖δ‖₂` against the derived band; and per-seed
wall clock.

**Red flag, hard failure on every seed:** `J_agent ≥ J_pow* − 1e-9·|J_pow*|`. The optimum
is certified in task 1, so a strictly lower objective is a defect in the metric, the env or
the grading path — never a win. Unchanged in form from M2 and M3; only the reference moves.

Sanity check before launching: the committed `reward.scale: 0.02` was set against a
Phase-1 objective of 2.3546 bps (M2's task-0 `J_optimal`, reproduced bitwise by M3) and the
power-law objective at this λ is 2.3832 bps — 1.2 % apart, so the scale carries at the
episode level. The *per-step* charge is another matter: it is more front-loaded under the
power law. Confirm the
scaled per-step reward stays in the range PPO's value head and advantage normalisation
were tuned on, and record the check rather than assuming it.

### 6. Degradation figure, and the README ladder moves one rung

`results/m4a_degradation.*`: excess over the certified power-law optimum against λ, for
TWAP, `ac_trajectory` and `optimal_trajectory` across the whole grid — oracle only, so it
is free — with the agent's ten seeds drawn individually at the one λ that was trained
(*Below n ≈ 10, draw every trace*). This is the ROADMAP's "AC-schedule degradation
quantified", and the agent's marker is the milestone in one picture: how far the closed
form is from the truth, and how much of that distance learning recovered.

The README's Phase-2 rung currently reads "when it lands (M4+)". Rewrite it to what is
actually established, in the same voice as the rest of the ladder: the advantage is
earned against a *mis-specified vendored* closed form inside a synthetic AC-shaped market;
it is worth ~1.5 % of expected cost at the reference case; it says the agent adapts to a
model change the formula cannot, and nothing about real fills. Say explicitly that the
liquidity half is not done.

## Pre-stated numbers (invariant 3 — loosen only by amending this brief before work starts)

| Item | Value |
| --- | --- |
| Case | AAPL, X = 100 000, T = 6.5 h, N = 13 — M2/M3's, unchanged |
| World | FrontierView's calibrated 0.6-power temporary impact; permanent, spread and shock model unchanged |
| λ | the rule-selected point of the **power-law** table; predicted 10^−3.5, and required to agree with the linear table's selection |
| Grading reference | the **certified** power-law optimum `J_pow*`, not `optimal_trajectory` |
| Advantage denominator | `J_opt(tangent) − J_pow*` — the available advantage. **Not** the TWAP gap: at 10^−3.5, 5 % of the TWAP gap is 1.8–2.0× the whole effect |
| Tolerance, median | excess over `J_pow*` ≤ **5 % of the available advantage** ⇒ capture fraction `c ≥ 0.95`. Predicted bar 0.00184 bps; M3 measured a median excess of 0.00013 bps at this λ, so 13.9× headroom against a measured precedent |
| Tolerance, per seed | ≤ **10 %** of the available advantage ⇒ `c ≥ 0.90` per seed |
| Red flag (hard fail) | any seed with `J_agent < J_pow* − 1e-9·\|J_pow*\|` |
| Gate — advantage | available advantage ≥ 1 % of `J_pow*` at the selected λ, or M4a does not train and M4b leads |
| Gate — λ agreement | both encodings' tables select the same λ |
| Gate — inherited guarantees | the four task-4 tests green **before** training starts |
| Certificate | Hessian PD at `x*`; KKT residual ≤ 1e-12 relative; 200 perturbations uphill; independent solver ≤ 1e-10 of X; tangent limit reproduces `optimal_trajectory` ≤ 1e-12 of X |
| Trajectory band | derived from `λ_min(H(x*))`, stated as **local**, validated by direct evaluation at the band radius; predicted 4 739 shares (4.74 % of X) at the median bar |
| Differential CIs | M1's tiers and CI level, unchanged, with `cost_moments` as the expectation reference |
| Seeds | 10, `train` pool; eval on `eval` pool; disjoint by construction |
| Dispersion | median + IQR across seeds. **There is no Monte-Carlo interval in M4a** — grading is analytic, so "CIs" here means seed dispersion and the report says so. Sampling intervals arrive with M4b's liquidity noise |
| Phase-1 regression | one M3 seed at 10^−3.5 retrained **bitwise** identical; `results/m3_frontier.png` byte-identical |
| Torch threads | 8, pinned in committed config (*Thread count is a reproducibility axis*) |
| Concurrency | serial |
| Dependencies | **no new ones.** The optimum is dense numpy; scipy is not added for one solve |
| Wall-clock ceiling | ~4 h total including the regression seed; over that, report and stop rather than trimming seeds |
| Suite impact | `make test` ≤ 3 min; the differential's deep tier behind its marker |

**Why the SNR argument is not repeated here and is repeated here anyway.** The effect is
0.0367 bps against a per-episode cost standard deviation of ~95 bps — a ratio of ~1:2 600,
roughly 37× worse than the 1:70 M2 measured. Any suggestion during the session of
estimating the agent's objective by sampling realised costs is answered by that number
before it is answered by anything else.

## Definition of done

- [ ] Power-law reference table committed; the rule applied to it; both encodings agree on λ.
- [ ] All three task-0 gates recorded green (or the milestone re-shaped and the reason written here).
- [ ] Certified optimum in `temper/oracle/`, with (a)–(e) of the certificate green and no scipy.
- [ ] Env seam landed; config must name its world; one `step` loop; repo-invariant test extended.
- [ ] Phase-1 regression: one M3 seed bitwise identical, `results/m3_frontier.png` byte-identical.
- [ ] Registry keyed by encoding; grader asserts world↔metric match; registry test covers both worlds.
- [ ] Differential green against the power-law env at M1's tiers, with `step_count` asserted.
- [ ] The four inherited guarantees checked and recorded **before** training.
- [ ] Ten seeds trained; median and per-seed capture fraction against the pre-stated bars;
      absolute excess in bps reported beside every fraction; `‖δ‖₂` against the band.
- [ ] Red-flag test green on every seed.
- [ ] `results/m4a_degradation.*` with every seed drawn; config hash + git rev; `git_dirty: false`.
- [ ] README Phase-2 rung rewritten to what was established, including what was not.
- [ ] Clean clone through the documented interface: `make help`, `make test`, `make reference`,
      configs dry-run, figures redraw byte-identical.
- [ ] `ROADMAP.md`: M4 row split into M4a (flipped) and M4b (queued); anything structural → §9.

### §9 amendments this milestone is expected to yield

Predicted, so the session recognises them rather than discovering them at 2 a.m. — and
records what it actually found, which may be none of these:

1. **A metric grades the world that charges it.** Supersedes the blanket refusal of the
   power-law encoding; the quarantine generalises rather than lifts.
2. **The power law's break is in a shock-free term, so the noise identity and the
   antithetic pairing survive it exactly.** Narrows the antithetic entry's "does not exist
   in Phase 2" to what is actually true: exactness ends with a second noise source or a
   price-bearing observation, not with curvature.
3. **The tolerance's denominator is the available advantage.** The successor to *The per-λ
   tolerance is meaningful only where the testbed is discriminative…*, and the reason M3's
   ε could not be reused.

## Out of scope (resist)

The observation — any enrichment of it, which is M4b's entire subject and which would
invalidate analytic grading and the pairing on the same night the world changed.
Stochastic liquidity. Transient impact with decay. Alpha. The Anvil wire. Hyperparameter
search — the configuration is M3's, and if it does not train in the power-law world that
is a finding, not a reason to tune. A λ sweep: the frontier is M3's and re-running it in
the power-law world is a separate, sized milestone, not a stretch goal for tonight. New
dependencies.

## Session notes

- **Task 0 is a gate, not a formality.** Three conditions, all readable off the oracle in
  minutes. If the available advantage at the selected λ is under 1 % of `J_pow*`, the
  honest move is to report that and let M4b lead — the power law alone would then not be
  worth an evening of the box, and finding that out from arithmetic beats finding it out
  from a flat training curve.
- **The numbers in this brief came from a cloud container on unpinned numpy and are
  predictions, not results.** Nothing here is a committed artefact. Task 0 regenerates all
  of them on the reference box from committed configs; if a regenerated number disagrees
  materially, the disagreement is the first thing to understand, and this brief is wrong
  before the code is.
- **A red test in task 4 is information, not an obstacle.** The noise identity going red
  would mean realised cost stopped being affine in the price shocks, which would be a
  genuine surprise and would change M4b's design. Report it and stop; do not relax it.
- **The capture fraction is the number to lead with, and the absolute excess in bps goes
  beside it every time.** M3's §9 entry exists because a fraction alone made a healthy
  agent look like a degrading one. The same trap is one milestone away in the other
  direction: a capture fraction near 1 on an advantage of 0.037 bps is a small absolute
  claim and should read as one.
- **"The agent beats Almgren–Chriss" is still not the sentence.** The sentence is "the
  agent finds the optimum of a world whose closed form is derived at a tangent, and the
  tangent costs 1.5 %". §1's red flag was about beating AC *inside AC's assumptions*; this
  is outside them, and the difference is worth saying out loud in the README rather than
  leaving to the reader.
- Preserve pre-run reasoning verbatim with retractions beneath it. House practice since M2,
  and task 0's gate is exactly the shape of thing that benefits.
