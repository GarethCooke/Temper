# M5 — Alpha-aware execution

**Track:** agentic · **Size:** one evening for the oracle and the differential, then ten
seeds; budget **attended** time separately from wall-clock — M4b's ceiling estimated the
machine's work and missed the session's by 55 % · **Reads first:** `ARCHITECTURE.md` §4
including *Amended by M4a*, §5, invariants 3, 4, 5, 6, 7; then five §9 entries **by title** —
*A liquidity-observing policy is graded by conditional expectation, not by sampling realised
cost* (M5 is its second application and the one that names the pattern), *The antithetic pair
holds liquidity common, so action identity survives a richer observation* (M5 is where action
identity finally ends, and the entry already says why), *A numerical reference is bracketed,
not certified, and the red-flag test moves to the bound that is rigorous* (M5 moves it
again, and to a better bound), *A metric grades the world that charges it*, and *The
tolerance's denominator is the available advantage, not the TWAP gap*. Also the house note
*No code path may be reachable only at the end of a long run*, which M5's ROADMAP row already
carries as a definition-of-done item rather than as advice.

## Amendments (invariant 3 — recorded before the work they govern)

**1. The soft red flag is calibrated against the evaluation half-width, not only the
DP's residual (2026-08-24, recorded after task 0 and before task 1).** The pre-stated
table said `J_agent < J_DP` *beyond the DP's own convergence residual*. Task 0 shows
that threshold is too tight by nearly three orders, and it shows it with the least
ambiguous possible witness: **the dynamic program's own greedy policy priced 0.6558 %
of the advantage below `J_DP`** at M = 20 000 held-out signal paths. That policy is
the DP's optimum rolled out — it cannot be a defect — so the old rule would have
raised a flag on the reference against itself.

The residual was never the binding uncertainty. Richardson on the inventory grid gives
**1.49e-06 bps = 0.0019 % of the advantage**; the *evaluation* half-width at M = 20 000
is **0.00084 bps = 1.0405 %**, five hundred times larger. A grade is a Monte-Carlo
estimate and the threshold has to be stated in its units.

Restated, with the M named so the threshold is a number rather than a policy:

> `J_agent < J_DP - (Richardson residual + evaluation half-width)`, which at the
> committed **M = 200 000** is `J_DP - 0.000261 bps` — **0.3238 % of the advantage**.

**M = 200 000, not M4b's 20 000.** Task 0 measured both on the same machinery: the
half-width falls from 1.0405 % of the advantage to **0.3219 %** for ~20 s of extra
rollout per policy, and the soft flag is only worth stating at a resolution where it
can distinguish a defect from a draw. It also puts the median tolerance bar **31x**
above the measurement floor, against M4b's 7.4x. `SIGNAL_BOUND_PATHS` in
`temper/eval/reference.py` is the reference table's own count and stays at 20 000;
this is the *grading* count and task 6 states it.

The false-alarm rate is stated rather than left implied: a policy that is exactly
optimal sits below a one-half-width threshold about **2.5 % of the time per seed**, so
across ten seeds roughly one run in five will show a flag from sampling alone. That is
precisely why this flag is **soft** — reported and investigated, never auto-failed —
and it is what the hard flag of the row above is for.

**Amended now rather than at task 6, on purpose.** The calibration exists as of task 0;
a tolerance loosened with ten trained seeds already in hand is not a pre-stated
tolerance, whatever the arithmetic behind it says.

---

## Objective

The observation becomes partially predictive of the price the order will pay. This is the
first milestone whose advantage comes from **information** rather than from a better-solved
optimisation — M4a's advantage was there because the closed form solved the wrong problem,
M4b's because the market was random and the agent could see it, and M5's because the agent
knows something about what is going to happen.

That difference is not cosmetic, and the scoping below is why this brief is longer than its
predecessors. A price signal is measured against per-bin volatility, and per-bin volatility
at this case is **18× the entire objective**. So an alpha signal weak enough to be honest is
still strong enough to dominate the milestone if it is let, and the naive reading of the
ROADMAP row — *weak signal, agent tilts the schedule, capture fraction green* — is not a
well-posed milestone. Making it well-posed is task 0's job and most of this brief.

## Context — four facts that fix this milestone's shape

Every number below was computed against the committed oracle in a cloud container on
unpinned numpy, using `powerlaw.power_law_charge`, `powerlaw.inventory_penalty_scale` and
`powerlaw.schedule_invariant_bps` — the repo's own functions, not a reimplementation. The
same bench reproduces M4b's committed table exactly (`J_static*` 2.496615, `J_DP` 2.434490,
advantage 0.062124), which is what licenses the predictions here. **None of it is a committed
artefact.** Task 0 regenerates all of it on the reference box.

### 1. The world is M4a's, not M4b's, and that is an attribution decision

Power law, deterministic liquidity, plus the signal. **Not** M4b's stochastic liquidity as
well.

Bundled, a red result could not be attributed — the same argument that split M4a from M4b,
and it is stronger here because the two adaptivities respond to different randomness and
would compete for the same schedule shape. It also keeps the dynamic program's state at two
dimensions instead of three. Stacking liquidity and alpha is a real milestone and it is
**backlog**, named in `ROADMAP.md` rather than smuggled in here.

The consequence to state plainly: M5 is not "M4b plus a signal". It is M4a's world with a
different second thing in the observation, measured the same way.

### 2. The signal, owned as invented, and its one parameter

At the decision point for bin *k* the observation carries `s_k ~ N(0,1)` with
`E[xi_{k+1} | s_k] = rho * s_k` — a one-step-ahead partial view of the shock that has not
landed yet. One parameter. **rho = 0.01**, so the signal explains `rho^2 = 1e-4` of next-bin
return variance: one part in ten thousand.

Invented, and FrontierView has no alpha model, so §7's *vendored, not invented* cover does
not extend to it. The same rule M4b earned applies without weakening: **rho appears in the
same sentence as the result, every time the result is stated.**

Why one part in ten thousand is the right order, and why the ROADMAP's word *weak* was
carrying four orders of magnitude:

| rho | R^2 | net advantage, bps | as a multiple of M4b's |
| ---: | ---: | ---: | ---: |
| 0.0025 | 6.3e-06 | 0.005698 | 0.09x |
| 0.0050 | 2.5e-05 | 0.021998 | 0.35x |
| **0.0100** | **1.0e-04** | **0.080753** | **1.30x** |
| 0.0200 | 4.0e-04 | 0.278472 | 4.48x |
| 0.0500 | 2.5e-03 | 1.267187 | 20.40x |
| 0.2000 | 4.0e-02 | 8.890347 | 143x — larger than the objective itself |

A signal an equities researcher would call vanishingly weak is worth more than everything
M4b measured. At rho = 0.05 the "execution cost" goes negative and the agent is no longer
executing, it is trading. **rho = 0.01 is chosen so that M4a, M4b and M5 report advantages
on one scale** (0.0368 / 0.0621 / 0.0808 bps), which is what makes the three Phase-2
milestones comparable and lets M4b's tolerance machinery transfer instead of being redesigned.

### 3. The advantage is a difference of two much larger numbers, and that decides the grading

At rho = 0.01, the dynamic program's optimum decomposes (1601 grid points, 15 nodes; the
identity `J = impact + risk + alpha + invariant` closes to the last digit):

| term | rho = 0 (certified) | rho = 0.01 DP | difference |
| --- | ---: | ---: | ---: |
| temporary impact | 0.907718 | 0.939129 | +0.031411 |
| inventory risk | 0.911870 | 0.947764 | +0.035894 |
| alpha | 0.000000 | **-0.148067** | -0.148067 |
| **objective** | **2.383215** | **2.302456** | **-0.080759** |

The optimal alpha-aware policy monetises **0.1481 bps** of signal and pays **0.0673 bps** of
worse execution to do it — **45.5 % of the alpha is given back**. The headline advantage is
what survives.

So a single capture fraction against the net advantage **cannot distinguish between two
completely different policies**: one that captures 0.15 of alpha and pays 0.07, and one that
captures 0.25 and pays 0.17, score identically at a headline that is about execution quality
and reports neither. M4a's methodological finding was that the tolerance's denominator was
wrong; M5's, pre-stated here, is that **a single denominator is not enough** — the milestone
grades the two halves separately or it grades nothing.

### 4. The inherited red-flag test dies, and its replacement is better

M4b's hard red flag was the perfect-information relaxation: no adapted policy beats
clairvoyance. Measured here, over 400 paths on a 401-point grid, price clairvoyance is worth
**-84.39 +/- 8.77 bps** — an advantage of 86.77 bps, **1,075x** the signal's. A test that
loose can never fire. Recorded rather than quietly dropped: retiring an inherited test needs
evidence.

Its replacement is rigorous, tight, and *certified* rather than numerical. Impact and risk
are convex in the inventory path and do not involve the signal at all, so by Jensen
`E[impact + risk] >= min over deterministic schedules = J_M4a_varying = 1.819586` for **any**
policy, adaptive or not, with equality only at M4a's certified optimum. The DP sits
0.067305 above that floor.

**The red-flag test therefore moves from a numerical bound on the whole objective to a
certified bound on the half that carries no information.** An agent reporting execution
cheaper than M4a's certified optimum has a defect, and the certificate — Cholesky, KKT
residual 1.2e-15 — is already committed. M4b had to bracket because it had no such object;
M5 does, and only for the half where it exists.

## Tasks

### 0. The alpha reference table and the four gates — *gate*

Oracle only. **No training code is written, imported or run until every gate below is
recorded green in the repo.**

Build the two-state dynamic program over `(inventory, signal)` on `VENDOR_LAMBDA_GRID`, and
at the selected lambda produce the decomposition of task-context 3 — impact, risk, alpha and
the objective, with the identity asserted rather than assumed.

**Lambda needs no new reading, and that is a result worth recording.** A zero-mean signal
does not move a deterministic schedule's objective by a single float, so the liquidity
world's *third reading* problem does not recur: the static reading here is **bit-identical**
to M4a's, selects 10^-3.5, and agrees with M3, M4a and M4b. Assert the bit-identity; do not
assume it.

**Gate conditions:**

- the rho = 0 dynamic program returns M4a's certified `power_law_optimum` value to within
  1e-4 of the advantage — the single most valuable check in the milestone, and the direct
  successor to M4b's `sigma_L -> 0`;
- the net advantage at rho = 0.01 is >= 1 % of the objective;
- the execution premium is between **25 % and 75 %** of the alpha. Below 25 % the
  decomposition is decorative and one headline would do; above 75 % the advantage is a small
  difference of large numbers and the milestone is re-shaped here, before training, rather
  than caveated afterwards. Predicted 45.5 %;
- the price-clairvoyant relaxation is computed, its looseness recorded as a multiple of the
  advantage, and the convexity bound asserted in its place with `E[impact + risk] >
  J_M4a_varying` shown to hold for the DP by a margin large enough to grade against.

### 1. The reference, and what word it earns

The dynamic program is **converged**, not certified — same word discipline as M4b, and
`AdaptiveOptimum`'s `"certified": false` and `reference_kind` fields already exist to carry
it. Report grid and quadrature convergence with a Richardson residual, and the sufficiency
check that an augmented state (carrying `s_{k-1}`) does not improve the value: the signal is
i.i.d. by construction, so it must not, and a violation means the seam leaks.

The **certified** object in this milestone is M4a's optimum, used as the execution floor. Say
which is which everywhere both appear; they are different kinds of confidence and M5 is the
first milestone to use both at once.

### 2. The third seam, and the third stream

The signal is a seam like the impact model and the liquidity law, injected rather than
subclassed, with **its own seed pool** — `POOLS` gains `m5/signal-train` and
`m5/signal-eval`, never drawn from the price generator or the liquidity generator. Invariant
5 asks for disjointness by construction.

Phase 1 and M4a must reproduce **bitwise** through the new seam, and so must M4b's committed
seed. Three worlds now, and the regression covers all three.

**The observation grows once**: `s_k` and nothing else.

### 3. The price-free assertion finally retires, and what does not retire with it

`ExecutionEnv`'s observation-minimality guard has refused a price-bearing observation since
M1. M5's observation *is* price-bearing — that is the milestone — so the guard is amended,
not deleted, and the amendment names exactly what is now permitted: a signal about a shock
that has **not yet landed**, never the realised price and never the realised shortfall. A
guard that stops asserting anything is worse than no guard.

The consequence §9 already predicts arrives here: the antithetic pair's **action identity
ends**, because the two halves see signals they disagree about. The per-step identity assertion
is retired *for the signal* and kept for everything else.

### 4. Grading stays analytic, one rung further along

Conditioning moves from the liquidity path to the **signal path**. Given `s_1..s_N` the
policy's actions are deterministic, cost is affine in the price draws given the inventory
path, so

`E[cost | s] = sum scale w^(1+beta) + lambda B sum h^2 - A rho sum h_k s_{k-1}`

is a closed form again, with `A = sigma_bin * BPS = 42.9893` bps. The Monte-Carlo interval is
over **signal** paths only; there is no price sampling anywhere, and the assertion that
licenses that — the policy never sees a realised price — is the half of the price-free entry
that does **not** retire in task 3.

The antithetic pair mirrors the **signal** draws. Predicted, and to be measured rather than
asserted: pairing cancels the part of the alpha term that is linear in `s` — which is the
noise, mean zero — and keeps the part that is quadratic, which is the value. The pairing
should therefore help *more* here than in M4b, not less, despite action identity ending.
If that prediction is wrong, the training budget is wrong and task 5 reshapes before it runs.

### 5. The differential — invariant 6, three worlds

M1's tiers against the new env, with `step_count` asserted rather than assumed. Two new
identities: the signal stream is independent of the price stream at the per-draw level, and
`E[cost | s]` agrees with a sampled mean over price draws at a stated CI with the signal path
pinned.

### 6. The training point, and the two controls that make it mean something

Ten seeds at the selected lambda, M4a's configuration unchanged — **no hyperparameter
search**; if it cannot train, that is the finding.

Report **three** numbers side by side, never the first alone:

- **alpha capture** — the agent's monetised signal over the DP's 0.148067;
- **execution premium** — the agent's `E[impact + risk]` excess over M4a's certified
  1.819586, as a fraction of the DP's 0.067305;
- **net capture** — the headline, `(J_M4a - J_agent) / (J_M4a - J_DP)`, with the absolute
  excess in bps beside it as everywhere else in this repo.

The **signal-shuffled control** is the claim, exactly as the liquidity shuffle was M4b's:
re-grade the trained policies with the observed signal drawn independently of the shock
charged. The policy still tilts, still pays the execution premium, and monetises nothing.
Predicted net capture **-0.83** — it should be *worse than not reacting at all*, and by
almost exactly the premium it paid.

## Pre-stated numbers (invariant 3 — loosen only by amending this brief before work starts)

| Item | Value |
| --- | --- |
| Case | AAPL, X = 100 000, T = 6.5 h, N = 13 — unchanged since M2 |
| World | M4a's power law, deterministic liquidity, plus the signal. **Not** M4b's liquidity |
| Signal | one-step-ahead, `E[xi_{k+1}|s_k] = rho s_k`, `s ~ N(0,1)`. **Invented, and labelled so** |
| rho | **0.01** trained; 0.005 / 0.01 / 0.02 reported from the oracle |
| lambda | the rule-selected point; the static reading is **bit-identical** to M4a's and selects 10^-3.5 |
| Reference | the `(inventory, signal)` DP — **converged**, not certified |
| Execution floor | M4a's **certified** optimum, `J_M4a_varying = 1.819586` bps — rigorous by convexity |
| `J_M4a` (no signal) | 2.383215 bps |
| `J_DP` (rho = 0.01) | 2.302456 bps |
| Net advantage | **0.080759 bps**, 3.39 % of the objective |
| Alpha available | 0.148067 bps |
| Execution premium at the optimum | 0.067305 bps = **45.5 %** of the alpha |
| Tolerance, median net capture | >= **0.90** — M4b's bar, transferred because the advantages are on one scale |
| Tolerance, per seed | >= **0.75** |
| Alpha capture, median | >= **0.85**, reported beside the net figure and never instead of it |
| Execution premium, median | <= **1.30x** the DP's — an agent may pay more for its alpha than the optimum does, but not half as much again |
| Red flag (hard) | `E[impact + risk] < 1.819586 - eps`. **Rigorous and certified**, not numerical |
| Red flag (soft) | `J_agent < J_DP - (Richardson residual + eval half-width)` = `J_DP - 0.000261 bps` = **0.3238 % of the advantage** at the committed M. Calibrated in **amendment 1** rather than guessed: the DP's own greedy policy sat 0.6558 % below `J_DP` at M = 20 k and was noise. Fires by chance ~2.5 % per seed — report and investigate, **never auto-fail** |
| Shuffled control | net capture <= **-0.50**; predicted -0.83 |
| Eval draws | **M = 200 000** paired signal paths, common random numbers across every policy; report the achieved half-width. 200 k rather than M4b's 20 k for the reason in **amendment 1**: the half-width falls from 1.0405 % of the advantage to 0.3219 % for seconds of rollout, which puts the median bar 31x above the measurement floor |
| Price clairvoyance | recorded and **retired** as a red flag: -84.39 +/- 8.77 bps, 1 075x the advantage |
| Pairing | mirror the **signal**; action identity ends by design and its assertion retires with a recorded reason |
| Objective | unchanged. The per-path functional gains a term that used to be zero. **No amendment to invariant 7** |
| Regression | one M3 seed, one M4a seed and one M4b seed bitwise; all committed figures byte-identical |
| Seeds | 10, `train`; eval on `eval`; the signal on its own pool in both |
| Torch threads | 8, pinned |
| Dependencies | none new |

## Definition of done

- [ ] Alpha reference table committed; all four task-0 gates recorded green, or the milestone
      re-shaped here with the reason.
- [ ] `rho -> 0` returns M4a's certified value; grid and quadrature convergence reported;
      the sufficiency check green.
- [ ] The `J = impact + risk + alpha + invariant` identity asserted, not assumed.
- [ ] lambda's static reading shown **bit-identical** to M4a's rather than merely agreeing.
- [ ] The signal seam landed with its own seed pool; the config must name its signal model;
      one `step` loop still.
- [ ] **Bitwise regression across three worlds** — M3, M4a and M4b seeds identical; every
      committed figure byte-identical.
- [ ] The observation-minimality guard **amended, not deleted**, and still refusing a
      realised price or a realised shortfall.
- [ ] Conditional grading on the signal path landed; `cost_moments` without a signal
      bit-identical to today.
- [ ] The pairing's behaviour on the alpha term **measured** against this brief's prediction.
- [ ] Differential green at M1's tiers with `step_count` asserted.
- [ ] Ten seeds; **three** numbers reported together — alpha capture, execution premium, net
      capture — with absolute bps beside every fraction.
- [ ] Signal-shuffled control run and inside its bar.
- [ ] The price-clairvoyant relaxation computed, its looseness recorded, and the convexity
      bound asserted in its place.
- [ ] `results/m5_alpha.*` with every seed drawn and the rho curve; `git_dirty: false`.
- [ ] Every artefact-producing path exercised on fabricated data **before** the training run —
      the whole path, not the producer (`docs/house-notes.md`, *No code path may be reachable
      only at the end of a long run*).
- [ ] README Phase-2 rung completed, naming the signal as Temper's own invention.
- [ ] Clean clone through the documented interface; `ROADMAP.md` M5 row flipped; structural
      findings to §9.

### §9 amendments this milestone is expected to yield

Predicted so the session recognises them, and records what it actually found instead:

1. **A price-bearing observation is permitted when the price has not happened yet.** The
   observation-minimality guard is amended rather than deleted, and the line it now draws is
   between a shock that has landed and one that has not.
2. **Conditioning is the pattern, not the trick.** Successor to the liquidity entry: price-free,
   then conditional on liquidity, now conditional on the signal — the same closed form each
   time, because the policy's actions are deterministic given whatever it observed.
3. **The red-flag test moves to the certified half.** A perfect-information bound on a price
   path is 1 075x too loose to fire; convexity gives a rigorous, tight bound on the half of
   the objective the signal cannot touch. Possibly the most portable of the three.
4. **An advantage that is a difference of larger numbers is graded in its parts.** The
   successor to the denominator entry: when 45 % of the gross effect is paid back, one
   fraction cannot distinguish a good policy from a lucky one.

## Out of scope (resist)

Stacking M4b's stochastic liquidity with the signal — a real milestone, backlogged, and
bundling it destroys attribution. Persistent or AR(1) alpha, and a multi-step-ahead signal:
both change the DP's state and neither is needed to make the point. Any signal strong enough
to make the objective negative. A lambda sweep in the alpha world. Hyperparameter search: the
configuration is M4a's. Trading the signal without the completion constraint — the parent
order still finishes. New dependencies. Re-rendering or restyling any committed figure. The
Anvil wire.

## Session notes

- **Gate 3's lambda-invariance is a candidate §9 entry, not a task-0 detail — noted
  here so it is not lost between tasks (measured 2026-08-24, task 0).** The execution
  premium is **44.9-49.9 % of the gross alpha across all seventeen lambdas** of the
  vendor grid, while the advantage it is a fraction of moves from 13.9 % of the
  objective at 10^-9 to 0.0003 % at 10^-1. That is what turns entry 4 of the list above
  — *an advantage that is a difference of larger numbers is graded in its parts* — from
  an accommodation for one operating point into a structural claim: the decomposition is
  needed everywhere on the grid, not at the lambda M2's rule happened to select. **M5's
  execution-premium tolerance rests on it**, because a bar of 1.30x the DP's premium
  means something only if that premium is a property of the problem rather than of the
  point. It lands with the milestone's other §9 amendments; task 1 confirms it is not a
  discretisation artefact before it is promoted anywhere.

- **The gate that matters most is the execution premium**, not the advantage. Everything else
  in this brief follows from 45 % of the gross alpha being paid back, and if that number
  comes back at 5 % or 90 % the milestone is a different milestone and this brief is wrong
  before the code is.
- **Do not let rho drift upward if training struggles.** A larger signal makes the advantage
  bigger and the milestone easier and also stops it being an execution result — at rho = 0.05
  the objective is more than half signal, and at rho = 0.2 the "cost" is negative. If ten
  seeds cannot find 0.08 bps under this brief's estimator, *that is the finding*, and it is
  the M2 noise-threshold result one rung along rather than a reason to move the parameter.
- **The clairvoyant number is worth computing even though it retires a test.** It is the
  cleanest way to show a reader why a signal explaining one part in ten thousand is worth
  more than the whole liquidity result: per-bin volatility is 18x the objective, so
  information about price is worth vastly more per unit than information about cost.
- **Every number in this brief is a prediction from a cloud container on unpinned numpy**,
  computed with the repo's own oracle functions against a bench that reproduces M4b's
  committed table exactly. A material disagreement is not a tolerance to loosen — it means
  this brief is wrong before the code is. Stop and report.
- **The claim is narrower than it will be tempting to write.** It is "with an invented
  one-step-ahead signal explaining one part in ten thousand of next-bin variance, seeing it
  is worth 3.4 % of the objective, the agent captured most of that, and it gave back 45 % of
  the gross effect in worse execution doing so" — not "the agent trades on alpha". The
  invented parameter belongs in the same sentence as the result, every time.
