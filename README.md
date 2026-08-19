# Temper

A reinforcement-learning execution agent graded against its own analytic oracle. Temper
trains a policy to work a parent order and holds it to the Almgren–Chriss closed forms
that [FrontierView](https://garethcooke.com) already implements — first proving the agent
*rediscovers* the optimal schedule where the maths says one exists, then breaking the
model's assumptions to show where learning earns a genuine advantage.

The forge family, continued: Anvil matches orders, Crucible measures metal under heat,
Temper is the controlled treatment that follows forging.

**The three-phase claim structure** (the honest version, in ascending order of difficulty):

1. **Rediscovery** — under exact AC assumptions the closed form is provably optimal, so
   success is "the agent converges to the sinh trajectory within a pre-stated tolerance",
   never "the agent beats Almgren–Chriss". The analytic solution plays the role Anvil's
   `ref_engine.py` plays: a second, independent answer the implementation must match.
2. **Earned advantage** — add what the closed form cannot see (transient impact with decay,
   stochastic liquidity, a weak alpha signal) and show the agent adapting where the
   now-mis-specified AC schedule degrades — out-of-sample, ≥5 seeds, error bars, with TWAP
   and AC still on every chart.
3. **The wire (stretch)** — the trained policy works a parent order on the live
   [Anvil](https://anvil.garethcooke.com) book as `PROTOCOL.md`'s third independent
   client. A demo that the policy speaks a real venue wire; performance claims stay in the
   simulator.

**Status:** M4a done — the first *earned* advantage, and it is bounded on both sides.
FrontierView's temporary impact is a 0.6-power law, which Almgren–Chriss has no closed form
for; the vendored library linearises at the tangent and solves that instead. M4a makes the
power law the world, and the agent has to find *that* world's optimum while the tangent-derived
schedule does not: `results/m4a_degradation.png`.

The numerator is what the agent beat the closed form by. The denominator is what there was
to beat — a **certified** power-law optimum rather than a best-so-far, solved by Newton on
the KKT system and checked five ways (Cholesky PD, relative KKT residual 1.2e-15, 3 600
perturbations uphill, an independent bisection solver agreeing to 3.1e-15 of the parent
order, and the same solver at exponent 1 returning the sinh to 3.5e-16). Without the
denominator, "the agent beats AC" is a number with no scale.

At the rule-selected λ the mis-specification is worth **0.0367 bps** — 1.54 % of the
objective, 2.50 % of expected cost, **16 878 shares (16.9 % of the parent order)** in
trajectory space. Ten seeds captured a median **99.4 %** of it (IQR 0.2 %, worst seed
99.0 %), a median absolute excess over the certified optimum of **+0.00021 bps**, with the
red-flag test green on every seed. The tolerance is a fraction of that *available
advantage* and not of the TWAP gap, which is the milestone's methodological finding: 5 % of
the TWAP gap here is twice the whole effect, so M2's and M3's bar could not have told
success from complete failure.

Two things the milestone establishes about the machinery, not the agent. Phase 1 reproduces
**bitwise** through the new env seam — one `step` loop, the impact model injected rather
than subclassed — so every M2 and M3 number still regenerates from code that exists. And the
four guarantees the new world inherits were checked *before* training: the exact per-episode
noise identity, the antithetic pair's exact cancellation, its action identity, and the
open-loop schedule check are all still green, because the power law replaces a term that
carries no shock. That check earned its place immediately by catching an antithetic mirror
that was quietly charging the Phase-1 world.

Earlier: **M3** traced the risk–cost frontier — nine λ across four decades, ten seeds each,
every seed drawn (`results/m3_frontier.png`), median excess over the certified optimum
between **+0.004 % and +0.33 %** everywhere, red-flag test green throughout, and the per-λ
tolerance met at eight of nine (the exception, λ = 10^−5, measures the tolerance's
denominator rather than the agent). What made the sweep affordable is the reward regime. M2
needed a control variate that subtracts the analytic noise form — exact, but it needs a
closed form for the noise. M3 replaced it with **antithetic pairing**: every episode runs
twice, against the shock path and its exact negation, and the rewards are averaged. Because
the observation carries no price the agent takes identical actions in both halves —
asserted bitwise on every step, not assumed — so the noise cancels on the average, and the
realised per-update reward variance drops by **eleven orders of magnitude** under the
agent's own actions. It was validated at M2's λ against M2's committed answer before being
used anywhere: median gap fraction 0.000168 against a gate of 0.002, and on the seeds the
two runs share, the trained policies agree *bitwise*.

The honesty ladder below says what this does and does not establish.

Before that: `temper/oracle` lands the Almgren–Chriss closed forms and matches 16
vendored FrontierView cases plus a 17-point frontier to float round-off, ten orders of
magnitude inside the pre-stated 1e-6 tolerance (M0). `temper/env` lands `ExecutionEnv` and
proves it by differential: TWAP and both AC schedules run *as policies* through the real
step loop, and their simulated cost moments match the closed forms across the full 3 × 3
golden grid at 200,000 episodes a cell. The differential is mostly not statistical — an
exact per-episode identity pins the realised noise to the specific draws the env made, so
the cost assembly holds by construction and the Monte-Carlo tiers certify only that the
shocks are iid normal. Alongside it: six exact per-episode identities, an exact step
count, and a variational certificate that the schedule M2 grades against really is the
minimiser. M4b (stochastic liquidity) is next. See `ROADMAP.md`.

## What this does and does not establish

The honesty ladder, written while the limits are fresh (M3 task 6). Each rung is a claim
the repo can back with a committed config, a committed result and a green suite — and each
is deliberately narrower than it might sound.

- **Phase 1 — the pipeline works.** The agent recovers Almgren–Chriss in a world where AC is
  *provably* optimal. That is a statement about the optimiser, the environment and the
  grading path agreeing with a closed form, not about trading: the world is arithmetic
  Brownian motion with linear impact, the observation is `(time left, inventory left)`, and
  the reward the headline agents train on is variance-reduced (M2's control variate, M3's
  antithetic pairing) — on the realised reward at this case's ~1:70 per-episode
  signal-to-noise ratio the same agent misses the bar as a lottery, and that miss is
  committed beside the pass. The frontier (M3) is nine points of that same world.
- **Phase 2 — the agent finds the optimum of a world whose closed form is derived at a
  tangent.** Half of it has landed (M4a). FrontierView's temporary impact is a 0.6-power
  law; Almgren–Chriss has no closed form for that, so the vendored library *linearises* at
  the tangent to it and solves the linear problem instead. M4a makes the power law the
  world and grades the agent against that world's own optimum — solved by Newton on the KKT
  system and **certified** (Cholesky PD, relative KKT residual 1.2e-15, 3 600 perturbations
  uphill, an independent bisection solver agreeing to 3.1e-15 of the parent order), because
  a reference nobody checked is not a reference.
  The mis-specification is real and small. At the reference case the tangent-derived
  schedule costs **1.54 % of the objective** more than the optimum — 2.50 % of expected
  cost alone, 0.0367 bps either way — which in trajectory space is **16 878 shares, 16.9 %
  of the parent order**. Ten seeds captured a median **99.4 %** of that (IQR 0.2 %, worst
  seed 99.0 %): a median excess over the certified optimum of **+0.00021 bps**, and a
  median ‖δ‖₂ of 727 shares, 23× closer to the optimum than the closed form is. The two
  numbers travel together everywhere, because the fraction alone would make a very small
  claim sound like a large one — and because the *denominator* is the thing that had to
  change: 5 % of this λ's TWAP gap, which is what M2 and M3 graded against, is twice the
  entire available advantage, so the old bar could not have told success from complete
  failure.
  The now-wrong AC schedule and TWAP are on every chart. It says the agent adapts to a
  model change the formula cannot. It says nothing about real fills, and 0.037 bps is a
  small absolute claim that should read as one. **The liquidity half is not done.** M4b
  makes liquidity a second, *independent* noise source, which is what actually breaks
  analytic grading and the antithetic pairing — M4a deliberately left the observation
  untouched so that a red result could be attributed to one thing.
- **Phase 3 — it runs on a wire against synthetic data.** The stretch leg (M6) works a parent
  order on the live Anvil book. That is plumbing evidence — the policy speaks a versioned
  venue protocol end to end — and not execution-quality evidence: the flow it trades
  against is a synthetic feeder, non-adversarial by construction.

**None of this establishes real-market performance.** That would need real fills, or
historical order-book data to replay against, and neither is in the portfolio. Every number
here is a statement about a simulator whose dynamics are analytic and whose parameters are
FrontierView's calibrated synthetic set; the differential tests are what make it a *good*
simulator, and the honest reading of the results is "the machinery is correct", not "the
strategy would work". This is also the answer to the obvious interview question, and it is
better volunteered than extracted.

## Layout

```
temper/     the package: oracle/ (AC closed forms), env/ (seeded market models),
            agents/ (baselines, single-file PPO, the fraction-of-remaining action),
            eval/ (grading — analytic and Monte-Carlo, reference tables, figures)
configs/    one committed config per experiment/figure
tests/      pytest; golden/vendor/ holds FrontierView-generated fixtures (provenance-stamped)
results/    committed metrics JSON + figures — regenerable from config + seed
client/     (M6) the Anvil participant — the only networked code
tools/      dev scripts
docs/       milestone briefs (the work orders) + vendored contract snapshots
```

Orientation docs at root: `ARCHITECTURE.md` (the constitution — read first),
`ROADMAP.md` (milestones and status). Practices that outgrew the milestone that found
them live in [`docs/house-notes.md`](docs/house-notes.md) — currently *thread count is a
reproducibility axis* and *below n ≈ 10, draw every trace*, both measured here and both
applying to any project in the portfolio that reports a trained or sampled number.

## Build (host)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make test          # or, without GNU make: python -m pytest
make differential  # M1's deep Monte-Carlo tier: pytest -m deep
make smoke         # M2's PPO convergence check on Pendulum + CartPole
make reference     # M2's oracle-only table, and the lambda its rule fixes
make sweep         # M2's 5-seed sweeps, both estimators — hours, unattended
make validate      # M3 task 1: antithetic pairing at M2's λ, 10 seeds — a night
make frontier      # M3 tasks 4–5: the nine-λ sweep, then the frontier figure — a day
make m4a-reference # M4a task 0: the power-law table and its three gates — minutes
make m4a           # M4a task 5: ten seeds in the power-law world — ~2 h
```

`make test` is the per-commit gate and stays evening-sized (~15 s; the brief's ceiling is
3 min). The rest are milestone acceptance gates, each behind a pytest marker or a driver so
the per-commit loop never waits on them:

- `make differential` — the deep Monte-Carlo tiers of both worlds: M1's 27 (case, schedule)
  cells at 200,000 episodes each and M4a's 36, 163.8 M calls into the real `step` loop,
  counted and asserted; from `configs/m1_differential.yaml` and
  `configs/m4a_differential.yaml`.
- `make smoke` — the CleanRL adaptation solving `Pendulum-v1` and `CartPole-v1` on three
  seeds each, ~7 min, from `configs/ppo_smoke.yaml`. It stays in the suite permanently
  because it is what separates "PPO is broken" from "the env is hard" for every later
  milestone.
- `make sweep` — M2's acceptance run, from `configs/m2_ppo.yaml` and
  `configs/m2_ppo_sampled.yaml`. Hours, unattended, and it writes the committed
  `results/*.json` and the overlay figures. `--figure-only` redraws a figure from the
  committed JSON without retraining.
- `make validate` — M3's gate, from `configs/m3_antithetic_validation.yaml`: the
  antithetic-pairing regime at M2's λ on ten seeds, everything else identical to M2's
  control-variate config, accepted against that run's committed median rather than
  against ε. A night, unattended, strictly serial with everything else on the box.
  All the training targets go through one driver, `tools/train.py`, which reads
  the milestone, the reward regime and the λ off the config.
- `make frontier` — M3's sweep, from `configs/m3_frontier.yaml` and the nine point configs
  it generates under `configs/m3_frontier/` (byte-checked against the generator before
  anything runs). Each λ is an ordinary experiment through `tools/train.py`; the aggregate
  `results/m3_frontier.json` and the frontier figure are views of the nine results files
  and redraw without training (`python tools/m3_frontier.py figure`).
- `make m4a-reference` — M4a's task 0, oracle only and minutes: the power-law table, M2's
  selection rule applied to it, and the three gates. Exit status is whether all three are
  green, so it is a check rather than a report.
- `make m4a-guarantees` — the four guarantees the power-law world inherits from Phase 1,
  run *before* training. `make m4a-regression` — one M3 seed retrained through the new env
  seam and required to reproduce its committed grade **bitwise**, ~12 min.
- `make m4a` — M4a's acceptance run, from `configs/m4a_power_law.yaml`: ten seeds in the
  power-law world, everything except the world and the graded encoding identical to M3's
  point at the same λ. ~2 h, unattended. `make m4a-figure` redraws
  `results/m4a_degradation.png` from the committed JSON without training.

Everything regenerates from a committed config plus one root seed. Every entry in
`results/` carries the config's SHA-256 and the git revision that produced it, and the
suite re-reads that digest — a result cannot survive an edit to the thresholds it was
measured against.

Python ≥ 3.11, CPU-only by design (constitution §6.9) — reference box is the Ryzen 7
3800X the rest of the portfolio benchmarks on; a GPU only ever accelerates. The suite
needs no network and no GPU, and `tests/test_repo_invariants.py` enforces both.

On Linux the default PyPI `torch` wheel bundles CUDA; add
`--extra-index-url https://download.pytorch.org/whl/cpu` to keep the install CPU-sized.
Torch enters only with M2's agent — the oracle, the env and the differential run on numpy,
gymnasium, pyyaml and pytest alone, and the repo-invariant tests keep torch out of
`temper/oracle` and `temper/env` for good. Matplotlib arrives with M2's figure and is
confined by the same mechanism to `temper/eval/figures.py`, which forces the `Agg` backend
before importing pyplot; importing `temper.eval` pulls neither.

### Regenerating the goldens

`temper/oracle` earns its authority by agreeing with an implementation this repo did not
write. The fixture under `tests/golden/vendor/` is exported from a FrontierView checkout
by a script that is read-only with respect to that repo (constitution §7):

```bash
make goldens FRONTIERVIEW=/path/to/FrontierView
```

Provenance, the case list and the upstream quirks the export surfaced are recorded in
[`docs/vendor/frontierview-goldens.md`](docs/vendor/frontierview-goldens.md). Never
hand-edit the fixture, and never synthesise it from Temper's own closed forms — that
would collapse the differential into a tautology.

## Development model

Milestone briefs in `docs/briefs/` are executed by agentic sessions converging on the
pytest suite's red/green — the DepthCharge model. Correctness is pinned by the analytic
oracle and by expectation tests on every market model: if the environment's moments don't
match the closed forms for a fixed policy, nothing downstream merges. Every reported
number regenerates from a committed config + seed, carries baselines (TWAP, AC at matched
λ), and reports dispersion across ≥5 training seeds.

Related properties: [Anvil](https://garethcooke.com/projects/anvil) ·
[FrontierView](https://garethcooke.com) · [Crucible](https://crucible.garethcooke.com) ·
[DepthCharge](https://github.com/GarethCooke/DepthCharge) ·
[garethcooke.com](https://garethcooke.com)
