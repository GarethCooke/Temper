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

**Status:** M2 done. The agent rediscovers Almgren–Chriss — and the interesting part is what
it took. Trained on the noise-free reward it converges on the exact discrete optimum to a
median of **+0.0115 % of `J_optimal`** across five seeds (0.0002 of the TWAP gap against a
pre-stated ε of 0.05), sitting a median 1 336 shares from the sinh where the objective's own
curvature allows 28 797. Trained on the *realised* reward — the same agent, the same
hyperparameters, the full 1:70 per-episode signal-to-noise ratio — it misses the same bar
(median 0.098, worst seed 0.819), and misses it as a **lottery** rather than a plateau: one
seed in five lands inside ε, another barely learns at all. Both runs are committed, differ
in exactly one config field, and the weaker claim travels with the figure. Grading is
analytic: the observation carries no price, so a deterministic policy induces an open-loop
schedule whose objective is a closed form — one rollout per seed, zero Monte-Carlo error,
behind a bitwise assertion that the schedule really is shock-independent.

Earlier: `temper/oracle` lands the Almgren–Chriss closed forms and matches 16
vendored FrontierView cases plus a 17-point frontier to float round-off, ten orders of
magnitude inside the pre-stated 1e-6 tolerance (M0). `temper/env` lands `ExecutionEnv` and
proves it by differential: TWAP and both AC schedules run *as policies* through the real
step loop, and their simulated cost moments match the closed forms across the full 3 × 3
golden grid at 200,000 episodes a cell. The differential is mostly not statistical — an
exact per-episode identity pins the realised noise to the specific draws the env made, so
the cost assembly holds by construction and the Monte-Carlo tiers certify only that the
shocks are iid normal. Alongside it: six exact per-episode identities, an exact step
count, and a variational certificate that the schedule M2 grades against really is the
minimiser. M3 (the frontier sweep) is next. See `ROADMAP.md`.

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
- **Phase 2 — the agent beats AC where AC's formula breaks, inside an AC-shaped market.**
  When it lands (M4+), the "advantage" is earned against a *mis-specified* closed form —
  FrontierView's calibrated 0.6-power temporary impact, stochastic liquidity, a weak alpha
  signal — with the now-wrong AC schedule and TWAP still on every chart. It is still a
  synthetic, AC-shaped market with realistic impact curvature: it says the agent adapts to
  a model change the formula cannot, not that it would make money.
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
`ROADMAP.md` (milestones and status).

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
```

`make test` is the per-commit gate and stays evening-sized (~15 s; the brief's ceiling is
3 min). The rest are milestone acceptance gates, each behind a pytest marker or a driver so
the per-commit loop never waits on them:

- `make differential` — 27 (case, schedule) cells at 200,000 episodes each, 70.2 M calls
  into the real `step` loop, counted and asserted; ~5.5 min, from `configs/m1_differential.yaml`.
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
