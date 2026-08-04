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

**Status:** M1 done. `temper/oracle` lands the Almgren–Chriss closed forms and matches 16
vendored FrontierView cases plus a 17-point frontier to float round-off, ten orders of
magnitude inside the pre-stated 1e-6 tolerance (M0). `temper/env` lands `ExecutionEnv` and
proves it by differential: TWAP and both AC schedules run *as policies* through the real
step loop, and their simulated cost moments match the closed forms across the full 3 × 3
golden grid at 100,000 episodes a cell — plus five exact per-episode identities and a
variational certificate that the schedule M2 grades against really is the minimiser. M2
(PPO rediscovery) is next. See `ROADMAP.md`.

## Layout

```
temper/     the package: oracle/ (AC closed forms), env/ (seeded market models),
            agents/ (baselines + single-file PPO), eval/ (Monte-Carlo grading, figures)
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
make differential  # the deep Monte-Carlo tier: pytest -m deep
```

`make test` is the per-commit gate and stays evening-sized (~11 s). `make differential` is
the milestone acceptance gate: 27 (case, schedule) cells at 100,000 episodes each, ~2.5 min
on the reference box, driven by `configs/m1_differential.yaml`. Both regenerate from that
committed config plus one root seed.

Python ≥ 3.11, CPU-only by design (constitution §6.9) — reference box is the Ryzen 7
3800X the rest of the portfolio benchmarks on; a GPU only ever accelerates. The suite
needs no network and no GPU, and `tests/test_repo_invariants.py` enforces both.

On Linux the default PyPI `torch` wheel bundles CUDA; add
`--extra-index-url https://download.pytorch.org/whl/cpu` to keep the install CPU-sized.
Torch is unused until M2 — the oracle, the env and the differential run on numpy,
gymnasium, pyyaml and pytest alone, and the repo-invariant tests keep torch out of
`temper/oracle` and `temper/env` for good.

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
