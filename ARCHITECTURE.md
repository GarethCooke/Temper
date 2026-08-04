# Temper — Architecture

**Status:** Constitution. This document changes only by explicit decision (record it in §9).
Per-milestone work is specified in disposable briefs under `docs/briefs/`; this file is what
every session reads first and must not violate.

---

## 1. What this is

Temper trains a reinforcement-learning execution agent to work a parent order, and grades it
against the Almgren–Chriss closed forms that FrontierView already implements. It is the
differential-oracle pattern (Anvil's `ref_engine.py`) applied to RL, in three phases:

1. **Rediscovery.** Under exact AC assumptions the closed form *is* optimal, so the claim is
   never "the agent beats Almgren–Chriss" — it is "the agent independently converges to the
   closed form within a pre-stated tolerance". The analytic solution is the reference engine.
   Claiming to beat AC inside AC's own assumptions is a red flag, not a result.
2. **Earned advantage.** Break the assumptions the closed form needs — transient impact with
   decay, stochastic liquidity, a weak alpha signal — and show where learning genuinely wins,
   out-of-sample, with dispersion reported, and with the now-mis-specified AC schedule and
   TWAP still on every chart as baselines.
3. **The wire (stretch).** The trained policy works a parent order on the live Anvil book,
   becoming `PROTOCOL.md`'s third independent client. A demonstration that the policy speaks
   a real venue wire — explicitly *not* an evaluation venue (§7).

Temper consumes FrontierView's model the way DepthCharge consumes Anvil's wire: separate
repo, versioned boundary artefacts vendored with provenance, **zero changes required
upstream** in either FrontierView or Anvil.

## 2. System overview

```
 configs/*.yaml ──► ExecutionEnv ──► rollouts ──► PPO update ──► policy
                    (seeded market      │                          │
                     model, §4)         ▼                          ▼
 oracle/  ◄──────── eval harness ◄── baselines (TWAP, AC schedule as policies)
 (AC closed forms,       │
  golden-pinned)         ▼
                  results/*.json + figures   (regenerable from config + seed)
```

One Python package, four responsibilities: `oracle/` (closed forms), `env/` (market models
behind one Gymnasium-style interface), `agents/` (baselines + PPO), `eval/` (Monte-Carlo
grading + report generation). Everything runs on the host CPU; every reported artefact
regenerates from a committed config and seed.

## 3. Repo layout

| Path           | Owns                                                                       |
| -------------- | -------------------------------------------------------------------------- |
| `temper/oracle/` | AC closed forms: κ solve, sinh trajectory, E[cost], V[cost], frontier; TWAP moments. Must match the vendored goldens (§6.2). |
| `temper/env/`    | `ExecutionEnv` (§4) + market models: `ac.py` (Phase 1), later `transient.py`, `alpha.py`. |
| `temper/agents/` | `baselines.py` (TWAP, AC schedule wrapped as policies) + `ppo/` (single-file, CleanRL-derived, §5). |
| `temper/eval/`   | Policy-agnostic Monte-Carlo harness, metrics, figure/report generation.   |
| `configs/`     | One committed config per experiment/figure (the Crucible committed-results pattern). |
| `tests/`       | pytest; `tests/golden/vendor/` holds FrontierView-generated fixtures, provenance-stamped. |
| `results/`     | Committed metrics JSON + figures, each carrying config hash + git rev.     |
| `client/`      | (M6) the Anvil participant — the only code that touches a network.         |
| `tools/`       | Dev scripts (plotting, sweep drivers).                                     |
| `docs/`        | Milestone briefs (`docs/briefs/`) and vendored protocol/golden provenance notes (`docs/vendor/`). This file and `ROADMAP.md` live at the repo root. |

## 4. The environment & objective contract (normative)

Until code exists this section is the source of truth; from M1 onward
`temper/env/execution_env.py` is, and must stay in sync with this intent.

- **Episode.** Liquidate (sell) a parent order of `X` shares over horizon `T` on a fixed
  grid of `N` intervals, `τ = T/N`. The grid matches the vendored golden schedules — the
  goldens, not this document, are the numeric spec.
- **Action.** Shares to execute this interval: continuous, clipped to `[0, remaining]`.
  The env enforces the boundary condition `x_N = 0`: any remainder at the final step is
  force-liquidated and charged like any other trade — matching the closed form's terminal
  constraint rather than letting the agent dodge it.
- **Dynamics (Phase 1).** Arithmetic Brownian motion with volatility σ; linear permanent
  (γ) and temporary (η) impact plus the fixed spread term, with parameter values exactly as
  encoded in the vendored FrontierView goldens (`SYMBOL_PARAMS`, calibrated η/γ). Phase-2
  models are additive alternatives behind the same interface, never silent modifications of
  Phase 1.
- **Observation (Phase 1).** `(time remaining fraction, inventory remaining fraction)` —
  deliberately minimal. Rediscovery must not smuggle in signal; richer observations arrive
  only with the phases that need them, per brief.
- **Reward (frozen).** Per step:
  `r_k = −(execution shortfall of trade k vs arrival) − λ σ² τ x_k²`.
  The running quadratic-inventory penalty is the mean–quadratic-variation form of risk
  aversion. Under Phase-1 dynamics the optimum of this objective over adapted policies is
  the deterministic AC sinh trajectory — that identity is what makes "rediscovery within ε"
  a well-posed claim rather than a vibe. The objective is encoded **once** and shared
  verbatim by the training reward, the evaluation metric, and the oracle (§6.7).
- **Determinism.** The env is seeded; identical `(config, seed)` produces an identical
  trajectory. Train and eval seed pools are disjoint by construction (§6.5).
- **Sign convention.** v1 is sell-side only; the buy side is a mirror and out of scope (§8).
- **Amended by M0.** The three 2026-08-04 entries in §9 refine this section against
  what the vendored goldens turned out to contain: which κ the "AC sinh trajectory"
  above refers to, that the linear temporary coefficient is the tangent η̃ rather than
  η, and the canonical grid and units. Read them before implementing against §4.

## 5. Agent & training

- **PPO, single-file, boring on purpose.** Adapted from CleanRL with attribution, not
  imported from a framework: the point of the project is training something explainable
  line-by-line. The adaptation is validated by a convergence smoke test on a standard
  control task in `tests/` before it is ever pointed at a Temper env.
- **Small MLPs, CPU training.** Sized so any milestone config trains on the reference box
  (Ryzen 7 3800X, the Crucible/Anvil machine) within an evening (§6.9).
- **Baselines are policies, not special cases.** TWAP and the AC schedule execute through
  the same env and the same eval harness as the agent — anything exposing `act(obs)` is
  gradable, and all graded things run through identical code.
- **Multi-seed by default.** ≥5 training seeds per reported experiment; report median and
  IQR (or bootstrap CI). No single-run numbers anywhere (§6.4).

## 6. Invariants (frozen — do not refactor through these)

1. **Every reported number regenerates from a committed config + seed.** Entries in
   `results/` carry the config hash and git rev that produced them.
   *Why:* multi-session agentic work converges only when red/green is objective; an
   evidence-based portfolio claim must be reproducible on demand.
2. **The oracle is normative.** `temper/oracle` must match the vendored FrontierView
   goldens within the stated tolerance, and the agent is graded against the oracle — the
   success criterion never migrates toward whatever the agent happens to do.
   *Why:* the differential-oracle pattern; prevents quiet redefinition of success.
3. **Success thresholds are pre-stated.** ε, seed counts, and CI levels are fixed in the
   milestone brief *before* training runs, and changed only by amending the brief before
   work starts. *Why:* post-hoc thresholds are how RL projects lie to their authors.
4. **No metric without baselines and dispersion.** TWAP and AC at matched λ appear on every
   chart and table; ≥5 seeds; distributional reporting.
   *Why:* single-run RL numbers are noise — the same ethos as Anvil's latency
   distributions, where the tail is the story.
5. **Train/eval separation.** Disjoint seed pools; Phase-2+ claims are made only on
   held-out seeds/configs. *Why:* agents overfit simulators; the honesty of the object
   depends on it.
6. **No environment feature without an independent expectation test.** Any market-model
   change ships with a check that a *fixed* policy's simulated moments match an analytic or
   independently computed value. *Why:* environment bugs masquerade as agent skill — the
   classic RL failure mode, and the one this project is structured to catch.
7. **One objective, encoded once.** The training reward, the evaluation metric, and the
   oracle optimise the same functional; any change is a §9 amendment.
   *Why:* silent objective drift voids every cross-experiment comparison and the
   rediscovery claim itself.
8. **`temper/` performs no network I/O.** The Anvil participant lives in `client/`,
   consuming the package. *Why:* the engine-seam pattern inherited from Anvil and
   DepthCharge; the core stays replayable and testable on any host.
9. **CPU is sufficient.** Milestone configs must train on the reference box in an evening;
   a GPU may only ever accelerate, never unblock.
   *Why:* keeps sessions unblocked, milestones honestly sized, and the project portable.

## 7. Decisions already made (with rationale)

- **Python + PyTorch** — a deliberate departure from the C++ portfolio. The oracle and its
  goldens come from FrontierView's Python compute core; the RL ecosystem is Python; the
  C++ story is already told by Anvil and Crucible. A C++ inference leg (ONNX export, maybe
  a Crucible post) is backlog, not v1.
- **Separate repo.** The boundary is versioned artefacts, not shared code — FrontierView
  goldens vendored under `tests/golden/vendor/` with provenance (source commit, generation
  date, parameter set); Anvil's `PROTOCOL.md` snapshot vendored when M6 starts. One repo
  per deploy, per the portfolio convention; Temper deploys nothing in v1 and is portalled
  from the portfolio site like DepthCharge.
- **Zero upstream changes.** Goldens are produced by a read-only export script run in the
  FrontierView repo; Anvil is consumed through its existing public contract. Neither
  property changes for Temper's benefit.
- **PPO first.** Stable, on-policy, CPU-friendly, and the right shape for finite-horizon
  episodic control. Algorithm comparisons (SAC, behaviour-cloning warm starts) are backlog.
- **Rate-control action space; market-order execution at the impacted price.** Keeps v1
  directly AC-comparable. Limit-order placement and queue-position micro-decisions are a
  different (larger) project — explicitly v2+.
- **Floats, not integer ticks.** Deliberate non-inheritance of DepthCharge invariant 3: the
  simulator's dynamics are analytic, FrontierView's core is float, and exactness here is
  enforced by goldens + tolerances rather than integer arithmetic. Do not "fix" this.
- **Phase-1 observation is minimal.** `(t, inventory)` only — the rediscovery claim is
  meaningful only if the agent could not have learned anything else.
- **The live-Anvil leg is a demo, not an evaluation.** Feeder flow is synthetic and
  non-adversarial; performance claims live in the simulator. M6 demonstrates that the
  policy speaks a versioned venue wire end-to-end, nothing more — and says so.
- **doctest-equivalent discipline via pytest** — familiar red/green convergence for
  sessions; `make test` from a clean clone is the gate, mirroring every other property.

## 8. Deliberately unspecified / out of scope

Unspecified on purpose (sessions decide, briefs record): network widths and PPO
hyperparameters, Phase-2 observation features, the plotting stack, the config schema's
details, file decomposition below the four sub-packages.

Out of scope for v1: buy-side episodes (trivial mirror), multi-asset portfolios, real
historical data beyond FrontierView's calibrated synthetic parameters, limit-order
placement, GPU dependence, any venue other than Anvil, and any suggestion of live-capital
trading.

## 9. Amendment log

| Date       | Change                | Why |
| ---------- | --------------------- | --- |
| 2026-07-31 | Initial constitution. | —   |
| 2026-08-04 | **The oracle carries two Almgren–Chriss decay rates, not one.** `oracle.ac_kappa` reproduces FrontierView's `κ² = λ σ_bin²/η̃` and is what the vendored goldens pin. `oracle.optimal_kappa` solves the discrete stationarity condition `cosh(κτ) = 1 + μ/2` for the §4 objective and is what M2 onward grades the agent against. §4's "the optimum of this objective is the deterministic AC sinh trajectory" is unchanged and still exact — it now names `optimal_kappa`'s trajectory specifically. | M0 found FrontierView's κ to be a continuum-limit expression that drops a factor of `X·τ/10⁴`; the two rates differ by ~2.2× at realistic order sizes. Both trajectories are sinh, so this is invisible by eye. Grading rediscovery against the vendored κ would have made a correctly-trained agent score up to ~18 % *better* than the "optimum" on the frozen objective (AAPL, 100 k shares, λ = 1e-4) — reading as "the agent beats Almgren–Chriss inside AC's own assumptions", which §1.1 names a red flag rather than a result. Keeping both satisfies invariant 2 (the oracle still matches the goldens exactly) without making the rediscovery claim unfalsifiable. |
| 2026-08-04 | **Phase-1 temporary impact is linear at the tangent `η̃`, not at `η`.** §4's "linear temporary (η) impact" reads as `η̃ · v`, the tangent to FrontierView's power law at the order's own TWAP participation rate. `oracle.cost_moments` charges the power law (goldens); `oracle.linear_cost_moments` charges the tangent and is the frozen objective of §6.7. | FrontierView's temporary impact is a 0.6-power law, which admits no sinh closed form; it linearises only to derive the schedule. §4 assumed a natively linear model. Naming the tangent explicitly keeps invariant 7's "one objective, encoded once" true, and keeps the M1 env, the M2 reward and the oracle optimising the same functional. |
| 2026-08-04 | **The canonical grid is FrontierView's, in hours.** T = 6.5 trading hours, half-hour bins, `N = max(2, round(2T))` = 13; costs in bps of notional, variance in bps². | §4 defers to the goldens for the numeric spec and they arrive in these units. Recorded here because it is the first thing every later milestone needs. |
