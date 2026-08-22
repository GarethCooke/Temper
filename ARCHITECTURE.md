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
| `temper/oracle/` | AC closed forms: κ solve, sinh trajectory, E[cost], V[cost], frontier; TWAP moments. Must match the vendored goldens (§6.2). From M4a also `powerlaw.py`: the vendored power-law world's optimum, which has no closed form and is *solved* — two independent solvers, certified in `tests/test_power_law_certificate.py`. |
| `temper/env/`    | `ExecutionEnv` (§4) — **one** env and one `step` loop — plus `impact.py`, the temporary-impact models it is handed: `LinearTemporary` (Phase 1's tangent) and `PowerLawTemporary` (M4a's vendored 0.6-power law), each declaring the cost encoding it charges. Later models arrive here, never as a second loop. |
| `temper/agents/` | `baselines.py` (TWAP, both AC schedules wrapped as policies), `ppo.py` (single-file, CleanRL-derived, §5), `execution.py` (the fraction-of-remaining action, the fixed reward scale, a trained net as a gradable policy). No module here may name the env's shock key. |
| `temper/eval/`   | The two routes to a number and the rule for which is legitimate: `rollout.py` (Monte-Carlo, for testing the *simulator*) and `grading.py` (analytic, the only route by which an *agent* is graded — see §9). Plus `reference.py` (oracle-only tables and the derived bands a milestone's thresholds come from), `metrics.py` (the graded/context registries), `variate.py`, `experiment.py`, `sweep.py`, and `figures.py` — the one module that may import matplotlib. |
| `configs/`     | One committed config per experiment/figure (the Crucible committed-results pattern). |
| `tests/`       | pytest; `tests/golden/vendor/` holds FrontierView-generated fixtures, provenance-stamped. |
| `results/`     | Committed metrics JSON + figures, each carrying config hash + git rev — and, from M6's prerequisite, the exported policy `.npz` beside the sweep it was selected from (§9 *A trained policy is a committed artefact…*). |
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
- **Amended by M0.** Three §9 entries refine this section against what the vendored goldens
  turned out to contain: *The oracle carries two Almgren–Chriss decay rates, not one*
  (which κ the "AC sinh trajectory" above refers to), *Phase-1 temporary impact is linear
  at the tangent η̃, not at η*, and *The canonical grid is FrontierView's, in hours*. Read
  all three before implementing against §4.
- **Amended by M1a.** The shock is charged *before* each bin, so `x_k` in the reward and in
  `Σx_k²` is inventory **before** bin `k`, with `x_0 = X` included and `x_N` excluded. See
  the §9 entry *The shock lands before the bin executes, so the shortfall variance sums
  inventory before each bin*; the index convention is normative, pinned by
  `shortfall_variance_bps2` against the vendored goldens to 9.5e-16.
- **Amended by M4a.** "Phase-2 models are additive alternatives behind the same interface"
  is now mechanical rather than a convention. Temporary impact is an *injected* model
  (`temper/env/impact.py`) and the env republishes the cost functional it charges as
  `ExecutionEnv.cost_encoding`; the default is Phase 1's linear tangent in the env, the
  experiment loader and the reference table alike, so **no config can inherit a Phase-2
  world by omission** — it has to name one, and `tests/test_repo_invariants.py` checks that
  it did. There is still exactly one `step` loop: a subclass with a duplicated loop would
  make invariant 6's `step_count` claim unfalsifiable. Read the two §9 entries *A metric
  grades the world that charges it* and *The power law's break is in a shock-free term…*
  before implementing against this section.

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
- **Amended by M3.** *Antithetic pairing is the Phase-1 variance-reduction regime, and
  at this reward magnitude it is bitwise the control variate* — M4 inherits an estimator
  that survives a non-affine cost, and inherits the two per-step assertions that say when
  it has stopped being exact.
- **Amended by M4a.** That inheritance was measured, and it is stronger than M3 predicted:
  the pairing is still **exact** in the power-law world, because the break is in a term that
  carries no shock. The two per-step assertions earned their keep immediately — see the §9
  entry *The power law's break is in a shock-free term…* for the mirror env that was quietly
  charging the wrong world until they caught it. M4a also changes what a tolerance is a
  fraction *of* (*The tolerance's denominator is the available advantage…*); any milestone
  grading an agent against a closed form the world does not solve must read that first, or
  it will state a bar twice the size of the effect it is measuring.
- **Amended by M2.** Four §9 entries. Two change how an agent is graded and what it is
  trained on, and any later training milestone must read them first — M3 sweeps λ and
  inherits both: *A deterministic policy on a price-free observation is graded analytically,
  not by Monte Carlo*, and *Phase-1 rediscovery trains on the noise-free reward; sampled
  rewards do not resolve the objective*. Two are housekeeping with teeth: *matplotlib is
  pinned and confined to `temper/eval/figures.py`* and *Invariant 5 is enforced per module,
  not by a blanket ban*.

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

Entries here are *structural decisions about Temper*. Portable practice — a rule that
would apply just as well to Anvil or Crucible — goes in `docs/house-notes.md` instead, and
is cited from a brief by title the same way.

**Cite entries by title, not by date.** The `Date` column records the *session* date, not a
commit date, and several entries legitimately share one — M0, M1 and M1a all closed on
2026-08-04, and five entries carry it. Titles are unique; dates are not, so every pointer
elsewhere in the repo names the entry it means. A session reconciling this table against
`git log` will find more entries on a day than that day has commits: that is the sessions
and the commits keeping different time, not drift to be "corrected".

| Date       | Change                | Why |
| ---------- | --------------------- | --- |
| 2026-07-31 | Initial constitution. | —   |
| 2026-08-04 | **The oracle carries two Almgren–Chriss decay rates, not one.** `oracle.ac_kappa` reproduces FrontierView's `κ² = λ σ_bin²/η̃` and is what the vendored goldens pin. `oracle.optimal_kappa` solves the discrete stationarity condition `cosh(κτ) = 1 + μ/2` for the §4 objective and is what M2 onward grades the agent against. §4's "the optimum of this objective is the deterministic AC sinh trajectory" is unchanged and still exact — it now names `optimal_kappa`'s trajectory specifically. | M0 found FrontierView's κ to be a continuum-limit expression that drops a factor of `X·τ/10⁴`; the two rates differ by ~2.2× at realistic order sizes. Both trajectories are sinh, so this is invisible by eye. Grading rediscovery against the vendored κ would have made a correctly-trained agent score up to ~18 % *better* than the "optimum" on the frozen objective (AAPL, 100 k shares, λ = 1e-4) — reading as "the agent beats Almgren–Chriss inside AC's own assumptions", which §1.1 names a red flag rather than a result. Keeping both satisfies invariant 2 (the oracle still matches the goldens exactly) without making the rediscovery claim unfalsifiable. **Restated from M3's sweep, which measured it at nine λ:** the difference is a *displacement along the frontier*, not a worse schedule. `ac_trajectory` and `optimal_trajectory` are the same `sinh_trajectory` differing only in κ, so there is a λ′ solving `optimal_kappa(λ′) = ac_kappa(λ)` — and at every grid point the two trajectories then agree to 7.3e−11 shares in 100,000, i.e. float round-off. The vendored schedule is therefore the *exactly optimal* schedule for a different risk aversion, and its (E, V) point lies **on** the optimal frontier, never inside it; describing it as dominated would be wrong. The displacement factor is `c = λ′/λ → X·τ/10⁴ = 5.0` at this case in the low-κ limit — the same factor this entry names, with √5 = 2.236 the "~2.2× in κ" above — and it is not constant: 5.01 at 10^−5, 5.34 at M2's 10^−3.5, 29.3 at 10^−2, 147,000 at 10^−1, because `optimal_kappa` grows like `log μ` once `μ` is large while `ac_kappa` keeps growing as `√λ`. Two consequences. The economic reading of this finding is **"the risk-aversion dial is mislabelled by ~5× at usable λ"** rather than "AC is suboptimal", which is a much narrower and more defensible claim about a vendored library. And the single number M2 reported for AC (0.3435 of the TWAP gap at 10^−3.5) is a point on a curve that spans 8.2942 to 0.0001 across the M3 grid, so it should never be quoted as *the* cost of the vendored κ. |
| 2026-08-04 | **Phase-1 temporary impact is linear at the tangent `η̃`, not at `η`.** §4's "linear temporary (η) impact" reads as `η̃ · v`, the tangent to FrontierView's power law at the order's own TWAP participation rate. `oracle.cost_moments` charges the power law (goldens); `oracle.linear_cost_moments` charges the tangent and is the frozen objective of §6.7. | FrontierView's temporary impact is a 0.6-power law, which admits no sinh closed form; it linearises only to derive the schedule. §4 assumed a natively linear model. Naming the tangent explicitly keeps invariant 7's "one objective, encoded once" true, and keeps the M1 env, the M2 reward and the oracle optimising the same functional. |
| 2026-08-04 | **The canonical grid is FrontierView's, in hours.** T = 6.5 trading hours, half-hour bins, `N = max(2, round(2T))` = 13; costs in bps of notional, variance in bps². | §4 defers to the goldens for the numeric spec and they arrive in these units. Recorded here because it is the first thing every later milestone needs. |
| 2026-08-04 | **The shock lands *before* the bin executes, so the shortfall variance sums inventory before each bin.** `V = σ_bin² · Σ_{k=0}^{N−1} (x_k/X)²` — `N` terms, the first of them the whole order (`x_0 = X`). This is *not* the textbook Almgren–Chriss form `σ²τ · Σ_{k=1}^{N−1} x_k²` over post-bin holdings, and §4's "arithmetic Brownian motion with volatility σ" now names which of the two it means. The same ordering fixes the sign and the index range of the exact noise functional `C − E[cost] = −σ_bin · Σ_{k=0}^{N−1} (x_k/X) ξ_k` that M1a pins per episode. | §4 defers to the goldens for the numeric spec, and FrontierView charges the shock before the bin — `oracle.shortfall_variance_bps2` reproduces its variance to 9.5e-16, so invariant 2 settles it. The two conventions are not close: `Σ_{k=0}^{N−1}(x_k/X)² − Σ_{k=1}^{N}(x_k/X)² = 1` identically, for every schedule, which is 20.6 % of V for TWAP at N = 13 — and is exactly the off-by-one-in-`Σx_k²` class M0 flagged and the deep tier was sized to detect. Recorded because the ambiguity is invisible in code (both are one-line sums over a trajectory) and because M1a's own brief stated the functional in the post-bin convention: writing the exact identity to that formula would have been red against a correct env, and the resulting "finding" would have been the document. **Propagation:** the extra term is `λσ²τX²` — inventory before bin 0 is `X` for every schedule, so it is a constant in the decision variables, not a function of them. The interior Hessian is therefore identical under both conventions and `optimal_*` does not move: `cosh(κτ) = 1 + μ/2`, `optimal_kappa`, the λ-rescaling test and the task-0 certificate are all untouched. That task 0 stayed green through a 20.6 % change in `V` is the evidence, not a coincidence. Economically the same statement: the shock lands before the first trade can be placed, so one bin of volatility on the full position is unavoidable and cannot be scheduled away. **Consequence for the frontier:** `V` has a hard floor at `σ_bin²X²` — immediate liquidation gives `Σ(x_k/X)² = 1` exactly, not 0 — so as λ → ∞ the frontier approaches that floor rather than the origin. The textbook AC picture, where risk vanishes at instantaneous execution, is qualitatively wrong under this convention. |
| 2026-08-06 | **A deterministic policy on a price-free observation is graded analytically, not by Monte Carlo.** §2's diagram and §3's "policy-agnostic Monte-Carlo harness" now name two distinct routes to a number, and which one is legitimate depends on what is being tested. `temper/eval/rollout.sample_costs` estimates by sampling and is right for M1, whose subject is the *simulator*. `temper/eval/grading` computes exactly — one deterministic rollout, then `schedule_moments` on the schedule it realised — and is the only route by which an *agent* may be graded from M2 onward. Validity is asserted, not assumed: `deterministic_schedule` rolls the policy out on two unrelated shock streams and requires the trajectories to be bitwise equal. | The observation is `(time left, inventory left)` and carries no price; inventory evolves purely from actions. A deterministic policy therefore induces an open-loop schedule, whose moments are a closed form. That is not a convenience. At M2's case the objective is ~2.4 bps while the per-episode cost SD is ~95 bps, so resolving the milestone's ε = 0.066 bps by sampling needs on the order of 10⁷ episodes *per seed* — and would still report an interval where the analytic route reports a number. The bitwise assertion is what makes the shortcut sound rather than merely fast: it fails loudly the moment price reaches the observation, which is also the moment analytic grading would silently start lying. |
| 2026-08-06 | **Phase-1 rediscovery trains on the noise-free reward; sampled rewards do not resolve the objective.** M2's headline agent is trained on the deterministic reward — the realised reward with M1a's exact noise identity `C − E[cost] = −Σ_k (n_k/X)·walk_k` subtracted per step (`temper/eval/variate.py`). Under Phase-1 certainty equivalence the optimal policy is unchanged, so *what* is rediscovered does not move; what moves is the claim, from "RL under noise recovers AC" to "RL optimises a deterministic function and recovers AC". Every result carries the weaker sentence verbatim (`estimator.claim` in the config, copied into `results/` and the figure caption). The variate lives under `temper/eval/` and not beside the training loop because it reads the env's shock key, which M1a's static guard forbids anywhere under `temper/agents/`: it is an estimator, not a policy, and the seam stays visible. | Measured on five seeds, not extrapolated from one. Vanilla PPO on sampled rewards at the full 12 M-step per-seed budget scores 0.066, 0.009, 0.819, 0.098, 0.147 of the TWAP gap: median 0.098 against an ε of 0.05, and a worst seed at 0.819 against a per-seed floor of 0.10. It fails both halves of the pre-stated bar. The shape of the failure is the finding and it is **not** a plateau — one seed in five essentially rediscovers the sinh (0.009, inside ε) while another barely learns at all (0.819, ‖δ‖₂ larger than the parent order) and finishes near TWAP. At this SNR, training is a lottery rather than a slow climb. The identical agent on the noise-free reward, same hyperparameters, clears the bar with a spread orders of magnitude tighter (`results/m2_rediscovery.json`). Both runs are committed (`configs/m2_ppo{,_sampled}.yaml`) and differ in this one field, which is what makes "the variate closed the gap" a measurement. An earlier single-seed reading of this same question was badly misleading — the same seed *address* scored 0.165, 0.118 and 0.066 across runs differing only in torch's thread count — and the brief records that retraction in full; treat any single-run number on this env as noise, which is what invariant 4 says anyway. **Consequence for M3 and beyond:** every λ on the frontier faces the same ~1:70 per-episode ratio, and the high-λ end is worse — so a λ-sweep that reverted to sampled rewards would trace a frontier whose scatter is dominated by which seeds happened to win their lottery, and would read as "RL degrades away from the reference λ". The sampled-reward sweep is the control M3 reports against, not an attempt to repeat. |
| 2026-08-06 | **matplotlib is pinned and confined to `temper/eval/figures.py`.** §8 left the plotting stack deliberately unspecified; M2 needs a figure, so it is specified: `matplotlib==3.11.1` in `requirements.txt`, the `Agg` backend forced before `pyplot` is imported, and every other module under `temper/` forbidden to import it. `tests/test_repo_invariants.py` enforces the allow-list — which is one file long — and the backend ordering, which is otherwise invisible until a headless host blocks on a display. | The core is on the import path of every test, every training run and eventually the M6 Anvil client; a rendering stack arrives there one convenience import at a time. Recorded rather than left to taste because the confinement is the decision, not the library — a later session may swap the library, and must not swap the boundary. |
| 2026-08-06 | **`git_dirty` asks whether the *source* is uncommitted, and a result is produced from a committed tree or it is not an acceptance artefact.** `temper/eval/provenance.py` ignores `results/` when deciding dirtiness, and the stamp is taken at the *start* of a run rather than the end (`temper/eval/sweep.py`). `tests/test_m2_rediscovery.py` requires `git_dirty: false` on every committed result, and the driver warns before spending hours on a run that cannot satisfy it. Relatedly: the torch intra-op thread count is now a committed hyperparameter (`ppo.torch_threads`), not a property of the host. | Invariant 1 says every reported number regenerates from a committed config and seed. A result stamped `git_dirty: true` names a revision that does *not* contain the code that produced it, so it regenerates from nothing — the invariant failing quietly, which is worse than failing loudly. Two refinements make the flag trustworthy enough to gate on: dirtiness excludes `results/`, because otherwise a sweep that writes its own artefact makes the *next* sweep report dirty for a source tree nobody touched; and the stamp is taken before training, because a two-hour run should be attributed to the tree it started from, not to whatever the repo looked like when it finished. The thread pin closes the other half: torch's CPU reductions sum in a thread-count-dependent order, PPO compounds that over ~1 800 updates, and M2 measured the same seed address landing at 0.165 and 0.066 of the TWAP gap on four threads versus eight. Unpinned, "same config, same seed" reproduces only on a host with the same core count — invariant 1 holding by luck. |
| 2026-08-06 | **Invariant 5 is enforced per module, not by a blanket ban.** Until M2 the seed-pool guard was "nothing in the test path may open `train` or `eval`", which was checkable without knowing who was asking. M2 legitimately trains out of `train` and evaluates out of `eval`, so `tests/conftest.py` now attributes every stream the env opens to the module that opened it and checks it against a per-module allow-list of *pools*. A fourth pool, `m2/diagnostic`, joins `m1/differential` for checks that report no number. | The replacement is strictly stronger than what it replaced. The flat rule could never have caught the failure invariant 5 is actually about — M2's evaluation grading on a stream it trained on — because both pools were forbidden to everyone equally. Only one module is granted both pools (the sweep regeneration, which by definition needs both), and `tests/test_seed_pool_discipline.py` asserts that too. M1's original property survives verbatim for the modules it was written about. |
| 2026-08-16 | **Antithetic pairing is the Phase-1 variance-reduction regime, and at this reward magnitude it is bitwise the control variate.** §5's *Amended by M2* named the control variate as what Phase-1 rediscovery trains on; from M3 the default is `temper/eval/antithetic.py`. Each episode runs as (ξ, −ξ): a mirror `ExecutionEnv` at the *same* seed address whose generator draws are the exact elementwise negation of the primary's, stepped in lockstep with the same action, the two realised rewards averaged. `estimator.regime` selects between `sampled`, `control_variate` and `antithetic`; none is withdrawn. The estimator seam is unchanged — it lives under `temper/eval/`, names the env's shock key only to *assert* the negation, and reaches `train` as an env-factory parameter, so `temper/agents/` still has no route to the price path. | The variate subtracts M1a's analytic noise identity. That is exact, and it **does not exist in Phase 2**, where cost stops being affine in the shocks; the pairing needs only the ability to replay with negated draws, so it degrades to partial cancellation rather than vanishing, and M4 inherits an estimator instead of a cliff. Validated before use, not argued: ten seeds at M2's λ give a median gap fraction of 0.000168 against the variate's committed 0.000204 and a gate of 0.002, with the realised per-update reward variance measured *inside* the run under the agent's own actions — 3,377 bps² on the sampled half against 3.4e−08 averaged, a ratio of 1.0e−11. The stronger half of the finding is that on the five seeds the two runs share an address, the trained policies agree **bitwise** — objective to seventeen digits and the whole trajectory — because the two estimators' rewards differ by ~1e−17 bps while PPO's rollout buffers are float32 (`AGENT_DTYPE`), so the optimiser is handed identical numbers. That equality is a property of this reward magnitude and this dtype, not a theorem: it will break in Phase 2, which is why the pairing asserts its two assumptions per step in the wrapper (`PairDiverged`) rather than trusting them — bitwise-identical observations and trades across the halves, and a mirror shock that is the exact negation. When Phase 2 enriches the observation the action-identity check goes red, and that is correct: it is the signal that the pairing's exactness has lapsed. |
| 2026-08-17 | **A sweep point's λ is verified against a pre-stated sub-grid; only a single-point experiment is verified against task 0's selection rule.** `Experiment.verify_lambda_rule` gains a branch: a config naming a `frontier_grid` must name a grid that is a subset of the committed λ grid, that *contains* the rule-selected λ, and that contains the config's own λ. `FRONTIER_GRIDS` holds the grids by name — M3's is the nine half-decade points 10^−5 … 10^−1, taken **by index** from `VENDOR_LAMBDA_GRID` so each λ is the reference table's float exactly. | M2's rule fixes *one* λ and refuses every other, which is what stops a milestone λ being chosen after seeing a curve. A frontier must visit nine, so the rule as written would have to be bypassed — and a bypassed check is not a check. Requiring the sub-grid to contain the rule-selected point preserves the property that matters: the set of λ is fixed in a committed file before the sweep, and every sweep carries a point directly comparable to a committed result (here 10^−3.5, which the sweep runs *first* so an amended update budget is checked against a known answer after one point rather than nine). Sweep points are generated from a template by `tools/m3_frontier.py configs`, and `tests/test_m3_frontier.py` asserts the committed point configs are byte-identical to the generator's output, so a point cannot drift from the template it claims to be. |
| 2026-08-18 | **The per-λ tolerance is meaningful only where the testbed is discriminative, and the frontier measures where that stops being true.** ε is 5 % of *that λ's* `(J_twap − J_optimal)/J_optimal`, which is portable by construction — and degenerate wherever TWAP and the optimum have nearly converged. Reported results state the absolute excess over `J_optimal` beside the gap fraction, and the frontier figure's lower panel draws the gap fraction against λ with ε across it, so the reader can see which points the tolerance can speak about. | Measured across four decades. The agent's median excess over the certified optimum stays between +0.004 % and +0.33 % at every λ on the grid, while the gap fraction moves from 0.070 to 0.00002 — three and a half orders of magnitude — because the TWAP gap it is normalised by moves from 0.44 % to 374 %. The one ε miss in the sweep (10^−5, median 0.070) is therefore a statement about the denominator: the whole TWAP gap there is 0.0041 bps, so ε is 2 micro-bps and the agent is 2.9 micro-bps from the optimum, inside its derived trajectory band. M2 task 0's condition (i) *predicted* this — it rejected every λ with a TWAP gap below 20 % as non-discriminative — and the sweep visits those λ anyway because the frontier's shape needs its low end. Two consequences worth inheriting: dispersion tracks the objective's curvature rather than the optimiser's luck (the seed IQR falls 0.063 → 0.00003 as κT rises 0.92 → 5.14, exactly as the derived band widens from 3 % of X to 29 %), and a milestone that reports only a gap fraction will look like it degrades at low λ when nothing has degraded. |
| 2026-08-19 | **The power law's break is in a shock-free term, so the noise identity and the antithetic pairing survive it exactly.** Narrows the antithetic entry's "does not exist in Phase 2, where cost stops being affine in the shocks". Temporary impact is a function of the *schedule*; it carries no shock. Realised cost under the power law is still `C = f(x) − σ_bin·Σ_k (x_k/X) ξ_k` with only `f` changed, so it is still affine in the draws — and M1a's exact per-episode noise identity, the pair's exact cancellation, its action-identity assertion and `deterministic_schedule`'s open-loop check all hold verbatim. What ends exactness is **a second, independent noise source or a price-bearing observation**, not curvature in the cost. M4b is where that happens; M4a is not. | Measured before M4a's training point rather than argued (`tests/test_m4a_inherited_guarantees.py`, `make m4a-guarantees`), on twelve (case, schedule) cells: noise identity worst 2.3e-14 relative against a 1e-12 bar, antithetic cancellation worst 1.8e-15 bps per step against 1e-12, action identity and the open-loop check exact. **And it earned its keep on the first run.** All twelve cancellation cells were red by 0.06 bps per step — four orders outside the band — because `mirror_of` rebuilt the mirror env without the primary's temporary-impact model. It defaulted, so the antithetic *mirror* was a Phase-1 env being averaged against a power-law primary: the rewards still looked like rewards, the schedules were still identical, and the estimator was silently no longer the one the config named. That is exactly the class of defect this check is placed before training to catch, and catching it cost minutes instead of a night. The general lesson, which M4b should inherit: when a per-episode property is injected into the env, every env the estimator constructs has to be handed it, and the check that finds the one that was missed is a per-step identity rather than a training curve. |
| 2026-08-19 | **The tolerance's denominator is the available advantage, not the TWAP gap, wherever the closed form is the thing being beaten.** Successor to *The per-λ tolerance is meaningful only where the testbed is discriminative…*. `Tolerances` now carries a `denominator` — `twap_gap` (M2, M3) or `available_advantage` (M4a) — and the verdict is read on the matching `Grade` field, so a config states which question it is answering. M4a's headline is the **capture fraction** `c = (J_opt(tangent) − J_agent) / (J_opt(tangent) − J_pow*)`, and the **absolute excess in bps travels beside it everywhere**, in the results file, the driver's output, the figure caption and every assertion. | The predecessor entry said the denominator is the thing to watch; this is that arriving as a hard constraint. At M4a's λ the whole available advantage is **0.03674 bps**, while 5 % of that λ's TWAP gap is **0.06628 bps** as M3 actually computed it and **0.07433 bps** re-derived in the power-law encoding — **1.8×–2.0× the entire effect**. An agent held to M3's ε in the power-law world would pass while capturing *none* of the mis-specification, which is a tolerance that cannot distinguish the milestone's success from its complete failure. Two things generalise. First, a portable denominator is only portable within the question it was written for: "5 % of the distance TWAP covers" is the right unit for *rediscovery* and the wrong one for *advantage*, and the two differ by 36× here. Second, the direction of the reporting trap flips with the denominator. M3's entry exists because a gap fraction alone made a healthy agent look like it was degrading at low λ; here the risk runs the other way — a capture fraction near 1 on an advantage of 0.037 bps is a *small absolute claim*, and reporting only the fraction would make it sound like a large one. Hence both numbers, always, in that order. |
| 2026-08-19 | **A metric grades the world that charges it.** `temper/eval/metrics.py`'s registries are keyed by cost encoding: `GRADED[encoding]` holds the metrics that may score an env charging that encoding, `metrics_for` refuses any other world, and `temper/eval/grading.py` calls `check_grades_world` before it computes a number. The env states its half — `ExecutionEnv.cost_encoding`, read off the injected `temper/env/impact.py` model — and a `ReferenceRow` carries the world its optimum was solved in, so the schedule, the optimum and the metric travel as one. This supersedes `GRADEABLE_ENCODINGS = {LINEAR}` from the entry above. The quarantine is *generalised, not lifted*: `cost_moments` keeps its `CONTEXT` entries under the names M1, M2 and M3 report it by, so a Phase-1 result still quotes the vendored number beside its own and the grading path still has no route to it. | The flat rule was the right rule stated in the only way that was checkable when one world existed. M4a makes the power law a world, so the ban would have had to be bypassed — and a bypassed check is not a check. The replacement is strictly stronger, and concretely so: the flat rule could never have caught a **linear metric grading a power-law env**, because linear was permitted to everything equally, and that is now the live failure mode. Same shape as M2's replacement of the blanket seed-pool ban with a per-module allow-list (*Invariant 5 is enforced per module*): the check that was wide enough to be free was also wide enough to be blind. Two smaller consequences fell out. `temper/oracle/model.py` owns the encoding names, because `temper/env` and `temper/eval` both need them and neither may depend on the other. And the three metrics are registered *separately* per world rather than shared where they agree — `shortfall_variance` is bit-for-bit identical in both, since the shock model is untouched, and registering it twice makes that equality something `tests/test_objective_registry.py` measures rather than a fact about which function object two dictionary entries point at. It is also why M4a's differential needed a new analytic reference for E[cost] alone. |
| 2026-08-04 | **Phase 1 is the linearised world end-to-end; `cost_moments` is reporting context only.** Env dynamics, training reward, evaluation metric and oracle all charge linear temporary impact at the tangent η̃, encoded once as `oracle.schedule_moments` (= `linear_cost_moments` at `linearised_eta`). `oracle.cost_moments` — FrontierView's 0.6-power charge — may be *reported beside* a Phase-1 result and may never be one: `temper/eval/metrics.py` keeps two registries and `register_graded` refuses a power-law-encoded metric. This supersedes M0's note that M1 should source the env reward from `linear_cost_moments` and the eval metric from `cost_moments`. | M1 task 1 measured whether the two paths reduce to each other on the Phase-1 golden parameter sets: they differ by 12 %–54 % of expected cost (a power law against its own tangent agrees only where they touch). Splitting reward from metric across that gap would have violated invariant 7 outright — the agent would have been trained on one functional and graded on another, and every rediscovery and frontier claim would compare two different objectives. Quarantining rather than deleting `cost_moments` keeps invariant 2 intact (it is what the goldens pin) and keeps M4's assumption break one env swap away, where the power law becomes the world and a gradeable encoding by a further amendment. |
| 2026-08-20 | **The oracle stays independent, and the question is closed rather than carried.** §7's "separate repo, zero upstream changes" is reaffirmed with three milestones of evidence behind it; the ROADMAP's *Revisit the oracle↔FrontierView code-sharing decision* item is retired. Temper's oracle does not import FrontierView and will not. | Reopened as promised, and the number the item rested on no longer describes the repo. The M0 audit sized the parallel formula code at ~55 executable lines when the oracle was essentially a re-derivation of FrontierView. `temper/oracle/` is now ~1,080 non-blank lines across seven modules: thirteen functions reproduce vendored formulas, twenty-four have no upstream counterpart, and `powerlaw.py` alone — 564 lines of certified power-law optimum, two independent solvers, KKT and curvature machinery — has no FrontierView analogue at all. The shared surface is ~5 % of the module, so the trade on offer is a dependency that saves one line in twenty while destroying the property the other nineteen exist to provide. What the independence has bought is not hypothetical: without the second derivation M2 grades against `κ² = λσ²/η̃` and a correctly trained agent scores up to 18 % better than "optimal" — the exact failure §1.1 names a red flag, shipped. Drift, the standard objection, is already answered by a mechanism this repo built: the goldens pin `f87795f6` with provenance and the suite fails if the oracle moves off them, which is what §7 means by "the boundary is versioned artefacts, not shared code". What would reopen this: if FrontierView adopts the discrete κ, the two implementations converge and the parallel surface stops being a second derivation — at which point the argument above weakens and the question is worth asking again. |
| 2026-08-22 | **A trained policy is a committed artefact, selected by rule and verified like every other.** §3's `results/` row now also holds an exported policy. `temper/agents/checkpoint.py` writes a plain `.npz` — numpy-readable, `allow_pickle=False`, no torch anywhere on the read path — carrying the actor and critic parameters under the checkpoint's own layer names, the evaluation observations, actions and trajectory, and a provenance stamp that names the sweep it came from by content hash. `tools/train.py --export-checkpoint` produces it by retraining **one** named seed of a committed sweep, never by keeping a network a sweep happened to leave in memory; the seed is chosen by `temper.eval.grading.median_ordinal` applied to the committed objectives, and the export refuses to write a policy that misses that sweep's own per-seed bar or raises a red flag. `tests/test_policy_checkpoint.py` re-applies the selection rule, re-grades the stored schedule against the certified optimum, and rolls the rebuilt network out through the real env — none of it needing a training run. | Through M4a the project discarded every network it trained the moment it had been graded: `*.pt` and `*.ckpt` were gitignored, nothing called `state_dict`, and results carried trajectories, grades and provenance but never weights. That is enough to *check* a claim and not enough to *re-examine* one — no policy could be inspected after the fact, no reported schedule re-derived from the thing that produced it, and the backlog's C++/ONNX inference leg was impossible by construction. It surfaced as M6's prerequisite, but it is a repo gap rather than an M6 one, which is why it landed as its own change before that milestone started. Three choices carry the weight. **`.npz` rather than `torch.save`**, because a pickle is a program and requires the training stack to read: the format keeps invariant 8's seam honest by letting `client/` run the policy with numpy alone, and makes a second implementation of the forward pass a pinnable thing rather than a rewrite. **Selection by a committed rule**, because anything picked out of a sweep and then shipped is a choice that could flatter the artefact — the rule takes the upper of the two central ranks at even seed counts, so the tie-break can only ever cost, and the test re-derives it rather than trusting the metadata. **The evaluation arrays travel inside the archive**, because the alternative is a pin in a second file that can go missing, and a committed binary that nothing verifies is not an artefact this repo keeps. What the export does *not* claim is bitwise reproduction: retraining a seed reproduces its verdict, not its digits (`tests/test_m2_rediscovery.py` measured the same address landing at 0.165 and 0.066 of the TWAP gap at four threads versus eight), so whether it reproduced bitwise is recorded in the metadata and the assertions are stated on the bar the sweep was reported under. |
