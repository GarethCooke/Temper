# M1a — Acceptance hardening

**Track:** agentic · **Size:** one short session (+ one ≤10 min acceptance run) · **Reads
first:** `docs/briefs/M1-env-and-analytic-differential.md`, then `ARCHITECTURE.md` §4 and
invariants 3, 6, 7.

## Objective

M1 landed green on branch `m1-env-and-analytic-differential`. This brief closes the gaps
between "green" and "accepted", and does not flip the M1 roadmap row until it is green
itself. Three of the tasks convert statistical checks into exact ones; the rest close
guards the M1 session either did not reach or did not report.

## Context

M1's identity tests are stronger than the brief asked for. Test 5(b) pins the inventory
penalty against λ·V exactly; finding 2's per-episode expectation identity pins the
deterministic skeleton exactly. The consequence is that exactly **one** leg of the
differential still rests on statistics: whether the env's realised noise carries the right
variance. That leg is the one whose resolution the M1 session found had been mis-stated in
the parent brief (0.9 %/1.3 % were the N = 200,000 figures, not N = 100,000).

At N = 100,000 the named off-by-one-in-`Σx_k²` class, at its low end (~2 %), sits 4.47σ
against a 4σ gate — roughly two detections in three. N_sim has already been raised to
200,000, which puts it at 6.3σ (~99 %). Task 1 removes the reliance entirely.

## Preflight — invariant-3 bookkeeping (apply to the parent brief before any code)

Three pre-stated numbers moved. Record each in
`docs/briefs/M1-env-and-analytic-differential.md`, with reason, before starting work:

1. Deep tier N_sim 100,000 → **200,000**. Reason: restores the resolution the parent
   brief's own prose claimed, and the margin on the weakest named bug class. **Done —
   confirm the brief text, not just the YAML.**
2. Deep-tier resolution prose corrected to the N = 200,000 figures: mean shifts ≳ **0.89 %**
   of σ_C, variance mis-scalings ≳ **1.26 %**.
3. The identity-test tolerance denominator: `≤ 1e-10 relative` now reads *relative to the
   summed absolute terms*, not to the surviving total. Reason: `Σr = −(IS + λV)` cancels
   ~1e2 bps terms to ~1e-2 bps, so a total-relative verdict is seed-dependent. Worst
   observed 8.8e-12.

Item 3 generalises past Temper — cross-file "scale-relative tolerances for cancelling
identities" as a house convention alongside the Anvil/Crucible notes.

## Tasks

1. **Exact noise identity — retire the last statistical leg.** The env already publishes
   the per-bin shock (finding 2). Assert per episode that the realised cost minus the
   analytic `E[cost]` equals the noise functional `σ√τ · Σ_{k=1}^{N−1} x_k ξ_k` exactly,
   where `x_k` are the post-bin holdings of the schedule actually executed and `ξ_k` are
   the raw draws the env used.

   Anti-tautology conditions, all load-bearing: the functional is assembled **in the test**
   from §4 and the case parameters, importing only σ, τ and the schedule — never from an
   env internal or an oracle helper that already computes it; the sign follows §4's
   sell-side convention and is **stated in the test**, not inferred by matching the env.
   Run on every (case, schedule) cell plus the task-4 force-liquidated schedule.

   Why this is worth more than the N_sim bump: if the per-episode residual is exactly the
   right linear functional of the draws, `V = σ²τ Σ x_k²` holds **by construction**, not
   by sampling. The Monte-Carlo tier then confirms only that the draws are iid standard
   normal and uncorrelated across bins — belt-and-braces, not the gate.

2. **Step-count assertion — make "no vectorised side-channel" a test.** The env exposes a
   monotone step counter (or a test-only wrapper does). Per cell, assert the counter delta
   equals `N_sim × N_bins` exactly. Totals are pre-stated below. This permanently settles
   whether both tiers run the same code path, and stops a future session from
   accelerating the loop out of the contract.

3. **Observation-minimality guard — close the M2 leak before M2 exists.** §4: rediscovery
   must not smuggle in signal. Assert the observation space is exactly 2-dimensional and
   that `reset`/`step` return a length-2 observation; assert the shock is reachable only
   through `info`. Give the shock key a single module-level constant so
   `tests/test_repo_invariants.py` can statically reject that literal anywhere under
   `temper/agents/`, `temper/eval/rollout.py`, or any future training path — tests and
   `temper/env/` excepted.

4. **The M0 watch item, actually exercised.** M0's session notes: the
   `sinh-overflow-asymptote` branch leaves terminal inventory at `X·e^{−κT}` rather than a
   hard zero, and the env's force-liquidation must not assume otherwise. Determine whether
   any cell of the 3 × 3 golden grid reaches that branch. If none does — expected — add a
   dedicated **guard case** to `configs/m1_differential.yaml`, marked as a guard rather
   than a golden, whose parameters reach it. Feed the raw asymptote trajectory *including*
   its residual terminal holding through the env, and require that the realised
   (force-liquidated) schedule's cost matches `schedule_moments` of that realised
   schedule. Task 5(d)'s deliberate under-trader does not cover this: the watch item is a
   **named baseline** arriving with residual inventory, which is a different path.

5. **Seed-pool discipline.** M0 pins pool disjointness by construction; that is not the
   same assertion as the M1 harness *using* the diagnostic pool. Assert every M1
   Monte-Carlo draw resolves through `"m1/differential"`, and that the train and eval
   pools are untouched by the whole M1 test path.

6. **Task-0 confirmation.** The M1 report did not mention task 0. Confirm all four parts
   green and report them individually: (a) Cholesky succeeds; (b) generic solve matches
   `optimal_trajectory` ≤ 1e-12 relative to X on the 3 × 3 grid; (c) 200 random interior
   directions at **both** ‖δ‖ ∈ {1e-3, 1e-6}·X; (d) the monotonicity assert. Land
   whichever is missing. (d) is not decoration — it is what licenses dropping `ε·Σ|n_k|`
   from the gradient, and the certificate is unsound without it.

7. **Clean-clone verification, for real.** Finding 3 (`.gitignore`'s `env/` excluding
   `temper/env/`) is precisely a clean-clone bug, and cannot be verified from a working
   tree. Now that the branch is committed: clone the repo to a fresh directory, build a
   fresh venv from `requirements.txt`, run the suite. The **test count must match** the
   in-tree run — a lower count means files are still missing from the commit, which is the
   same bug class the guard was written for. Confirm explicitly that
   `configs/m1_differential.yaml`, the vendored goldens and the export script are all
   tracked.

8. **Documentation placement.**
   - The §9 invariant-7 entry carries the **magnitude**: the power-law and linear
     encodings differ by 12 %–54 % of expected cost on the Phase-1 golden sets. That number
     is what shows the split would have been a real defect rather than a stylistic one,
     and it is the entry's evidence.
   - The supersession note goes **at** the bolded sentence in
     `docs/briefs/M0-oracle-and-goldens.md` ("M1 must encode the env reward from
     `linear_cost_moments` and the eval metric from `cost_moments`"), not only at the head
     of the brief. That sentence is what a future session's grep lands on.
   - `ROADMAP.md`'s M1 row stays `☐` until this brief is green.

## Pre-stated numbers (invariant 3 — loosen only by amending this brief before work)

| Item | Value |
| --- | --- |
| Deep tier | 3 × 3 golden grid × 3 schedules = 27 cells, N_sim = 200,000 |
| Fast tier | unchanged — 3 cases × 3 schedules = 9 cells, N_sim = 20,000 |
| Canonical bins | N = 13 (§9, 2026-08-04) |
| Step count, deep tier | exactly 27 × 200,000 × 13 = **70,200,000** |
| Step count, fast tier | exactly 9 × 20,000 × 13 = **2,340,000** |
| Mean band (standardised) | \|mean(z)\| ≤ 4/√N_sim → 0.89 % at deep tier |
| Variance band (standardised) | \|var(z) − 1\| ≤ 4·√(2/N_sim) → 1.26 % at deep tier |
| Exact noise identity (task 1) | ≤ 1e-12, relative to summed absolute terms |
| Force-liquidation guard case (task 4) | realised-schedule cost vs `schedule_moments` ≤ 1e-10 relative |
| Runtime, deep tier | ≤ 30 min unchanged; **expected ≈ 300 s** at M1's observed rate (236 k steps/s). Record the actual. A run materially under ~250 s is a signal to check the step counter, not to celebrate. |
| Runtime, fast tier | ≤ 90 s added to `make test`; suite total ≤ 3 min |

## Definition of done

- [x] Preflight items 1–3 recorded in the parent brief. All three are in a new
      **Amendments** section there, and the pre-stated table now reads 200,000 rather than
      carrying a correction underneath it.
- [x] Task 1 green on all 27 cells + the guard case; the test states its sign convention
      and assembles the functional independently. `tests/test_noise_identity.py` — 29
      cells × 64 episodes, worst residual **1.29e-13** against the 1e-12 band. The sign is
      a named constant with §4's reasoning attached; the draws are regenerated from the
      seed address, never read off the env. **The brief's index convention was wrong —
      see finding 1.**
- [x] Step counts assert to the exact pre-stated totals, both tiers. Deep
      **70,200,000**, fast **2,340,000**, each also asserted per cell.
- [x] Observation guard green; static rejection of the shock literal outside `temper/env/`
      and `tests/`. Plus a behavioural guard: two shock streams, byte-identical
      observations; and the env's public attribute surface is pinned exactly.
- [x] Guard case present in the config and green; the M0 watch item is closed with a named
      test, not a note — `tests/test_sinh_asymptote_guard.py`. **Closed as unreachable,
      not merely as untriggered — see finding 2.**
- [x] Seed-pool assertion green; train/eval pools provably untouched. All 36 cells checked
      by address, and a session-wide recorder wraps the env's one route to randomness so
      the verdict covers the whole test path rather than the cells one test drove.
- [x] Task 0 reported part-by-part, all four green, monotonicity included. Nothing was
      missing — M1 had landed all four; M1a added the reporting. (a) Cholesky 9/9,
      smallest pivot 1.6e-5; (b) generic solve vs `optimal_trajectory` worst **1.4e-15**
      of X against a 1e-12 band, 108 interior holdings; (c) 3,600 perturbations
      (9 cases × 2 scales × 200 directions), worst ΔU/|U| **+1.3e-12** — every one uphill,
      floor −1e-9; (d) monotonicity 117 trades, smallest **+1.8e-4** of X.
- [x] Clean clone from the committed branch: fresh venv, suite green, **test count matches
      the in-tree run**; goldens, config and export script confirmed tracked.
- [x] §9 carries the 12–54 % magnitude; M0 brief supersession note sits at the bolded
      sentence. Both were already true at M1 — confirmed, not re-landed. §9 gained one new
      entry from this session (the shock-ordering convention, finding 1).
- [x] `make differential` green at N_sim = 200,000, actual runtime recorded. **317 s**,
      221 k steps/s, against a 1800 s budget and an expectation of ~300 s. Not materially
      under, so the counter and the wall clock agree. Worst band use: mean 55 %, variance
      50 % (both MSFT, deep).
- [x] `ROADMAP.md` M1 row flipped only once all of the above is green.

## Closed 2026-08-04

Six tasks were hardening and landed as specified. Two produced findings, and one of them
is about this brief.

**1. The functional this brief specified is the bug class it was written to kill.**
Task 1 asks for `σ√τ · Σ_{k=1}^{N−1} x_k ξ_k` over *post-bin* holdings. That is the
textbook Almgren–Chriss convention, in which the first trade executes at the arrival price
and bears no volatility. It is not this project's: FrontierView charges the shock *before*
the bin, so every share still held at the start of a bin carries it and the sum runs
`k = 0 … N−1` over inventory before each bin, keeping its `x_0 = X` term.
`oracle.shortfall_variance_bps2` reproduces the vendored variance to 9.5e-16 and invariant
2 makes the goldens normative, so the goldens decide — and §4 says as much ("the goldens,
not this document, are the numeric spec").

The gap is not subtle: `Σ_{k=0}^{N−1}(x_k/X)² − Σ_{k=1}^{N}(x_k/X)² = 1` identically, for
every schedule, which is **20.6 % of V for TWAP at N = 13** — the off-by-one-in-`Σx_k²`
class M0 flagged and this brief's own §Context cites as what the deep tier must kill.
Written to the brief's formula the test would have been red against a correct env, and the
"finding" would have been the document rather than the code. Implemented to the normative
convention, stated in the test module's docstring, pinned algebraically by
`test_the_post_bin_convention_is_the_named_off_by_one`, and recorded as an
`ARCHITECTURE.md` §9 amendment because the ambiguity is invisible in code — both
conventions are one-line sums over a trajectory — and expensive in prose.

**2. The M0 watch item cannot manifest, so it is closed rather than carried.**
No cell of the 3 × 3 grid reaches the `sinh-overflow-asymptote` branch (largest `κT` = 20.6
vendored, 9.0 exact, against a threshold of 500), as expected. But the stronger statement
is available and is what got landed: taking the branch requires `κT > 500`, so on the
canonical 13-bin grid the per-bin decay is at most `e^{−500/13}` = 2.0e-17 — below half an
ulp at `X`. The first bin's planned trade `X − x₁` therefore rounds to exactly `X`, the env
holds zero from bin 1 onward, and the terminal residual `X·e^{−κT}` is annihilated some 200
orders of magnitude before anything could charge it. M0's caution was right and the code
honours it; the note can stop travelling. Recorded at the watch item in M0's brief, where
a future grep lands.

A corollary worth stating: the vendored `sinh-overflow-asymptote` golden could not serve as
the guard case. At its λ = 100 the residual *underflows to exactly 0.0*, so it takes the
branch without ever producing the residual the watch item is about. The guard uses λ = 1.0
— `κT` ≈ 650, past the threshold and under the ~709 where `e^{−κT}` stops being
representable — which is also why the guard is genuinely not a golden.

**3. What task 1 buys, demonstrated rather than argued.** Flipping the sign of the env's
price shock (`self._walk += …` → `-=`) leaves **the entire Monte-Carlo differential green**
— the standardised cost distribution is symmetric, so neither the mean band nor the
variance band can see it — and turns every cell of the noise identity red. Five mutations
were run against the new guards in total (double-incrementing step counter; flipped shock
sign; post-bin index convention; a shock-key literal planted in `temper/agents/`; the
config pointed at the `train` pool) and each goes red on the guard written for it.

**4. Deliberately not done.** No timing assertion, per M1's finding 5 — the step counter is
asserted and the wall clock is only reported, because a test that goes red because the box
was busy teaches a session to rerun until green. The two are printed side by side so a
disagreement between them is visible rather than reconcilable.

## Out of scope (resist)

PPO and anything that trains; figures and plotting deps; new dependencies; any change to
`temper/env/` made to close a numerical gap; re-deriving or re-vendoring κ; power-law
dynamics; `client/`.

## Session notes

- Tasks 1 and 2 exist to *remove* trust, not to add coverage. If either fails, the
  finding is the product — report it and stop. Do not adjust the env, the tolerance or the
  functional to make it pass. This is the same rule that made M0's κ finding valuable.
- The step counter and the runtime budget interact: making the loop faster is fine, making
  it faster by processing more than one episode per `step` call is the thing the counter
  exists to catch. If the counter and the wall clock disagree with M1's observed rate,
  say so rather than reconciling them silently.
- Task 4's guard case is not a golden and must not be presented as one. It exercises a
  branch the vendored fixture does not reach; label it in the config so a future re-export
  does not try to source it from FrontierView.
- If task 1 lands cleanly, note in the parent brief that the Monte-Carlo tier's role has
  changed — it now certifies the *draws*, not the cost assembly. M2 will want to know
  which of its guarantees are exact and which are sampled.
