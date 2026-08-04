# Temper — Roadmap

Semi-stable. Update the **Status** column as milestones complete; anything structural
belongs in `ARCHITECTURE.md` §9 instead. Sessions: your milestone's brief in
`docs/briefs/` overrides the one-line summary here, and the constitution overrides both.

**Standing priority note:** Anvil, FrontierView interview prep, and DepthCharge bench work
rank ahead of Temper in the owner's queue. Temper sessions are opportunistic; keep every
milestone evening-sized on the reference box (constitution §6.9).

All milestones are agentic (Opus / Claude Code sessions converging on pytest red/green);
there is no bench track. M0's golden export was the one owner-input step; re-exports are
`make goldens` against a FrontierView checkout (see `docs/vendor/frontierview-goldens.md`).

## Milestones

| #  | Milestone                    | Goal / definition of done                                                                                           | Depends on | Status |
| -- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------- | ------ |
| M0 | Oracle + goldens             | Repo scaffold (pytest, Makefile, pinned requirements); `oracle/` closed forms (κ, sinh trajectory, E/V, frontier, TWAP moments); FrontierView goldens vendored with provenance; oracle matches every golden case within pre-stated tolerance; green from a clean clone. | —          | ☑ **Done** 2026-08-04 — brief: `docs/briefs/M0-oracle-and-goldens.md`; 16 vendored cases + 17 frontier points matched to ~1e-15, ten orders inside the 1e-6 tolerance. Yielded three §9 amendments. |
| M1 | Env + analytic differential  | `ExecutionEnv` (Phase-1 dynamics, §4 contract *as amended* — see `ARCHITECTURE.md` §9, 2026-08-04); TWAP + AC schedule wrapped as policies; Monte-Carlo harness; simulated cost mean/variance for both fixed policies match closed forms within pre-stated CI. **The load-bearing correctness milestone** — env bugs die here, not in training curves. | M0 | ☑ **Done** 2026-08-04 — briefs: `docs/briefs/M1-env-and-analytic-differential.md`, then `docs/briefs/M1a-acceptance-hardening.md`. Deep tier (27 cells × 200 k episodes) green in 5 min 17 s, no cell using more than 55 % of its band, and 70,200,000 calls into the real `step` loop *asserted* rather than assumed. The differential is now only statistical about the *draws*: an exact per-episode identity pins the realised noise to the specific shocks the env made (worst 1.3e-13), so the cost assembly holds by construction. Seven exact identities, a variational certificate, and an observation-minimality guard that closes M2's leak before M2 exists. Yielded two §9 amendments (Phase 1 is linearised end-to-end with `cost_moments` quarantined; the shock lands before the bin, which fixes the variance sum's index range). |
| M2 | PPO rediscovery              | Single-file PPO (control-task smoke test green first); trained on `(t, inventory)` at one λ; mean shortfall within brief-stated ε of the oracle across ≥5 seeds; trajectory-overlay figure (agent vs sinh) committed to `results/`. | M1 | ☐ **Next** — grade against `optimal_trajectory` (§9, 2026-08-04), reward and metric both `oracle.schedule_moments`; `train`/`eval` seed pools are untouched and M1's draws came from `m1/differential`. |
| M3 | Frontier sweep               | λ grid; RL-traced (E, V) points overlaid on the analytic efficient frontier — the hero figure; per-λ tolerance table committed. | M2 | ☐ |
| M4 | Broken assumptions — power law first | The env's temporary impact becomes FrontierView's **calibrated 0.6-power law** (the vendored mis-specification, not an invented one) plus stochastic liquidity, behind the same env interface; `oracle.cost_moments` is the reporting reference for the power-law world; expectation tests for each new model (invariant 6); AC-schedule degradation quantified; agent advantage shown out-of-sample with CIs. | M2 | ☐ |
| M5 | Alpha-aware execution        | Weak short-horizon signal in the observation; agent learns to tilt the schedule; advantage holds on held-out seeds; overfit check (signal-shuffled control) green. | M4 | ☐ |
| M6 | Anvil live leg *(stretch)*   | `client/` works a parent order on the live Anvil book via `PROTOCOL.md` (vendored snapshot) — third independent client; arrival-slippage measured and reported as a demo, not an evaluation (constitution §7); DepthCharge panel cameo optional but delightful. | M2 + Anvil deployed | ☐ |
| MP | Portfolio portal             | **Executes in the `garethcooke-portfolio` repo.** Stage 1 (after M3): `/projects/temper` live with the frontier hero figure, In Progress badge, tags, repo/architecture links. Stage 2 (after M6): live-leg writeup. | Stage 1: M3 · Stage 2: M6 | ☐ |

## Backlog (not scheduled)

- **Revisit the oracle↔FrontierView code-sharing decision after M2, with M2's evidence in
  hand** (owner unconvinced, 2026-08-04). The 2026-08-04 audit sized the parallel formula
  code at ~55 executable lines across ~12 sites; the counter-case is the κ finding and
  vendor-doc quirk #1, both products of the second derivation. Constitution §7
  ("separate repo", "zero upstream changes") stands unless amended via §9.
- Transient impact with exponential decay (Obizhaeva–Wang-style) as a further Phase-2
  break beyond M4 — demoted from M4 when the calibrated power law was promoted to lead.
- SAC / behaviour-cloning-warm-start comparison against the PPO baseline.
- C++ inference leg: ONNX export of the trained policy + a Crucible post (policy inference
  under fire; ties the ML work back to the C++ story).
- Real-data calibration (LOBSTER or crypto L2) replacing the synthetic parameter set.
- Limit-order placement / queue-position action space — the v2-sized extension.
- FrontierView-side (lives on FrontierView's backlog, cross-referenced only): the
  discrete-κ convention question surfaced by M0 — if upstream ever adopts
  `cosh(κτ) = 1 + μ/2`, that is a golden re-vendor with fresh provenance, not a Temper
  break; and a `Market`-style derived-quantities object (2026-08-04 audit: `v_hourly`,
  `dt`, `sigma_bin` re-derived at three-plus call sites, with unnamed literals).
- Anvil-side (lives on Anvil's backlog, cross-referenced only): feeder realism — Hawkes
  arrivals / mirror mode — which would make the M6 book livelier; the sequenced L2 feed.
- DepthCharge cameo hardening: a dedicated capture of the agent working an order, vendored
  as a DepthCharge replay trace.
