# Temper — Roadmap

Semi-stable. Update the **Status** column as milestones complete; anything structural
belongs in `ARCHITECTURE.md` §9 instead. Sessions: your milestone's brief in
`docs/briefs/` overrides the one-line summary here, and the constitution overrides both.

**Standing priority note:** Anvil, FrontierView interview prep, and DepthCharge bench work
rank ahead of Temper in the owner's queue. Temper sessions are opportunistic; keep every
milestone evening-sized on the reference box (constitution §6.9).

All milestones are agentic (Opus / Claude Code sessions converging on pytest red/green);
there is no bench track. M0 has one owner-input step: the golden export is run in the
FrontierView repo (see the M0 brief) — everything else is in-repo.

## Milestones

| #  | Milestone                    | Goal / definition of done                                                                                           | Depends on | Status |
| -- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------- | ------ |
| M0 | Oracle + goldens             | Repo scaffold (pytest, Makefile, pinned requirements); `oracle/` closed forms (κ, sinh trajectory, E/V, frontier, TWAP moments); FrontierView goldens vendored with provenance; oracle matches every golden case within pre-stated tolerance; green from a clean clone. | —          | ☑ **Done** 2026-08-04 — brief: `docs/briefs/M0-oracle-and-goldens.md`; 16 vendored cases + 17 frontier points matched to ~1e-15, ten orders inside the 1e-6 tolerance. Yielded three §9 amendments. |
| M1 | Env + analytic differential  | `ExecutionEnv` (Phase-1 dynamics, §4 contract *as amended* — see `ARCHITECTURE.md` §9, 2026-08-04); TWAP + AC schedule wrapped as policies; Monte-Carlo harness; simulated cost mean/variance for both fixed policies match closed forms within pre-stated CI. **The load-bearing correctness milestone** — env bugs die here, not in training curves. | M0 | ☐ **Next** |
| M2 | PPO rediscovery              | Single-file PPO (control-task smoke test green first); trained on `(t, inventory)` at one λ; mean shortfall within brief-stated ε of the oracle across ≥5 seeds; trajectory-overlay figure (agent vs sinh) committed to `results/`. | M1 | ☐ |
| M3 | Frontier sweep               | λ grid; RL-traced (E, V) points overlaid on the analytic efficient frontier — the hero figure; per-λ tolerance table committed. | M2 | ☐ |
| M4 | Broken assumptions           | Transient impact with exponential decay + stochastic liquidity behind the same env interface; expectation tests for the new model (invariant 6); AC-schedule degradation quantified; agent advantage shown out-of-sample with CIs. | M2 | ☐ |
| M5 | Alpha-aware execution        | Weak short-horizon signal in the observation; agent learns to tilt the schedule; advantage holds on held-out seeds; overfit check (signal-shuffled control) green. | M4 | ☐ |
| M6 | Anvil live leg *(stretch)*   | `client/` works a parent order on the live Anvil book via `PROTOCOL.md` (vendored snapshot) — third independent client; arrival-slippage measured and reported as a demo, not an evaluation (constitution §7); DepthCharge panel cameo optional but delightful. | M2 + Anvil deployed | ☐ |
| MP | Portfolio portal             | **Executes in the `garethcooke-portfolio` repo.** Stage 1 (after M3): `/projects/temper` live with the frontier hero figure, In Progress badge, tags, repo/architecture links. Stage 2 (after M6): live-leg writeup. | Stage 1: M3 · Stage 2: M6 | ☐ |

## Backlog (not scheduled)

- SAC / behaviour-cloning-warm-start comparison against the PPO baseline.
- C++ inference leg: ONNX export of the trained policy + a Crucible post (policy inference
  under fire; ties the ML work back to the C++ story).
- Real-data calibration (LOBSTER or crypto L2) replacing the synthetic parameter set.
- Limit-order placement / queue-position action space — the v2-sized extension.
- Anvil-side (lives on Anvil's backlog, cross-referenced only): feeder realism — Hawkes
  arrivals / mirror mode — which would make the M6 book livelier; the sequenced L2 feed.
- DepthCharge cameo hardening: a dedicated capture of the agent working an order, vendored
  as a DepthCharge replay trace.
