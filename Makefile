# Temper — the red/green gate. `make test` from a clean clone is the milestone bar.

# The venv's interpreter, not the host's: every target here needs numpy, torch and
# pyyaml, and a bare `python` on a Windows box is usually the system install that
# has none of them (README: create .venv and install requirements first). Override
# for a different layout: `make test PYTHON=/path/to/python`.
PYTHON ?= python
FRONTIERVIEW ?= ../FrontierView
GOLDENS := tests/golden/vendor/frontierview_goldens.json

M2_CONFIG   ?= configs/m2_ppo.yaml
M2_SAMPLED  ?= configs/m2_ppo_sampled.yaml
M3_VALIDATE ?= configs/m3_antithetic_validation.yaml
M3_FRONTIER ?= configs/m3_frontier.yaml
M4A_CONFIG  ?= configs/m4a_power_law.yaml
M4B_CONFIG  ?= configs/m4b_liquidity.yaml
M5_CONFIG   ?= configs/m5_alpha.yaml

.PHONY: help test test-verbose differential smoke sweep reference validate frontier frontier-figure frontier-check m4a-reference m4a-guarantees m4a-regression m4a m4a-figure checkpoint m4b-reference m4b-oracle m4b-differential m4b-guarantees m4b-regression m4b m4b-figure m5-reference m5-oracle m5-regression m5-guard m6-figure anvil-check m6-predict m6 m6-thin m6-wide m6-feeder goldens clean

help:
	@echo "make test          run the pytest suite (the gate); excludes the marked tiers"
	@echo "make differential  the deep Monte-Carlo tier: M1's acceptance gate, minutes"
	@echo "make smoke         PPO convergence on Pendulum + CartPole: M2's task 2, ~7 min"
	@echo "make reference     M2 task 0: the oracle-only table, and the lambda it fixes"
	@echo "make sweep         M2's 5-seed sweeps, both estimators - hours, unattended"
	@echo "make validate      M3 task 1: antithetic pairing at M2's lambda, 10 seeds - a night"
	@echo "make frontier      M3 tasks 4-5: the nine-lambda sweep, then the frontier figure - a day"
	@echo "make frontier-check M3's amended update budget, checked at one lambda first - ~2 h"
	@echo "make frontier-figure redraw the committed frontier from results/m3_frontier.json"
	@echo "make m4a-reference M4a task 0: the power-law table and its three gates"
	@echo "make m4a-guarantees M4a task 4: the four inherited guarantees, before training"
	@echo "make m4a-regression M4a task 2: one M3 seed retrained bitwise - ~20 min"
	@echo "make m4a           M4a task 5: ten seeds in the power-law world - ~3 h"
	@echo "make m4a-figure    redraw results/m4a_degradation.* from the committed result"
	@echo "make m4b-reference M4b task 0: the liquidity table and its four gates - ~5 min"
	@echo "make m4b-oracle    M4b task 1: the adaptive oracle checks, incl. sigma_L -> 0"
	@echo "make m4b-guarantees M4b task 4: the inherited guarantees through two seams"
	@echo "make m4b-regression M4b task 2: one M3 seed AND one M4a seed bitwise - ~40 min"
	@echo "make m4b           M4b task 5: ten seeds in the stochastic-liquidity world - ~3 h"
	@echo "make m4b-figure    redraw results/m4b_adaptivity.* from the committed result"
	@echo "make m5-reference  M5 tasks 0+1: the alpha table, its gates and its word - ~7 min"
	@echo "make m5-oracle     M5 tasks 0+1: the alpha oracle checks, incl. rho -> 0"
	@echo "make m5-regression M5 task 2: one committed seed per world, bitwise - ~1 h"
	@echo "make m5-guard      M5 task 3: the amended guard, and what it still refuses"
	@echo "make m6-figure     redraw results/m6_prediction.* from the five committed M6 runs"
	@echo "make checkpoint    M6 prerequisite: export M4a's median seed as a policy .npz - ~15 min"
	@echo "make anvil-check   M6 task 0: Anvil's documented behaviour, against a live server"
	@echo "make m6-predict    M6 task 4: the closed-form prediction, no server needed"
	@echo "make m6            M6 task 5: the measured ladder run (live anvil_server, feeder off)"
	@echo "make m6-thin / m6-wide / m6-feeder  the other three M6 runs"
	@echo "make goldens       re-export the FrontierView fixtures (read-only there)"
	@echo "                   override the checkout with FRONTIERVIEW=/path/to/FrontierView"
	@echo "make clean         remove caches and scratch results"

test:
	$(PYTHON) -m pytest

test-verbose:
	$(PYTHON) -m pytest -vv

# The full 3 x 3 golden grid x 3 schedules at N_sim = 200,000 — 27 cells and
# 70,200,000 calls into ExecutionEnv.step, per configs/m1_differential.yaml. Run
# at least once at milestone acceptance; the command-line -m overrides the
# `not deep` in addopts.
differential:
	$(PYTHON) -m pytest -m deep -v

# M2 task 2 — the CleanRL adaptation solves a standard control task, on three
# seeds each, before anything it produces on ExecutionEnv is believed. Also the
# thing that tells later milestones whether a flat training curve means "PPO is
# broken" or "the env is hard".
smoke:
	$(PYTHON) -m pytest tests/test_ppo_smoke.py -m training -v

# M2 task 0 — oracle only, no agent, no training. Safe to run any time; it is
# what fixes lambda and what `tests/test_m2_reference.py` re-derives.
reference:
	$(PYTHON) tools/m2_reference_table.py --config $(M2_CONFIG)

# M2's acceptance run: five training seeds per estimator, graded analytically
# against the oracle, writing results/*.json and the overlay figures. Hours on
# the reference box — this is the unattended run, not a per-commit gate. Both
# configs are run because the sampled-reward miss is a committed result, not a
# discarded attempt (docs/briefs/M2-ppo-rediscovery.md, amendment 1).
#
# `--expect miss` on the first line is load-bearing rather than a mute button.
# The driver's exit status is "did this run reach the verdict it was expected
# to?", so the sampled sweep failing to clear epsilon is a success here and
# *clearing* it would stop the target — which is right, because a recorded miss
# that starts passing invalidates the amendment resting on it. Written as `-`
# (ignore errors) this line would have hidden that in exactly the case worth
# hearing about.
sweep:
	$(PYTHON) tools/train.py --config $(M2_SAMPLED) --quiet --expect miss
	$(PYTHON) tools/train.py --config $(M2_CONFIG) --quiet --expect pass

# M3 task 1 — the gate for everything else in that milestone: antithetic pairing
# at M2's lambda, ten seeds, everything else identical to configs/m2_ppo.yaml.
# The verdict it is expected to reach is the epsilon one; the *gate* (median gap
# within an order of magnitude of the control variate's) is reported beside it
# and asserted by tests/test_m3_validation.py against the committed result.
# Serial with everything else, and unattended: 512 envs x 8 threads saturates
# the reference box, and two concurrent sweeps truncate each other (M2).
validate:
	$(PYTHON) tools/train.py --config $(M3_VALIDATE) --quiet --expect pass

# M3 tasks 4-5 — the frontier sweep. `check` refuses to start if a committed
# point config is not what the generator writes from the manifest; `run` takes
# each of the nine lambda points through tools/train.py in a fresh process,
# strictly serially, with `--expect any` (a per-lambda epsilon miss is a finding
# the frontier reports; a red flag still stops it); `figure` aggregates the nine
# results files into results/m3_frontier.json and draws results/m3_frontier.png
# — and redraws it byte-identically from the committed JSON without training.
frontier:
	$(PYTHON) tools/m3_frontier.py check
	$(PYTHON) tools/m3_frontier.py run --quiet
	$(PYTHON) tools/m3_frontier.py figure

# The amended update budget, checked at the one lambda two committed results
# already answer, before the other eight points spend sixteen hours on it.
# Redraw the committed frontier without retraining or re-aggregating: the figure
# is a view of results/m3_frontier.json, and this is the form that reproduces its
# bytes from a clean clone.
frontier-figure:
	$(PYTHON) tools/m3_frontier.py figure --redraw

frontier-check:
	$(PYTHON) tools/m3_frontier.py run --quiet --only 3.1622776601683794e-04

# M4a task 0 — oracle only, no agent, no training, minutes. Applies M2's
# selection rule to the power-law table, checks it agrees with the linear one,
# and reports the three gates. Exit status is whether all three are green.
m4a-reference:
	$(PYTHON) tools/m4a_reference_table.py --config $(M4A_CONFIG)

# M4a task 4 — the four guarantees the power-law world inherits, run and recorded
# *before* the training point. If any goes red, that is the milestone's finding
# and training does not start.
m4a-guarantees:
	$(PYTHON) -m pytest tests/test_m4a_inherited_guarantees.py -v

# M4a task 2 — the env seam's acceptance. `make test` green is necessary and not
# sufficient: one M3 seed at 10^-3.5 is retrained through the new code and its
# objective and whole trajectory must be *bitwise* the committed ones. ~20 min.
m4a-regression:
	$(PYTHON) -m pytest tests/test_m4a_phase1_regression.py -m training -v

# M4a task 5 — the training point. Ten seeds at the rule-selected lambda in the
# power-law world, graded analytically against the certified optimum. Serial and
# unattended: 512 envs x 8 threads saturates the reference box.
m4a:
	$(PYTHON) tools/train.py --config $(M4A_CONFIG) --quiet --expect pass

# M6's prerequisite — the exported policy. Retrains *one* seed of the committed
# M4a sweep (the median rule picks it; see docs/briefs/M6-anvil-live-leg.md) and
# writes results/m4a_power_law_policy.npz with provenance. ~15 min, and it must
# run from a committed tree: the artefact carries `git_dirty: false` or it is not
# an artefact (invariant 1).
checkpoint:
	$(PYTHON) tools/train.py --config $(M4A_CONFIG) --quiet --export-checkpoint

# M4b task 0 — oracle only, no agent, no training, ~5 minutes. Extends M4a's
# table to the liquidity world: the best fixed schedule that knows the liquidity
# LAW, the dynamic-programming optimum over adapted policies, and the two sampled
# bounds that bracket it. Applies M2's selection rule in every reading and RECORDS
# which one it is applied to — liquidity is not a new cost encoding, so that had
# to be decided rather than inherited. Exit status is whether all four gates are
# green. Writes results/m4b_reference.json.
m4b-reference:
	$(PYTHON) tools/m4b_reference_table.py --config $(M4B_CONFIG)

# M4b task 1 — the adaptive oracle's own checks, including the sigma_L -> 0
# differential against M4a's *certified* value. Seconds; the grid-convergence
# tier is behind the `deep` marker.
m4b-oracle:
	$(PYTHON) -m pytest tests/test_m4b_adaptive_oracle.py -v

# M4b task 4 — the differential through two seams, and the liquidity process's own
# moments against the oracle's closed forms. The fast tier is in `make test`; this
# runs the deep one, 45 cells x 200,000 episodes.
m4b-differential:
	$(PYTHON) -m pytest tests/test_m4b_differential.py -m deep -v

# M4b task 4's precondition — the guarantees the liquidity world inherits, run and
# recorded *before* the training point. If any goes red, that is the milestone's
# finding and training does not start.
m4b-guarantees:
	$(PYTHON) -m pytest tests/test_m4b_inherited_guarantees.py -v

# M4b task 2 — the seam's acceptance, across BOTH worlds. One committed M3 seed
# and one committed M4a seed retrained through the second seam, each required to
# reproduce its grade bitwise. ~40 min.
m4b-regression:
	$(PYTHON) -m pytest tests/test_m4b_phase1_regression.py -m training -v

# M4b task 5 — the training point. Ten seeds in the stochastic-liquidity world,
# graded by conditional expectation on held-out liquidity paths. Serial and
# unattended.
m4b:
	$(PYTHON) tools/train.py --config $(M4B_CONFIG) --expect pass

# M4b task 6 — the adaptivity figure. Both panels are views of committed
# artefacts: the rungs from the sweep, the value-of-sight curve from task 0's
# table, so it redraws without retraining.
m4b-figure:
	$(PYTHON) tools/m4b_adaptivity.py --config $(M4B_CONFIG)

# M5 tasks 0 and 1 - oracle only, no agent, no training, ~7 minutes. The dynamic
# over (inventory, signal), the reference table across the vendor grid, and the
# impact / risk / alpha / objective decomposition with the identity asserted at
# every node rather than assumed. Also asserts what M4b had to decide: lambda's
# static reading here is BIT-IDENTICAL to M4a's, so there is no new reading and no
# choice to record. Exit status is whether all four gates are green. Writes
# results/m5_reference.json.
#
# Task 1 rides the same artefact because there is one reference and it has two
# halves of different kinds: grid and quadrature convergence with a Richardson
# residual, the (k, x_k, s_k) sufficiency check on an augmented state, the timing
# check that a signal about an already-landed shock is worth zero, and both
# references carrying their own certified/converged word.
#
# The milestone's own rule is that no training code is written, imported or run
# until those four gates are green in the repo, so this is the first thing M5 runs
# and `make m5-oracle` is the second.
m5-reference:
	$(PYTHON) tools/m5_reference_table.py --config $(M5_CONFIG)

# M5 task 2 - the signal seam's acceptance, across ALL THREE worlds. One
# committed M3 seed, one M4a seed and one M4b seed retrained through the new seam,
# each required to reproduce its grade bitwise. ~1 h, and it runs BEFORE the seam
# is wired into anything that trains: what it measures is that the signal draws
# from its own pools and so cannot reach the price generator. A moved digit means
# the correlation M5 exists to measure would be partly manufactured by two noise
# sources sharing a stream.
m5-regression:
	$(PYTHON) -m pytest tests/test_m5_signal_seam.py -m training -v

# M5 task 3 - the amended observation-minimality guard, pointed at what it now
# permits AND at every case it must still refuse: the realised price, the realised
# shortfall, and a signal about a shock that is already committed. Seconds, and it
# runs in `make test` as well; this is the named entry point.
m5-guard:
	$(PYTHON) -m pytest tests/test_m5_observation_guard.py -v

# M5's oracle checks - the signal's joint law, the conditional cost against
# sampled price draws, the decomposition by two routes, the convexity floor, the
# rho -> 0 differential against M4a's *certified* value, and task 1's two: the
# augmented-state sufficiency solve and the already-landed timing instrument.
# Seconds.
m5-oracle:
	$(PYTHON) -m pytest tests/test_m5_alpha_oracle.py -v

# M6's figure. Both panels are views of the five committed run artefacts —
# the per-bin series off each run's own bins, the tier rows off each run's
# measurement block — so this redraws from a clean clone and touches NO venue
# and NO network: no anvil_server, no socket, nothing to restart between
# invocations. There is no --config because the artefacts are the input.
m6-figure:
	$(PYTHON) tools/m6_prediction.py

# M4a task 6 — the degradation figure. Oracle curves are free; the agent's ten
# seeds are read off the committed result, so this redraws without retraining.
m4a-figure:
	$(PYTHON) tools/m4a_degradation.py --config $(M4A_CONFIG)

# M6 — the Anvil live leg. All four runs need a live anvil_server; start one on a
# single-ticker roster with the feeder off, and RESTART IT between measured runs.
# A measured run predicts from the committed ladder and nothing else, so it
# refuses to build on a book that already has orders resting:
#
#   ANVIL_TICKERS=101 ANVIL_DEFAULT_TICKER=101 ANVIL_FEEDER=0 ANVIL_PORT=18080 #       anvil_server
#
# The demonstration run wants the opposite: the feeder on, its seed pinned.
#
#   ANVIL_TICKERS=101 ANVIL_DEFAULT_TICKER=101 ANVIL_FEEDER=1 #       ANVIL_FEEDER_SEED=20260822 ANVIL_PORT=18080 anvil_server
M6_CONFIG ?= configs/m6_anvil.yaml

# Anvil's documented behaviour, read back against the thing it documents. The
# vendored snapshot is a set of predictions about somebody else's software; this
# is what keeps them honest. Needs the feeder OFF.
anvil-check:
	$(PYTHON) -m pytest tests/test_anvil_live.py tests/test_m6_void.py -m anvil -v

# The prediction, computed from the committed ladder before anything is sent.
# No server needed, and it is the order the brief requires.
m6-predict:
	$(PYTHON) -m client.run --config $(M6_CONFIG) --run ladder --dry-run

# The measured run: client-built ladder, feeder off, regenerable.
m6:
	$(PYTHON) -m client.run --config $(M6_CONFIG) --run ladder

# The prediction machinery, harder: a ladder too thin to fill bin one (partial
# fills and the cancel path) and one with three times the spread.
m6-thin:
	$(PYTHON) -m client.run --config $(M6_CONFIG) --run thin

m6-wide:
	$(PYTHON) -m client.run --config $(M6_CONFIG) --run wide

# The demonstration: Anvil's feeder builds and trades the book. Reported
# separately, and reproducible only in the weak sense.
m6-feeder:
	$(PYTHON) -m client.run --config $(M6_CONFIG) --run feeder

# Regenerates $(GOLDENS) from a FrontierView checkout. Writes nothing into that
# repo — the export imports its `api` package and nothing more (constitution §7).
goldens:
	@test -f "$(FRONTIERVIEW)/api/market_impact.py" || \
		{ echo "no FrontierView checkout at $(FRONTIERVIEW); set FRONTIERVIEW=..."; exit 1; }
	$(PYTHON) tools/export_frontierview_goldens.py \
		--frontierview-root "$(FRONTIERVIEW)" --out "$(GOLDENS)"

clean:
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
