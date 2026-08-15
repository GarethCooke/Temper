# Temper — the red/green gate. `make test` from a clean clone is the milestone bar.

PYTHON ?= python
FRONTIERVIEW ?= ../FrontierView
GOLDENS := tests/golden/vendor/frontierview_goldens.json

M2_CONFIG   ?= configs/m2_ppo.yaml
M2_SAMPLED  ?= configs/m2_ppo_sampled.yaml
M3_VALIDATE ?= configs/m3_antithetic_validation.yaml

.PHONY: help test test-verbose differential smoke sweep reference validate goldens clean

help:
	@echo "make test          run the pytest suite (the gate); excludes the marked tiers"
	@echo "make differential  the deep Monte-Carlo tier: M1's acceptance gate, minutes"
	@echo "make smoke         PPO convergence on Pendulum + CartPole: M2's task 2, ~7 min"
	@echo "make reference     M2 task 0: the oracle-only table, and the lambda it fixes"
	@echo "make sweep         M2's 5-seed sweeps, both estimators - hours, unattended"
	@echo "make validate      M3 task 1: antithetic pairing at M2's lambda, 10 seeds - a night"
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
