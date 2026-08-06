# Temper — the red/green gate. `make test` from a clean clone is the milestone bar.

PYTHON ?= python
FRONTIERVIEW ?= ../FrontierView
GOLDENS := tests/golden/vendor/frontierview_goldens.json

M2_CONFIG   ?= configs/m2_ppo.yaml
M2_SAMPLED  ?= configs/m2_ppo_sampled.yaml

.PHONY: help test test-verbose differential smoke sweep reference goldens clean

help:
	@echo "make test          run the pytest suite (the gate); excludes the marked tiers"
	@echo "make differential  the deep Monte-Carlo tier: M1's acceptance gate, minutes"
	@echo "make smoke         PPO convergence on Pendulum + CartPole: M2's task 2, ~7 min"
	@echo "make reference     M2 task 0: the oracle-only table, and the lambda it fixes"
	@echo "make sweep         M2's 5-seed sweeps, both estimators - hours, unattended"
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
# configs are run because the sampled-reward plateau is a committed result, not a
# discarded attempt (docs/briefs/M2-ppo-rediscovery.md, amendment 1).
sweep:
	$(PYTHON) tools/m2_train.py --config $(M2_SAMPLED) --quiet
	$(PYTHON) tools/m2_train.py --config $(M2_CONFIG) --quiet

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
