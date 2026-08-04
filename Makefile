# Temper — the red/green gate. `make test` from a clean clone is the milestone bar.

PYTHON ?= python
FRONTIERVIEW ?= ../FrontierView
GOLDENS := tests/golden/vendor/frontierview_goldens.json

.PHONY: help test test-verbose differential goldens clean

help:
	@echo "make test          run the pytest suite (the gate); excludes the deep tier"
	@echo "make differential  the deep Monte-Carlo tier: M1's acceptance gate, minutes"
	@echo "make goldens       re-export the FrontierView fixtures (read-only there)"
	@echo "                   override the checkout with FRONTIERVIEW=/path/to/FrontierView"
	@echo "make clean         remove caches and scratch results"

test:
	$(PYTHON) -m pytest

test-verbose:
	$(PYTHON) -m pytest -vv

# The full 3 x 3 golden grid x 3 schedules at N_sim = 100,000, per
# configs/m1_differential.yaml. Run at least once at milestone acceptance; the
# command-line -m overrides the `not deep` in addopts.
differential:
	$(PYTHON) -m pytest -m deep -v

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
