"""M4a task 2 — the seam's acceptance: Phase 1 reproduces **bitwise**.

``make test`` green is necessary and not sufficient. M4a made the env's temporary
charge an injected model instead of a precomputed per-share constant, and the
Phase-1 model is supposed to be *exactly* what was there before — not
"equivalent", not "within 1e-12". M2 and M3 both established that this repo
reproduces bitwise on a fixed thread count, so anything less than bitwise here is
the seam having changed float order in Phase 1, and every committed result would
be a number produced by code that no longer exists.

What is re-run: seed 0 of ``configs/m3_frontier/lambda_1e-3.5.yaml``, the frontier
point at the lambda M2 and M3 both committed and the lambda M4a trains at. Its
grade must reproduce the committed one to the last bit — the objective to
seventeen digits and every point of the trajectory.

Marked ``training``: it is a 5 M-step seed, about twenty minutes on the reference
box, and it belongs to milestone acceptance rather than to the per-commit gate.

The other half of the regression is not here because it is not a training run:
``results/m3_frontier.png`` must redraw byte-identically from the committed JSON,
which ``tests/test_m3_frontier.py`` already asserts and which ``make
frontier-figure`` reproduces from a clean clone.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from temper.agents.execution import PPOPolicy
from temper.eval.experiment import load_experiment
from temper.eval.grading import grade_policy
from temper.eval.sweep import refuse_if_budget_bound, train_seed
from temper.oracle import LINEAR_ENCODING

from .conftest import REPO_ROOT

POINT = REPO_ROOT / "configs" / "m3_frontier" / "lambda_1e-3.5.yaml"
RESULT = REPO_ROOT / "results" / "m3_frontier" / "lambda_1e-3.5.json"

M3 = load_experiment(POINT)
DOCUMENT = json.loads(RESULT.read_text(encoding="utf-8"))


def test_the_regression_point_is_the_phase_one_world_at_m4as_lambda():
    """Cheap, and it runs in `make test`: the fixture is what it claims to be.

    A regression that silently pointed at a different lambda, or at a config the
    seam had quietly moved into the power-law world, would retrain something and
    prove nothing.
    """
    assert M3.cost_encoding == LINEAR_ENCODING
    assert M3.lambda_risk == 10.0**-3.5
    assert M3.estimator.regime == "antithetic"
    assert M3.ppo.torch_threads == 8
    assert DOCUMENT["config"]["lambda_risk"] == M3.lambda_risk
    assert DOCUMENT["provenance"]["git_dirty"] is False


@pytest.mark.training
def test_one_m3_seed_retrains_bitwise_identically_through_the_new_seam():
    """The seam's acceptance. Bitwise, not ``allclose``.

    ``allclose`` would pass on a seam that changed the order of a float addition
    — which is exactly the failure mode worth catching, because PPO compounds it
    over ~750 updates and M2 measured the same seed address landing at 0.165 and
    0.066 of the TWAP gap under nothing worse than a different thread count.
    Equality is the only bar that means "the Phase-1 arithmetic is unchanged".
    """
    committed = DOCUMENT["seeds"][0]["grade"]
    committed_training = DOCUMENT["seeds"][0]["training"]
    _, policy = train_seed(M3, 0)
    assert isinstance(policy, PPOPolicy)

    regraded = grade_policy(
        policy,
        M3.case.market,
        M3.case.order_size,
        M3.reference(),
        root_seed=M3.seeds.root_seed,
        pool=M3.seeds.eval_pool,
        streams=M3.seeds.eval_streams,
        name="seed0",
    )

    # Neither side of a bitwise comparison may be a run whose wall-clock guard
    # bound early: that is fewer updates than the config named, and the
    # comparison would be between two different amounts of training. M5 task 2
    # spent an hour producing exactly that RED before the guard existed.
    refuse_if_budget_bound(
        [committed_training, result],
        comparison="the bitwise seed regression",
        labels=["the committed seed 0", "the retrained seed 0"],
    )

    assert regraded.objective == committed["objective_bps"], (
        f"the retrained seed's objective is {regraded.objective!r}, the committed "
        f"value is {committed['objective_bps']!r}; the seam changed Phase-1 "
        "arithmetic"
    )
    expected = np.asarray(committed["trajectory"], dtype=float)
    assert np.array_equal(regraded.trajectory, expected), (
        "the retrained seed's trajectory is not bitwise the committed one; worst "
        f"difference {float(np.max(np.abs(regraded.trajectory - expected))):.3e} "
        "shares"
    )
    assert regraded.gap_fraction == committed["gap_fraction"]
    assert not regraded.red_flag
