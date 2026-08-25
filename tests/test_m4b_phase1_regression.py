"""M4b task 2 — the second seam's acceptance, across **two** worlds.

M4a's regression re-ran one M3 seed and required its grade bitwise. M4b has to do
that twice, because it added a second injected seam to an env that already had
one: the Phase-1 world (M3's frontier point) and the power-law world (M4a's
training point) must both come back to the last bit through code that now draws a
second noise source, publishes a third observation coordinate in some worlds, and
passes a liquidity multiplier into the temporary charge on every step.

Three ways it could have failed, and each is why one of the fast checks below
exists:

* **The liquidity variate out of the price generator.** The single most
  destructive possibility. Every downstream shock would shift, every committed
  Phase-1 and M4a number would stop regenerating, and *nothing would look wrong* —
  each result would still reproduce perfectly from its own config, against a
  different market. Prevented by construction (liquidity has its own pools) and
  checked here by seed address.
* **A charge that is no longer bitwise at ``L = 1``.** The temporary models gained
  a liquidity argument. ``x / 1.0 == x`` exactly in IEEE, so the arithmetic is
  unchanged — but "so it is fine" is an argument, and the trained digits are a
  measurement.
* **An observation that grew where it should not have.** The third coordinate
  exists only in a stochastic-liquidity world. A two-coordinate world that
  acquired a third would change the network's input width and every weight after
  it, which would be loud; one that acquired a *constant* third would be quiet and
  would still move every digit.

Marked ``training``: two 5 M-step seeds, about forty minutes on the reference box.
Milestone acceptance, not the per-commit gate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from temper.agents.execution import PPOPolicy
from temper.env import DETERMINISTIC_LIQUIDITY, ExecutionEnv
from temper.eval.experiment import load_experiment
from temper.eval.grading import grade_policy
from temper.eval.sweep import refuse_if_budget_bound, train_seed, training_liquidity
from temper.oracle import LINEAR_ENCODING, POWER_LAW_ENCODING, Market, SymbolParams

from .conftest import REPO_ROOT

#: The two committed points, one per world. Both at 10^-3.5, both antithetic,
#: both ten-seed sweeps whose seed 0 is re-trained here.
POINTS = {
    LINEAR_ENCODING: (
        REPO_ROOT / "configs" / "m3_frontier" / "lambda_1e-3.5.yaml",
        REPO_ROOT / "results" / "m3_frontier" / "lambda_1e-3.5.json",
    ),
    POWER_LAW_ENCODING: (
        REPO_ROOT / "configs" / "m4a_power_law.yaml",
        REPO_ROOT / "results" / "m4a_power_law.json",
    ),
}


def _load(encoding: str):
    config, result = POINTS[encoding]
    return load_experiment(config), json.loads(result.read_text(encoding="utf-8"))


@pytest.mark.parametrize("encoding", sorted(POINTS))
def test_the_regression_points_are_what_they_claim_to_be(encoding):
    """Cheap, and it runs in ``make test``: the fixtures have not moved worlds."""
    experiment, document = _load(encoding)
    assert experiment.cost_encoding == encoding
    assert experiment.lambda_risk == 10.0**-3.5
    assert experiment.estimator.regime == "antithetic"
    assert experiment.ppo.torch_threads == 8
    assert document["config"]["lambda_risk"] == experiment.lambda_risk
    assert document["provenance"]["git_dirty"] is False


@pytest.mark.parametrize("encoding", sorted(POINTS))
def test_neither_committed_point_acquired_a_liquidity_world_by_omission(encoding):
    """Constitution §4, on the *second* seam: Phase 2 is never inherited.

    Both configs predate ``world.liquidity`` entirely, so they must resolve to
    ``L = 1`` — the market they were trained in — and every env their training
    path builds must too. This is the check that would have caught a default
    moving, which is a one-line edit and would silently re-run two milestones in
    a world neither of them names.
    """
    experiment, _ = _load(encoding)
    assert not experiment.liquidity.stochastic
    assert experiment.liquidity.inverse_power_moment(0.6) == 1.0
    stream = training_liquidity(experiment)
    assert not stream.stochastic


@pytest.mark.parametrize("encoding", sorted(POINTS))
def test_the_observation_is_two_dimensional_where_it_always_was(encoding):
    """A third coordinate in a deterministic world would move every trained digit."""
    experiment, _ = _load(encoding)
    env = ExecutionEnv(
        experiment.case.market,
        experiment.case.order_size,
        experiment.lambda_risk,
        liquidity=training_liquidity(experiment),
        root_seed=experiment.seeds.root_seed,
        pool=experiment.seeds.train_pool,
    )
    assert env.observation_space.shape == (2,)
    observation, _ = env.reset(seed=0)
    assert observation.shape == (2,)
    assert np.all(env.multipliers == 1.0)


def test_the_two_noise_sources_are_addressed_in_different_pools():
    """The failure that would be invisible, made arithmetic.

    Same root seed, same stream index, different pool — so a liquidity draw
    cannot advance a price generator and no committed shock can move because a
    multiplier was needed. Asserted on the *addresses* rather than on a sample,
    because two streams that happened to agree on their first few draws would
    pass a sampled check and still be the same stream.
    """
    market = Market(
        params=SymbolParams(
            adv=6e7, sigma=0.0155, half_spread=0.3, eta=0.142, gamma=0.314
        ),
        horizon_hours=6.5,
        n_bins=13,
    )
    env = ExecutionEnv(
        market,
        100_000.0,
        1e-4,
        liquidity=DETERMINISTIC_LIQUIDITY,
        root_seed=5,
        pool="train",
        stream_index=17,
    )
    price_root, price_pool, price_index = env.seed_address
    liquidity_root, liquidity_pool, liquidity_index = env.liquidity_address
    assert (price_root, price_index) == (liquidity_root, liquidity_index) == (5, 17)
    assert price_pool == "train" and liquidity_pool != "train"


@pytest.mark.training
@pytest.mark.parametrize("encoding", sorted(POINTS))
def test_one_committed_seed_per_world_retrains_bitwise(encoding):
    """The seam's acceptance. Bitwise, not ``allclose``, in **both** worlds.

    ``allclose`` would pass on a seam that changed the order of a float addition
    — exactly the failure worth catching, because PPO compounds it over ~750
    updates and M2 measured the same seed address landing at 0.165 and 0.066 of
    the TWAP gap under nothing worse than a different thread count. Equality is
    the only bar that means "the arithmetic in the worlds M4b did not touch is
    unchanged".
    """
    experiment, document = _load(encoding)
    committed = document["seeds"][0]["grade"]
    committed_training = document["seeds"][0]["training"]

    _, policy = train_seed(experiment, 0)
    assert isinstance(policy, PPOPolicy)
    regraded = grade_policy(
        policy,
        experiment.case.market,
        experiment.case.order_size,
        experiment.reference(),
        root_seed=experiment.seeds.root_seed,
        pool=experiment.seeds.eval_pool,
        streams=experiment.seeds.eval_streams,
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
        f"{encoding}: seed 0's objective moved by "
        f"{regraded.objective - committed['objective_bps']:.3e} bps through the "
        "liquidity seam; the Phase-1 arithmetic is no longer what produced the "
        "committed result"
    )
    assert np.array_equal(
        regraded.trajectory, np.asarray(committed["trajectory"], dtype=float)
    ), f"{encoding}: seed 0's trajectory moved through the liquidity seam"
    assert regraded.red_flag == committed["red_flag"]
