"""M2 tasks 4 and 5 — the action parameterisation, and the reward scale.

Two claims the milestone rests on, made executable.

**The baseline is representable.** M2's headline is that the agent converged on
the Almgren–Chriss sinh rather than on TWAP. That comparison is worthless if
TWAP is awkward to express in the agent's coordinates — a policy that cannot
easily *be* TWAP will beat it for reasons that have nothing to do with learning.
Under the fraction-of-remaining parameterisation TWAP is the fixed sequence
``1/13, 1/12, ..., 1``, and so is exactly representable; so is the optimum. Both
are checked here by replaying them through the real env and comparing the
realised trajectory against the oracle's.

**The reward scale cannot move the answer.** It is one affine constant from the
committed config, stateless, and applied identically wherever it is applied at
all — and the graded metric never sees it, because grading is analytic on the
unscaled objective. Anything with running statistics would be objective drift by
the back door (invariant 7); ``tests/test_repo_invariants.py`` rejects those
statically, and what is here is the positive statement.

Envs are built on the ``m2/diagnostic`` pool: none of these checks reports a
number, so none of them may spend a stream a committed result is addressed by.
"""

from __future__ import annotations

import numpy as np
import pytest

from temper.agents import (
    FRACTION_SPACE,
    FractionAction,
    FractionPolicy,
    RewardScale,
    as_fraction,
    fraction_to_shares,
    twap_fractions,
)
from temper.env import EPISODE_KEY, ExecutionEnv
from temper.eval import run_episode
from temper.oracle import optimal_trajectory, schedule_moments, twap_trajectory
from temper.seeding import M2_DIAGNOSTIC_POOL

from .conftest import m2_experiment

EXPERIMENT = m2_experiment()
MARKET = EXPERIMENT.case.market
ORDER_SIZE = EXPERIMENT.case.order_size
LAMBDA = EXPERIMENT.lambda_risk
N_BINS = MARKET.n_bins

#: Diagnostic streams, offset from each other so no two tests here share shocks.
STREAMS = {"twap": 0, "optimal": 1, "clip": 2, "scale_a": 3, "scale_b": 3, "monotone": 4}


def _env(stream: int, *, lambda_risk: float | None = None) -> ExecutionEnv:
    return ExecutionEnv(
        MARKET,
        ORDER_SIZE,
        LAMBDA if lambda_risk is None else lambda_risk,
        root_seed=EXPERIMENT.seeds.root_seed,
        pool=M2_DIAGNOSTIC_POOL,
        stream_index=stream,
    )


def _drive(env, fractions) -> np.ndarray:
    """Step a `FractionAction`-wrapped env with a fraction sequence."""
    env.reset()
    info: dict = {}
    for fraction in fractions:
        _, _, terminated, _, info = env.step(np.array([fraction]))
    assert terminated, "the fraction sequence did not run the episode to the end"
    return info[EPISODE_KEY]["trajectory"]


def _fractions_of(trajectory) -> np.ndarray:
    """The fraction-of-remaining sequence that reproduces a trajectory."""
    x = np.asarray(trajectory, dtype=float)
    return (x[:-1] - x[1:]) / x[:-1]


# ---------------------------------------------------------------------------
# Task 5 — the action is a fraction of remaining inventory
# ---------------------------------------------------------------------------


def test_twap_is_one_over_the_bins_remaining():
    """The brief's sequence, stated: 1/13, 1/12, ..., 1.

    Not a constant fraction — selling a thirteenth of what is *left* every bin
    leaves inventory decaying geometrically, not linearly.
    """
    fractions = twap_fractions(N_BINS)
    assert fractions.size == N_BINS
    assert fractions[0] == pytest.approx(1.0 / 13.0, rel=0.0, abs=0.0)
    assert fractions[-1] == 1.0
    assert np.array_equal(fractions, 1.0 / np.arange(N_BINS, 0, -1, dtype=float))
    assert np.all(np.diff(fractions) > 0.0)


def test_the_twap_fractions_realise_the_twap_schedule_through_the_training_wrapper():
    """The path the agent trains on reproduces the oracle's TWAP exactly."""
    realised = _drive(FractionAction(_env(STREAMS["twap"])), twap_fractions(N_BINS))
    assert realised == pytest.approx(twap_trajectory(MARKET, ORDER_SIZE), rel=1e-12)


def test_the_twap_fractions_realise_the_twap_schedule_through_the_eval_policy():
    """...and so does the path it is *graded* on, through the shared rollout.

    Two different code paths — the gymnasium wrapper during training and
    :class:`~temper.agents.execution.FractionPolicy` under
    :func:`~temper.eval.rollout.run_episode` — must agree, or the agent would be
    graded in coordinates it did not train in.
    """
    policy = FractionPolicy(twap_fractions(N_BINS), ORDER_SIZE, "twap_fractions")
    result = run_episode(_env(STREAMS["twap"]), policy)
    assert result.trajectory == pytest.approx(
        twap_trajectory(MARKET, ORDER_SIZE), rel=1e-12
    )


def test_the_optimal_schedule_is_representable_in_the_same_coordinates():
    """The target is reachable too — otherwise "within epsilon" is unattainable.

    The optimum's fractions are close to constant in the interior (a sinh decays
    almost geometrically) and rise to 1 in the final bin, which is exactly the
    terminal condition the env enforces regardless.
    """
    optimum = optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA)
    fractions = _fractions_of(optimum)
    assert np.all((fractions >= 0.0) & (fractions <= 1.0))
    assert fractions[-1] == pytest.approx(1.0, rel=1e-12)

    realised = _drive(FractionAction(_env(STREAMS["optimal"])), fractions)
    assert realised == pytest.approx(optimum, rel=1e-9, abs=1e-6)
    assert schedule_moments(realised, MARKET, order_size=ORDER_SIZE).objective(
        LAMBDA
    ) == pytest.approx(
        schedule_moments(optimum, MARKET, order_size=ORDER_SIZE).objective(LAMBDA),
        rel=1e-12,
    )


def test_the_action_space_is_the_unit_interval_and_the_squash_is_a_clip():
    """A clip, so both endpoints are reachable — the last bin's TWAP fraction is 1."""
    assert FRACTION_SPACE.shape == (1,)
    assert float(FRACTION_SPACE.low[0]) == 0.0
    assert float(FRACTION_SPACE.high[0]) == 1.0

    assert as_fraction(-4.0) == 0.0
    assert as_fraction(0.0) == 0.0
    assert as_fraction(0.37) == 0.37
    assert as_fraction(1.0) == 1.0
    assert as_fraction(9.0) == 1.0
    assert as_fraction(np.array([0.25])) == 0.25
    with pytest.raises(ValueError):
        as_fraction(np.array([0.1, 0.2]))


def test_the_wrapper_and_the_eval_policy_convert_a_fraction_identically():
    """One conversion, shared. Two copies would be two action spaces."""
    observation = np.array([0.5384615384615384, 0.42])
    for fraction in (0.0, 0.13, 0.5, 1.0, -2.0, 3.0):
        expected = as_fraction(fraction) * 0.42 * ORDER_SIZE
        assert fraction_to_shares(fraction, observation, ORDER_SIZE) == expected


def test_the_env_clip_makes_every_reachable_trajectory_monotone():
    """The reachable set is sell-only, by the env's clip rather than by custom.

    This is what makes the objective exactly quadratic on everything M2 can
    grade, and therefore what makes the derived trajectory band exact rather
    than approximate (:class:`~temper.eval.reference.TrajectoryBand`). Driven
    with deliberately hostile actions — negative, enormous, and zero — because a
    policy mid-training will emit all three.
    """
    env = FractionAction(_env(STREAMS["monotone"]))
    hostile = [-5.0, 12.0, 0.0, -0.001, 4.0, 0.0, 0.0, 1.5, 0.0, -9.0, 0.0, 0.0, 0.0]
    realised = _drive(env, hostile)
    assert np.all(np.diff(realised) <= 0.0), "an action bought inventory back"
    assert realised[0] == ORDER_SIZE
    assert realised[-1] == 0.0


def test_a_nan_action_is_refused_rather_than_silently_clipped():
    """A NaN is a bug upstream; clipping it to zero would hide a broken policy."""
    env = FractionAction(_env(STREAMS["clip"]))
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array([float("nan")]))


# ---------------------------------------------------------------------------
# Task 4 — reward scaling is a fixed affine constant
# ---------------------------------------------------------------------------


def test_the_reward_scale_is_exactly_affine_over_a_whole_episode():
    """`scaled == scale * unscaled`, step by step, on the same shocks.

    Both envs are addressed to the same stream, so they see identical shocks and
    the comparison is exact rather than statistical.
    """
    scale = EXPERIMENT.reward_scale
    fractions = twap_fractions(N_BINS)

    plain = FractionAction(_env(STREAMS["scale_a"]))
    scaled = RewardScale(FractionAction(_env(STREAMS["scale_b"])), scale)
    plain.reset()
    scaled.reset()

    for fraction in fractions:
        action = np.array([fraction])
        _, bare, _, _, _ = plain.step(action)
        _, times, _, _, _ = scaled.step(action)
        assert times == scale * bare


def test_the_reward_scale_wrapper_carries_no_running_statistics():
    """Stateless by inspection: nothing about it changes as episodes go by.

    A running normaliser would make the reward non-stationary and seed
    dependent, which is objective drift by the back door — and it would be
    invisible in any single-episode test, which is why this one looks at the
    object rather than at its output.
    """
    wrapper = RewardScale(FractionAction(_env(STREAMS["scale_b"])), 0.02)
    before = dict(vars(wrapper))
    wrapper.reset()
    for _ in range(N_BINS):
        wrapper.step(np.array([0.3]))
    after = dict(vars(wrapper))
    assert before.keys() == after.keys()
    assert before["scale"] == after["scale"] == 0.02
    assert not hasattr(wrapper, "update")

    with pytest.raises(ValueError):
        RewardScale(FractionAction(_env(STREAMS["scale_b"])), 0.0)


def test_scaling_cannot_move_the_schedule_or_the_number_it_is_graded_on():
    """Whatever the training reward was multiplied by, the graded number is the same.

    The scale is an optimiser convenience: it changes the size of a reward and
    nothing about which schedule a fixed policy realises, so it cannot change the
    analytic objective computed from that schedule. Checked across four decades
    of scale, bitwise on the trajectory.
    """
    fractions = _fractions_of(optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA))
    graded: set[float] = set()
    schedules: list[np.ndarray] = []
    for scale in (1.0, EXPERIMENT.reward_scale, 1e-4, 50.0):
        env = RewardScale(FractionAction(_env(STREAMS["scale_b"])), scale)
        realised = _drive(env, fractions)
        schedules.append(realised)
        graded.add(
            schedule_moments(realised, MARKET, order_size=ORDER_SIZE).objective(LAMBDA)
        )

    for realised in schedules[1:]:
        assert np.array_equal(realised, schedules[0])
    assert len(graded) == 1, f"the graded objective moved with the scale: {graded}"
