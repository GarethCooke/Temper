"""M2 task 3 — the sanctioned control variate does exactly what it claims.

The variate's whole justification is that it is *exact*: M1a pinned the noise
identity ``C - E[cost] = -sum_k (n_k / X) walk_k`` per episode, and both factors
are published in the env's ``info``, so subtracting it removes the noise rather
than reducing it. Reward variance goes to **zero**, which is a claim with a
bitwise test — and this module is that test.

That distinction matters for what the milestone is allowed to say. A variate
that merely lowered variance would leave "RL under noise recovers AC" partly
intact. One that removes it entirely means the agent is trained on the expected
reward, and the honest sentence becomes "RL optimises a deterministic function".
The brief requires that restatement, and it is only correct if the reward really
is deterministic — so it is checked here rather than reasoned about.

Nothing in this module reports a number, so it draws from ``m2/diagnostic``.
"""

from __future__ import annotations

import numpy as np
import pytest

from temper.agents import FractionAction, FractionPolicy, RewardScale, twap_fractions
from temper.env import EPISODE_KEY, ExecutionEnv
from temper.eval import run_episode
from temper.eval.variate import DeterministicReward, deterministic_reward, noise_component
from temper.oracle import optimal_trajectory, schedule_moments
from temper.seeding import M2_DIAGNOSTIC_POOL

from .conftest import m2_experiment

EXPERIMENT = m2_experiment()
MARKET = EXPERIMENT.case.market
ORDER_SIZE = EXPERIMENT.case.order_size
LAMBDA = EXPERIMENT.lambda_risk
N_BINS = MARKET.n_bins

#: Relative band on the exact identities. The same order as M1a's noise identity
#: (1e-12), because it is the same identity: these sums cancel ~1e2 bps of
#: per-bin quantities down to ~1 bps, so the tolerance is relative to the summed
#: absolute terms rather than to the surviving total.
IDENTITY_RTOL = 1e-11


def _env(stream: int) -> ExecutionEnv:
    return ExecutionEnv(
        MARKET,
        ORDER_SIZE,
        LAMBDA,
        root_seed=EXPERIMENT.seeds.root_seed,
        pool=M2_DIAGNOSTIC_POOL,
        stream_index=stream,
    )


def _run(env, fractions) -> tuple[list[float], np.ndarray]:
    """Step a wrapped env through one episode; return its rewards and schedule."""
    env.reset()
    rewards: list[float] = []
    info: dict = {}
    for fraction in fractions:
        _, reward, _, _, info = env.step(np.array([fraction]))
        rewards.append(float(reward))
    return rewards, info[EPISODE_KEY]["trajectory"]


OPTIMAL_FRACTIONS = (
    lambda x: (x[:-1] - x[1:]) / x[:-1]
)(optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA))


# ---------------------------------------------------------------------------
# Zero variance, not low variance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fractions", [twap_fractions(N_BINS), OPTIMAL_FRACTIONS], ids=["twap", "optimal"]
)
def test_the_variate_reduces_the_reward_spread_to_floating_point_dust(fractions):
    """Two unrelated streams, the same actions, the same rewards to ~1 ulp.

    Not bitwise, and the reason is worth writing down rather than absorbing into
    a tolerance. The env forms ``weight * price_bps`` on the *summed* price, so
    subtracting ``weight * walk`` afterwards rounds differently from forming
    ``weight * (price_bps - walk)`` in one operation. The residue is a couple of
    ulps of the noise term — about fifteen orders of magnitude below the noise
    itself, which is what "variance reduced to zero" means in float64. Removing
    it would mean changing the env, which M2 may not do.

    The ratio is asserted rather than an absolute band, so the check keeps its
    meaning if the case, the schedule or the shocks change.
    """
    clean_a, _ = _run(DeterministicReward(FractionAction(_env(200))), fractions)
    clean_b, _ = _run(DeterministicReward(FractionAction(_env(201))), fractions)
    noisy_a, _ = _run(FractionAction(_env(200)), fractions)
    noisy_b, _ = _run(FractionAction(_env(201)), fractions)

    clean_spread = float(np.max(np.abs(np.array(clean_a) - np.array(clean_b))))
    noisy_spread = float(np.max(np.abs(np.array(noisy_a) - np.array(noisy_b))))

    assert noisy_spread > 1.0, (
        f"the two streams differ by only {noisy_spread:.4f} bps per step; this "
        "check needs real noise to be removing anything"
    )
    assert clean_spread <= 1e-12 * noisy_spread, (
        f"the variate left {clean_spread:.3e} bps of stream-dependence against "
        f"{noisy_spread:.3e} without it"
    )


def test_without_the_variate_the_same_actions_pay_very_different_rewards():
    """Non-vacuity: the noise the variate removes is enormous.

    The per-episode cost standard deviation on this case is ~95 bps against a
    ~2.4 bps objective — the 1:70 ratio that makes task 3 the milestone's real
    risk.
    """
    fractions = twap_fractions(N_BINS)
    first, _ = _run(FractionAction(_env(202)), fractions)
    second, _ = _run(FractionAction(_env(203)), fractions)
    assert first != second
    spread = abs(sum(first) - sum(second))
    assert spread > 1.0, (
        f"two episodes differed by only {spread:.4f} bps; the noise this variate "
        "removes should be an order of magnitude larger than the objective"
    )


# ---------------------------------------------------------------------------
# It removes exactly the noise, and nothing else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fractions", [twap_fractions(N_BINS), OPTIMAL_FRACTIONS], ids=["twap", "optimal"]
)
def test_the_deterministic_reward_sums_to_minus_the_analytic_objective(fractions):
    """The strongest form of the claim: what is left is exactly ``-(E + lambda V)``.

    This is invariant 7 restated for the fallback estimator. If the variate
    removed a little too much or too little, the agent would be optimising a
    functional the oracle does not minimise, and the rediscovery claim would be
    void whatever the trajectory looked like.
    """
    rewards, trajectory = _run(DeterministicReward(FractionAction(_env(204))), fractions)
    objective = schedule_moments(trajectory, MARKET, order_size=ORDER_SIZE).objective(
        LAMBDA
    )
    scale = sum(abs(r) for r in rewards)
    assert sum(rewards) == pytest.approx(-objective, rel=0.0, abs=IDENTITY_RTOL * scale)


def test_the_variate_is_the_episode_level_noise_identity_term_by_term():
    """Checked against `noise_component`, assembled from the recorder's own arrays.

    The wrapper subtracts a per-step quantity; M1a's identity is stated per
    episode. Comparing the two closes the gap between "each line looks right" and
    "the sum is the identity".
    """
    policy = FractionPolicy(twap_fractions(N_BINS), ORDER_SIZE, "twap_fractions")
    recorded = run_episode(_env(205), policy)
    noise = noise_component(recorded.shares, recorded.walks, ORDER_SIZE)

    expected = schedule_moments(
        recorded.trajectory, MARKET, order_size=ORDER_SIZE
    ).expected
    scale = float(np.sum(np.abs(recorded.shortfalls)))
    assert recorded.cost_bps - expected == pytest.approx(
        -noise, rel=0.0, abs=IDENTITY_RTOL * scale
    )

    plain, _ = _run(FractionAction(_env(205)), twap_fractions(N_BINS))
    clean, _ = _run(DeterministicReward(FractionAction(_env(205))), twap_fractions(N_BINS))
    assert sum(plain) - sum(clean) == pytest.approx(
        noise, rel=0.0, abs=IDENTITY_RTOL * scale
    )


def test_the_variate_leaves_the_realised_schedule_untouched():
    """It changes the estimator, not the world: same actions, same trajectory."""
    fractions = twap_fractions(N_BINS)
    _, plain = _run(FractionAction(_env(206)), fractions)
    _, clean = _run(DeterministicReward(FractionAction(_env(206))), fractions)
    assert np.array_equal(plain, clean)


def test_the_variate_composes_under_the_reward_scale_in_that_order():
    """Wrapped inside the scaling, so the variate works in the env's own bps.

    The factory builds ``RewardScale(DeterministicReward(FractionAction(env)))``.
    If the order were reversed the variate would have to undo the scale, which is
    one more place for two constants to disagree.
    """
    scale = EXPERIMENT.reward_scale
    fractions = twap_fractions(N_BINS)
    bare, _ = _run(DeterministicReward(FractionAction(_env(207))), fractions)
    scaled, _ = _run(
        RewardScale(DeterministicReward(FractionAction(_env(207))), scale), fractions
    )
    assert scaled == [pytest.approx(scale * r, rel=1e-15) for r in bare]


def test_the_factory_form_is_the_wrapper():
    env = deterministic_reward(FractionAction(_env(208)))
    assert isinstance(env, DeterministicReward)
    assert env.order_size == ORDER_SIZE


def test_the_noise_component_helper_refuses_mismatched_arrays():
    with pytest.raises(ValueError):
        noise_component([1.0, 2.0], [1.0], ORDER_SIZE)


# ---------------------------------------------------------------------------
# ...and the committed config says which estimator produced the result
# ---------------------------------------------------------------------------


def test_the_committed_config_states_its_estimator_and_its_claim():
    """The switch and the sentence travel together (task 3's amendment rule)."""
    estimator = EXPERIMENT.estimator
    assert isinstance(estimator.control_variate, bool)
    assert estimator.claim, "the config must state the claim its estimator supports"
    if estimator.control_variate:
        assert "deterministic" in estimator.claim.lower(), (
            "a control-variate run trains on the expected reward and must say so"
        )
    else:
        assert "sampled" in estimator.claim.lower()
