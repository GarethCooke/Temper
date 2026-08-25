"""M5 task 4 — ``E[cost | s]``, and three ways of not believing the derivation.

The closed form is one line and the milestone rests entirely on it:

.. code::

    E[cost | s] = A_pow sum_k w_k^(1+beta)     temporary impact
                + lambda B sum_k h_k^2         inventory risk
                - A rho sum_k h_k s_{k-1}      alpha
                + permanent + half-spread      schedule-invariant

Three checks, and none of them is the derivation again in other words.

**The index, on paper.** That formula pairs the inventory held *before* bin ``k``
with the signal shown ``one bin earlier``, and this milestone has already spent a
session on an off-by-one in exactly that relationship. So it is asserted on a
three-bin case with a schedule chosen by hand, against a number written out term
by term — the M1 idiom. A shifted index survives every aggregate check here (the
level is right, the variance is right, usually even the sign) and dies on a case
small enough to do on paper.

**The formula against sampled prices.** The brief files this under task 5. A cheap
version runs here, at the moment the closed form is written, because if the formula
is wrong then task 5 is a differential verifying a wrong derivation against a wrong
tier — and ``docs/house-notes.md``'s law about sunk work applies to derivations as
much as to drivers. One pinned signal path, thousands of price draws, and the
sampled mean has to land on the closed form. The full-tier version stays task 5's.

**The pairing, measured rather than predicted.** The brief predicts the antithetic
pair cancels the part of the alpha term that is linear in ``s`` and keeps the part
that is quadratic, and that it should therefore help *more* here than in M4b. The
pair exists as of task 3, so this is a measurement — and it came back saying
something sharper and more useful than the prediction, which is recorded here
because task 6 is the run one least wants to discover it in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from temper.env import EPISODE_KEY, ExecutionEnv, impact_for, signal_stream
from temper.eval.antithetic import AntitheticPair, MirrorEnv, NegatedSignal
from temper.agents.baselines import twap_policy
from temper.eval.conditional import ConditionalCosts, signal_costs, signal_rollouts
from temper.oracle import (
    POWER_LAW_ENCODING,
    Market,
    OneStepSignal,
    SymbolParams,
    alpha_coefficient,
    conditional_alpha_bps,
    cost_moments,
    inventory_penalty_scale,
    power_law_charge,
    schedule_invariant_bps,
    signal_path_objective_bps,
    twap_trajectory,
)
from temper.seeding import M5_DIFFERENTIAL_POOL, SIGNAL_TRAIN_POOL

#: A rho the checks below can resolve. The milestone trains at 0.01; a *formula*
#: is right or wrong at every rho, and testing it at one the arithmetic can see is
#: the difference between a check and a coin toss.
CHECK_RHO = 0.4


def _market(n_bins: int = 13) -> Market:
    return Market(
        params=SymbolParams(
            adv=6e7, sigma=0.0155, half_spread=0.3, eta=0.142, gamma=0.314
        ),
        horizon_hours=0.5 * n_bins,
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# 1. The index, on a case small enough to do on paper
# ---------------------------------------------------------------------------


def test_the_alpha_term_pairs_the_hand_computed_inventory_with_the_hand_computed_signal():
    """Three bins, a schedule chosen by hand, and the alpha term written out.

    ``X = 1000`` over three bins with trades ``(500, 300, 200)`` gives inventory
    ``x = (1000, 500, 200, 0)`` and holdings ``h = (1.0, 0.5, 0.2)``. With the
    signal path ``s = (2, -3, 7)`` the alpha term is, term by term:

    .. code::

        -A rho [ h_1 s_0 + h_2 s_1 ]
        = -A rho [ 0.5 * 2 + 0.2 * (-3) ]
        = -A rho [ 1.0 - 0.6 ]
        = -A rho * 0.4

    Three things a shifted index would break, and every one of them is visible in
    that arithmetic. ``h_0 = 1`` never appears, because ``xi_0`` is predicted by
    nothing and every schedule holds the whole order through it. ``s_2 = 7`` never
    appears, because the signal shown at the last decision point predicts a shock
    after the horizon. And the two terms that do appear carry *different* weights
    with *different* signs, so shifting either index by one changes the answer
    rather than rearranging it: pairing ``h`` with the same-indexed ``s`` would
    give ``1.0*2 + 0.5*(-3) + 0.2*7 = 1.9``, nearly five times this and the other
    sign of interesting.
    """
    market = _market(3)
    trajectory = np.array([1000.0, 500.0, 200.0, 0.0])
    signals = np.array([2.0, -3.0, 7.0])
    amplitude = alpha_coefficient(market)

    by_hand = -amplitude * CHECK_RHO * 0.4
    computed = conditional_alpha_bps(
        trajectory, market, signals, OneStepSignal(CHECK_RHO)
    )
    assert computed == pytest.approx(by_hand, rel=1e-15)

    # The two shifted indices, spelled out, so the case has teeth rather than a
    # single number that happens to match.
    same_index = -amplitude * CHECK_RHO * (1.0 * 2.0 + 0.5 * -3.0 + 0.2 * 7.0)
    two_ahead = -amplitude * CHECK_RHO * (0.5 * -3.0 + 0.2 * 7.0)
    assert computed != pytest.approx(same_index, rel=1e-6)
    assert computed != pytest.approx(two_ahead, rel=1e-6)

    # And the whole conditional cost is the four named terms, each computed
    # independently of the function under test.
    weights = -np.diff(trajectory) / trajectory[0]
    holdings = trajectory[:-1] / trajectory[0]
    lambda_risk = 1e-3
    expected = (
        power_law_charge(market, trajectory[0]).cost_bps(weights)
        + lambda_risk * inventory_penalty_scale(market) * float(np.sum(holdings**2))
        + by_hand
        + schedule_invariant_bps(market, trajectory[0])
    )
    moments = cost_moments(
        trajectory, market, signal=OneStepSignal(CHECK_RHO), signals=signals
    )
    assert moments.objective(lambda_risk) == pytest.approx(expected, rel=1e-12)


def test_the_last_signal_and_the_first_holding_are_absent_by_construction():
    """The two ends of the sum, asserted rather than left to the arithmetic.

    Perturbing ``s`` at the last decision point must not move the alpha term at
    all, and neither must changing the order size (which scales ``h_0`` and
    nothing else about the holdings). Both are exact statements, and a shifted
    index breaks one or the other.
    """
    market = _market(3)
    trajectory = np.array([1000.0, 500.0, 200.0, 0.0])
    law = OneStepSignal(CHECK_RHO)

    base = conditional_alpha_bps(trajectory, market, np.array([2.0, -3.0, 7.0]), law)
    moved_last = conditional_alpha_bps(
        trajectory, market, np.array([2.0, -3.0, -99.0]), law
    )
    assert moved_last == base, "the final decision point's signal reached the cost"

    moved_first = conditional_alpha_bps(
        trajectory, market, np.array([-99.0, -3.0, 7.0]), law
    )
    assert moved_first != base, (
        "the first decision point's signal did NOT reach the cost, so the sum "
        "starts a bin too late"
    )


def test_an_already_committed_signal_reaches_a_different_bin():
    """The lag is honoured in the grader, not only in the env.

    At ``bins_ahead = 0`` the same path must produce
    ``-A rho [h_0 s_0 + h_1 s_1 + h_2 s_2]`` — all three terms, starting at the
    whole order. The grading formula and the seam have to agree about the index or
    a policy would be scored against a world it did not trade in.
    """
    market = _market(3)
    trajectory = np.array([1000.0, 500.0, 200.0, 0.0])
    signals = np.array([2.0, -3.0, 7.0])
    amplitude = alpha_coefficient(market)

    committed = conditional_alpha_bps(
        trajectory, market, signals, OneStepSignal(CHECK_RHO, bins_ahead=0)
    )
    assert committed == pytest.approx(
        -amplitude * CHECK_RHO * (1.0 * 2.0 + 0.5 * -3.0 + 0.2 * 7.0), rel=1e-15
    )


def test_the_graded_route_and_the_vectorised_twin_agree():
    """``cost_moments`` per schedule against ``signal_path_objective_bps`` batched.

    M4b's rule, unchanged: the reference's bounds use the fast route and the
    *graded* number uses the one every earlier milestone graded through, and the
    two are pinned so a fast twin cannot quietly become the definition.
    """
    market = _market()
    order_size, lambda_risk = 100_000.0, 10.0**-3.5
    law = OneStepSignal(CHECK_RHO)
    trajectory = twap_trajectory(market, order_size)
    rng = np.random.default_rng(20260825)
    signals = law.draw(rng, (64, market.n_bins))
    weights = -np.diff(trajectory) / order_size

    fast = signal_path_objective_bps(
        weights, signals, market, order_size, lambda_risk, law
    )
    slow = np.array(
        [
            cost_moments(trajectory, market, signal=law, signals=path).objective(
                lambda_risk
            )
            for path in signals
        ]
    )
    assert np.max(np.abs(fast - slow)) < 1e-12


def test_cost_moments_without_a_signal_is_bit_identical_to_before():
    """No M4a or earlier number moves because a later milestone widened a signature.

    ``alpha`` defaults to a float zero and ``expected`` adds it, so the equality is
    a property of IEEE arithmetic rather than of a branch that could be edited.
    """
    market = _market()
    for trajectory in (
        twap_trajectory(market, 100_000.0),
        np.array([100_000.0] + [100_000.0 * (1 - (k + 1) / 13) for k in range(13)]),
    ):
        plain = cost_moments(trajectory, market)
        assert plain.alpha == 0.0
        assert plain.expected == plain.temporary + plain.permanent + plain.spread
        zero_rho = cost_moments(
            trajectory,
            market,
            signal=OneStepSignal(0.0),
            signals=np.zeros(market.n_bins),
        )
        assert zero_rho.expected == plain.expected
        assert zero_rho.variance == plain.variance

    with pytest.raises(ValueError, match="go together"):
        cost_moments(twap_trajectory(market, 100_000.0), market, signal=OneStepSignal(0.1))


# ---------------------------------------------------------------------------
# 2. The closed form against sampled prices — task 5's check, cheaply, now
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.0, CHECK_RHO])
def test_the_closed_form_is_the_mean_over_price_draws_at_a_pinned_signal(rho):
    """Pin one signal path, sample the prices, and require convergence.

    The brief files the full version under task 5. It runs here because if this is
    wrong then task 5 is a differential verifying a wrong derivation against a
    wrong tier, and the cost of finding out then rather than now is a tier rather
    than a test.

    ``rho = 0`` is the control: the alpha term is exactly zero and the check
    reduces to M1's own statement that the env's realised cost has the mean the
    oracle predicts. At ``rho = 0.4`` the alpha term is tens of bps and the
    difference between the two is what the sampled mean has to find.

    The interval is over **price** draws, which is the one place in this milestone
    they are sampled at all. Everywhere else there is no price sampling: that is
    the whole point of conditioning.
    """
    market = _market()
    order_size, lambda_risk = 100_000.0, 10.0**-3.5
    episodes = 12_000
    law = OneStepSignal(rho)
    # Pinned, so every episode draws the SAME signal path while the price stream
    # moves. Without the pin the two would move together and the comparison would
    # be against a different conditional expectation each episode.
    stream = signal_stream(law, SIGNAL_TRAIN_POOL).pinned_to(77)
    env = ExecutionEnv(
        market,
        order_size,
        lambda_risk,
        temporary_impact=impact_for(POWER_LAW_ENCODING, market, order_size),
        signal=stream,
        root_seed=20260825,
        pool=M5_DIFFERENTIAL_POOL,
        stream_index=0,
    )
    schedule = -np.diff(twap_trajectory(market, order_size))

    realised = np.empty(episodes)
    for episode in range(episodes):
        env.reset(seed=episode)
        for shares in schedule:
            _, _, _, _, info = env.step(float(shares))
        realised[episode] = info[EPISODE_KEY]["cost_bps"]
    signals = env.signals

    trajectory = twap_trajectory(market, order_size)
    closed = cost_moments(trajectory, market, signal=law, signals=signals).expected
    half_width = 1.96 * realised.std(ddof=1) / math.sqrt(episodes)
    assert abs(realised.mean() - closed) < half_width, (
        f"sampled {realised.mean():.4f} vs closed form {closed:.4f} bps, "
        f"half-width {half_width:.4f}"
    )

    # Non-vacuity: at rho > 0 the alpha term has to be big enough that the check
    # could have failed. A test that cannot see the term it is verifying is not
    # verifying it.
    alpha = conditional_alpha_bps(trajectory, market, signals, law)
    if rho:
        assert abs(alpha) > 5.0 * half_width, (
            f"the alpha term is {alpha:.4f} bps against a half-width of "
            f"{half_width:.4f}; this check cannot resolve it"
        )
        without = cost_moments(trajectory, market).expected
        assert abs(realised.mean() - without) > 3.0 * half_width, (
            "the signal-free closed form is also inside the interval, so this "
            "check would pass on a formula with no alpha term at all"
        )
    else:
        assert alpha == 0.0


# ---------------------------------------------------------------------------
# 3. The pairing, measured
# ---------------------------------------------------------------------------


def _pair_average(mirror_signal, market, order_size, lambda_risk, law, episodes):
    """Average the two halves' realised cost, under a chosen mirror arrangement."""
    impact = impact_for(POWER_LAW_ENCODING, market, order_size)
    stream = signal_stream(law, SIGNAL_TRAIN_POOL)
    kwargs = dict(
        temporary_impact=impact,
        root_seed=20260825,
        pool=M5_DIFFERENTIAL_POOL,
        stream_index=0,
    )
    primary = ExecutionEnv(market, order_size, lambda_risk, signal=stream, **kwargs)
    mirror = MirrorEnv(market, order_size, lambda_risk, signal=mirror_signal, **kwargs)
    schedule = -np.diff(twap_trajectory(market, order_size))
    trajectory = twap_trajectory(market, order_size)

    averaged = np.empty(episodes)
    primary_half = np.empty(episodes)
    closed = np.empty(episodes)
    for episode in range(episodes):
        primary.reset(seed=episode)
        mirror.reset(seed=episode)
        for shares in schedule:
            _, _, _, _, info = primary.step(float(shares))
            _, _, _, _, m_info = mirror.step(float(shares))
        primary_half[episode] = info[EPISODE_KEY]["cost_bps"]
        averaged[episode] = 0.5 * (
            primary_half[episode] + m_info[EPISODE_KEY]["cost_bps"]
        )
        closed[episode] = cost_moments(
            trajectory, market, signal=law, signals=primary.signals
        ).expected
    return primary_half, averaged, closed


def test_the_pair_average_is_the_conditional_expectation_itself():
    """The brief's prediction, measured — and the mechanism is not the one it named.

    Predicted: *the pairing cancels the part of the alpha term that is linear in*
    ``s`` *and keeps the part that is quadratic*, and should therefore help more
    than in M4b. The conclusion is right and the mechanism is not, because it
    assumes each half acts on its own observation. This pairing hands **one action
    to both halves**, and that changes the answer completely.

    Sharing the signal and negating only the price gives the mirror
    ``rho s - sqrt(1 - rho^2) e``, so the two shocks average to ``rho s`` — the
    conditional mean — and the averaged reward is ``E[cost | s]`` *itself*. Not
    the quadratic part of it: the whole of it, with the unpredictable half of the
    price noise removed entirely and zero residual variance given ``s``.

    That is "helps more than in M4b" in the strongest available form. M4b's pair
    could not remove the liquidity noise at all and the agent trained through
    3.269e-02 bps² per update; here the conditional variance is zero and the
    training reward and the grading formula are the same object.
    """
    market = _market()
    order_size, lambda_risk = 100_000.0, 10.0**-3.5
    law = OneStepSignal(CHECK_RHO)
    stream = signal_stream(law, SIGNAL_TRAIN_POOL)

    primary_half, averaged, closed = _pair_average(
        stream, market, order_size, lambda_risk, law, episodes=600
    )
    worst = float(np.max(np.abs(averaged - closed)))
    assert worst < 1e-9, f"the pair average is not E[cost | s]: worst {worst:.3e} bps"
    assert float(np.corrcoef(averaged, closed)[0, 1]) == pytest.approx(1.0, abs=1e-12)

    # The variance the pairing removed, measured on the same episodes rather than
    # inferred: what a sampled-reward agent would have trained on, against what
    # this one does.
    assert primary_half.std(ddof=1) > 50.0
    residual = averaged - closed
    assert residual.std(ddof=1) < 1e-9, (
        "the averaged reward carries residual noise given s, so the pairing is not "
        "exact and the training budget has to account for it"
    )


def test_a_signal_negating_mirror_makes_the_estimator_blind():
    """The arrangement task 3 chose, and the measurement that rejected it.

    Negating the signal makes the mirror's shock the exact negation of the
    primary's — which is why it looked right — and therefore cancels the
    *predictable* half of the price along with the unpredictable half. The averaged
    reward becomes the shock-free cost: constant across signal paths, uncorrelated
    with the alpha term, and containing no reason whatsoever for an agent to tilt.

    Kept as a committed measurement rather than a paragraph, because it is the
    kind of choice that looks equally defensible either way until somebody runs it,
    and because task 6 is the run one least wants to discover it in.
    """
    market = _market()
    order_size, lambda_risk = 100_000.0, 10.0**-3.5
    law = OneStepSignal(CHECK_RHO)
    negated = signal_stream(NegatedSignal(law), SIGNAL_TRAIN_POOL)

    _, averaged, closed = _pair_average(
        negated, market, order_size, lambda_risk, law, episodes=600
    )
    alpha = closed - closed.mean()
    assert alpha.std(ddof=1) > 1.0, "the signal paths barely vary; nothing is measured"

    assert averaged.std(ddof=1) < 1e-9, (
        "the signal-negating average was supposed to be constant across signal "
        "paths; if it is not, the argument for sharing the signal is different"
    )
    # A correlation against a constant is undefined, which is precisely the
    # finding: there is nothing in the averaged reward for one to be taken with.
    assert float(np.var(averaged)) == pytest.approx(0.0, abs=1e-18)


def test_the_pair_wrapper_agrees_with_the_hand_rolled_arrangement():
    """`AntitheticPair` is the thing training uses; the measurement above is by hand.

    So the two are pinned against each other: the wrapper's averaged reward, over
    the env's own reward scale, must be the same conditional expectation the hand
    -rolled pair produced. Otherwise the measurement is about a fixture.
    """
    market = _market()
    order_size, lambda_risk = 100_000.0, 10.0**-3.5
    law = OneStepSignal(CHECK_RHO)
    pair = AntitheticPair(
        ExecutionEnv(
            market,
            order_size,
            lambda_risk,
            temporary_impact=impact_for(POWER_LAW_ENCODING, market, order_size),
            signal=signal_stream(law, SIGNAL_TRAIN_POOL),
            root_seed=20260825,
            pool=M5_DIFFERENTIAL_POOL,
            stream_index=0,
        )
    )
    schedule = -np.diff(twap_trajectory(market, order_size))
    trajectory = twap_trajectory(market, order_size)

    worst = 0.0
    for episode in range(200):
        pair.reset(seed=episode)
        signals = pair.primary.signals
        total = 0.0
        for shares in schedule:
            _, reward, _, _, _ = pair.step(float(shares))
            total += float(reward)
        moments = cost_moments(trajectory, market, signal=law, signals=signals)
        worst = max(worst, abs(-total - moments.objective(lambda_risk)))
    assert worst < 1e-9, (
        f"the pair's averaged reward is not -E[cost | s] - lambda V: worst "
        f"{worst:.3e} bps"
    )


# ---------------------------------------------------------------------------
# The grading path end to end
# ---------------------------------------------------------------------------


def test_signal_rollouts_and_costs_close_the_identity_they_report():
    """``objective == impact + risk + alpha + invariant``, per path, through the grader."""
    market = _market()
    order_size, lambda_risk = 100_000.0, 10.0**-3.5
    law = OneStepSignal(CHECK_RHO)
    trajectory = twap_trajectory(market, order_size)

    trajectories, signals = signal_rollouts(
        twap_policy(market, order_size),
        market,
        order_size,
        lambda_risk,
        temporary_impact=impact_for(POWER_LAW_ENCODING, market, order_size),
        signal=signal_stream(law, SIGNAL_TRAIN_POOL),
        root_seed=20260825,
        pool=M5_DIFFERENTIAL_POOL,
        paths=256,
    )
    costs = signal_costs(trajectories, signals, market, lambda_risk, law)
    assert isinstance(costs, ConditionalCosts)
    closed = costs.impact + costs.risk + costs.alpha + costs.invariant
    assert np.max(np.abs(costs.objective - closed)) < 1e-12
    assert np.array_equal(costs.execution, costs.impact + costs.risk)
    assert np.allclose(trajectories[:, 0], order_size)
    assert np.allclose(trajectories[:, -1], 0.0)
