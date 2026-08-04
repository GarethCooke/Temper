"""Mathematical invariants of the oracle, independent of the goldens.

The goldens prove Temper's closed forms agree with FrontierView's. They cannot
prove either is *right* — both could share a misconception. These tests attack
the closed forms from the other side: closed form against direct solve, analytic
identity against explicit accumulation, claimed optimum against perturbation.

Every failure here is a bug in `temper/oracle`, not in the vendor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from temper.oracle import (
    BPS,
    SINH_OVERFLOW_KT,
    TRADING_HOURS_PER_DAY,
    Market,
    SymbolParams,
    ac_frontier,
    ac_kappa,
    ac_trajectory,
    cost_moments,
    linear_cost_moments,
    linearised_eta,
    objective_curvature,
    optimal_frontier,
    optimal_kappa,
    optimal_trajectory,
    optimal_trajectory_by_solve,
    permanent_cost_bps,
    sinh_trajectory,
    trades,
    twap_trajectory,
)

AAPL = SymbolParams(adv=60_000_000, sigma=0.0155, half_spread=0.30, eta=0.142, gamma=0.314)
JPM = SymbolParams(adv=12_000_000, sigma=0.0185, half_spread=0.50, eta=0.142, gamma=0.314)

#: (id, market, order size) — several symbols, horizons and bin counts, including
#: a grid that is not FrontierView's half-hour default.
MARKETS = [
    ("aapl-full-day", Market.for_horizon(AAPL, 6.5), 100_000.0),
    ("jpm-full-day", Market.for_horizon(JPM, 6.5), 500_000.0),
    ("aapl-short", Market.for_horizon(AAPL, 1.0), 40_000.0),
    ("jpm-odd-grid", Market(params=JPM, horizon_hours=3.0, n_bins=7), 120_000.0),
]

LAMBDAS = [0.0, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1]


@pytest.fixture(params=[(m, x) for _, m, x in MARKETS], ids=[i for i, _, _ in MARKETS])
def market_case(request):
    """A (market, order_size) pair covering several grids and symbols."""
    return request.param


def _linear_objective(trajectory, market, order_size, lambda_risk) -> float:
    """The frozen objective, evaluated under the model the closed form solves."""
    eta_tilde = linearised_eta(market, order_size)
    return linear_cost_moments(trajectory, market, eta_tilde).objective(lambda_risk)


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lambda_risk", LAMBDAS)
def test_schedules_liquidate_and_never_buy_back(market_case, lambda_risk):
    """Inventory is monotone non-increasing and ends at (effectively) zero."""
    market, order_size = market_case
    for trajectory in (
        twap_trajectory(market, order_size),
        ac_trajectory(market, order_size, lambda_risk),
        optimal_trajectory(market, order_size, lambda_risk),
    ):
        executed = trades(trajectory, market)
        assert np.all(executed >= 0.0), "schedule buys back"
        assert trajectory[0] == pytest.approx(order_size, rel=1e-15)
        assert abs(trajectory[-1]) <= 1e-9 * order_size, "order not liquidated"
        assert executed.sum() == pytest.approx(order_size, rel=1e-9)


def test_expected_cost_is_the_sum_of_its_parts(market_case):
    """No fourth component hiding in the total."""
    market, order_size = market_case
    moments = cost_moments(ac_trajectory(market, order_size, 1e-5), market)
    assert moments.expected == pytest.approx(
        moments.temporary + moments.permanent + moments.spread, rel=1e-15
    )


def test_trajectory_length_is_validated(market_case):
    """A wrong-length trajectory is an error, not a silently truncated answer."""
    market, order_size = market_case
    with pytest.raises(ValueError, match="points for an"):
        cost_moments(np.linspace(order_size, 0.0, market.n_bins), market)


# ---------------------------------------------------------------------------
# Permanent cost: closed form vs explicit accumulation
# ---------------------------------------------------------------------------


def _permanent_cost_by_accumulation(trajectory, market) -> float:
    """The midpoint rule spelled out, one bin at a time.

    Deliberately a plain loop over the definition — the closed form in the
    oracle is what this exists to check, so it must not share its algebra.
    """
    x = np.asarray(trajectory, dtype=float)
    order_size = x[0]
    total = drift = 0.0
    for previous, current in zip(x[:-1], x[1:]):
        executed = previous - current
        rate = executed / market.dt
        own = (
            market.params.gamma
            * market.params.sigma
            * abs(rate / market.v_hourly)
            * BPS
            * market.dt
        )
        total += (drift + own / 2.0) * executed / order_size
        drift += own
    return total


@pytest.mark.parametrize("lambda_risk", LAMBDAS)
def test_permanent_cost_closed_form_matches_accumulation(market_case, lambda_risk):
    """The telescoped identity and the bin-by-bin loop agree."""
    market, order_size = market_case
    for trajectory in (
        twap_trajectory(market, order_size),
        ac_trajectory(market, order_size, lambda_risk),
    ):
        assert permanent_cost_bps(trajectory, market) == pytest.approx(
            _permanent_cost_by_accumulation(trajectory, market), rel=1e-12
        )


def test_permanent_cost_is_schedule_invariant(market_case):
    """Linear g makes permanent cost gamma*sigma*X/(2*v_hourly), whatever the shape."""
    market, order_size = market_case
    analytic = (
        market.params.gamma * market.params.sigma * BPS * order_size
        / (2.0 * market.v_hourly)
    )
    shapes = [twap_trajectory(market, order_size)]
    shapes += [ac_trajectory(market, order_size, lam) for lam in (1e-7, 1e-5, 1e-3)]
    for trajectory in shapes:
        assert permanent_cost_bps(trajectory, market) == pytest.approx(analytic, rel=1e-12)


def test_permanent_cost_handles_a_non_monotone_path(market_case):
    """A path that buys back falls off the closed form onto the accumulation."""
    market, order_size = market_case
    trajectory = twap_trajectory(market, order_size).copy()
    trajectory[1] = trajectory[0] * 1.1  # buy first, then liquidate
    assert permanent_cost_bps(trajectory, market) == pytest.approx(
        _permanent_cost_by_accumulation(trajectory, market), rel=1e-12
    )


# ---------------------------------------------------------------------------
# The exact optimum really is the optimum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lambda_risk", LAMBDAS)
def test_closed_form_optimum_matches_direct_solve(market_case, lambda_risk):
    """sinh(arccosh) closed form == solving the first-order conditions."""
    market, order_size = market_case
    closed = optimal_trajectory(market, order_size, lambda_risk)
    solved = optimal_trajectory_by_solve(market, order_size, lambda_risk)
    worst = float(np.max(np.abs(closed - solved))) / order_size
    assert worst <= 1e-9, f"closed form and direct solve differ by {worst:.3e} of X"


@pytest.mark.parametrize("lambda_risk", [1e-9, 1e-6, 1e-4, 1e-2])
def test_optimum_beats_every_perturbation_of_itself(market_case, lambda_risk):
    """The frozen objective is strictly worse anywhere near the claimed optimum.

    The linearised objective is a strictly convex quadratic in the interior
    inventory levels, so *any* feasible perturbation must increase it. Steps are
    scaled to the order size, not to the smallest bin: a strongly front-loaded
    optimum has bins of ~1e-5 shares, and perturbations that small move the
    objective by less than a float can represent, which would make this test
    pass on noise. Each perturbed path is projected back onto a sell schedule so
    permanent cost stays constant across the comparison and cannot mask a
    difference in the quadratic part.
    """
    market, order_size = market_case
    optimum = optimal_trajectory(market, order_size, lambda_risk)
    best = _linear_objective(optimum, market, order_size, lambda_risk)

    rng = np.random.default_rng(20260804)
    for step in (1e-3, 1e-2, 1e-1):
        for _ in range(8):
            perturbed = optimum.copy()
            perturbed[1:-1] += rng.uniform(-1.0, 1.0, market.n_bins - 1) * step * order_size
            # Monotone, non-negative, same endpoints: still a feasible schedule.
            perturbed = np.clip(np.minimum.accumulate(perturbed), 0.0, None)
            assert np.all(trades(perturbed, market) >= 0.0), "perturbation left the domain"
            assert _linear_objective(perturbed, market, order_size, lambda_risk) > best


@pytest.mark.parametrize("lambda_risk", [1e-7, 1e-5, 1e-3])
def test_optimum_beats_twap_and_the_vendored_schedule(market_case, lambda_risk):
    """Both baselines score worse on the objective the optimum minimises."""
    market, order_size = market_case
    best = _linear_objective(
        optimal_trajectory(market, order_size, lambda_risk),
        market,
        order_size,
        lambda_risk,
    )
    for name, trajectory in (
        ("twap", twap_trajectory(market, order_size)),
        ("vendored ac", ac_trajectory(market, order_size, lambda_risk)),
    ):
        assert _linear_objective(trajectory, market, order_size, lambda_risk) >= best, (
            f"{name} scores better than the claimed optimum"
        )


# ---------------------------------------------------------------------------
# The two kappas (ARCHITECTURE.md §9, 2026-08-04)
# ---------------------------------------------------------------------------


def test_the_two_kappas_are_not_the_same_number(market_case):
    """Guard rail: nobody may quietly collapse one convention into the other.

    If this test starts passing trivially it is because someone "simplified" the
    oracle and silently changed which schedule M2 grades against.
    """
    market, order_size = market_case
    vendored = ac_kappa(market, order_size, 1e-5)
    exact = optimal_kappa(market, order_size, 1e-5)
    assert not math.isclose(vendored, exact, rel_tol=1e-3)


def test_exact_kappa_reduces_to_the_vendored_one_under_a_known_rescaling(market_case):
    """optimal_kappa^2 / ac_kappa^2 -> BPS / (X * dt) as kappa*dt -> 0.

    This pins the *relationship* between the conventions, which is what makes
    "the vendored schedule is the optimum at a rescaled lambda" a checkable
    statement rather than a hand-wave.
    """
    market, order_size = market_case
    lambda_risk = 1e-9  # small enough that discrete and continuum kappas agree
    ratio = (
        optimal_kappa(market, order_size, lambda_risk) ** 2
        / ac_kappa(market, order_size, lambda_risk) ** 2
    )
    assert ratio == pytest.approx(BPS / (order_size * market.dt), rel=1e-6)


def test_the_vendored_schedule_is_optimal_at_a_rescaled_lambda(market_case):
    """Rescale lambda and the vendored schedule *is* the exact optimum.

    Matching is done on the objective curvature rather than through the
    continuum approximation, so the recovery is exact on the discrete grid.
    """
    market, order_size = market_case
    vendored_lambda = 1e-7
    vendored = ac_trajectory(market, order_size, vendored_lambda)

    curvature = 2.0 * (
        math.cosh(ac_kappa(market, order_size, vendored_lambda) * market.dt) - 1.0
    )
    rescaled_lambda = curvature / objective_curvature(market, order_size, 1.0)

    recovered = optimal_trajectory(market, order_size, rescaled_lambda)
    worst = float(np.max(np.abs(vendored - recovered))) / order_size
    assert worst <= 1e-9, f"rescaling did not recover the vendored schedule ({worst:.3e})"


# ---------------------------------------------------------------------------
# Limits and monotonicity
# ---------------------------------------------------------------------------


def test_zero_risk_aversion_is_twap(market_case):
    """lambda = 0 removes the inventory penalty, so the optimum is flat."""
    market, order_size = market_case
    flat = twap_trajectory(market, order_size)
    optimum = optimal_trajectory(market, order_size, 0.0)
    assert np.max(np.abs(optimum - flat)) <= 1e-9 * order_size


def test_large_risk_aversion_front_loads(market_case):
    """As lambda grows the first bin takes essentially the whole order."""
    market, order_size = market_case
    executed = trades(optimal_trajectory(market, order_size, 1e6), market)
    assert executed[0] / order_size > 0.90


@pytest.mark.parametrize(
    "frontier_fn", [ac_frontier, optimal_frontier], ids=["ac", "optimal"]
)
def test_frontier_is_monotone(market_case, frontier_fn):
    """Higher risk aversion buys lower variance with higher cost, never both."""
    market, order_size = market_case
    points = frontier_fn(market, order_size, [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    for previous, current in zip(points[:-1], points[1:]):
        assert current.expected_cost >= previous.expected_cost - 1e-12
        assert current.variance <= previous.variance + 1e-12


def test_twap_minimises_temporary_cost(market_case):
    """Convex impact plus Jensen: the flat schedule is the cheapest on impact."""
    market, order_size = market_case
    flat = cost_moments(twap_trajectory(market, order_size), market).temporary
    for lam in (1e-6, 1e-4, 1e-2):
        assert cost_moments(ac_trajectory(market, order_size, lam), market).temporary >= flat


def test_variance_falls_as_the_schedule_front_loads(market_case):
    """Selling earlier means less inventory exposed to price drift."""
    market, order_size = market_case
    variances = [
        cost_moments(optimal_trajectory(market, order_size, lam), market).variance
        for lam in (1e-8, 1e-6, 1e-4, 1e-2)
    ]
    assert variances == sorted(variances, reverse=True)


# ---------------------------------------------------------------------------
# Numerical guards
# ---------------------------------------------------------------------------


def test_sinh_branch_is_continuous_at_the_overflow_threshold(market_case):
    """The exp asymptote must join the sinh form, not step away from it."""
    market, order_size = market_case
    kappa = SINH_OVERFLOW_KT / market.horizon_hours
    below = sinh_trajectory(market, order_size, kappa * (1.0 - 1e-12))
    above = sinh_trajectory(market, order_size, kappa * (1.0 + 1e-12))
    assert np.max(np.abs(below - above)) <= 1e-12 * order_size


def test_tiny_kappa_does_not_cancel_to_zero(market_case):
    """arccosh(1 + mu/2) is evaluated stably as mu -> 0."""
    market, order_size = market_case
    tiny = optimal_kappa(market, order_size, 1e-30)
    assert tiny > 0.0, "kappa cancelled to zero; the log1p form was lost"
    assert math.isfinite(tiny)


def test_sigma_bin_scales_with_the_square_root_of_the_bin(market_case):
    """sigma_bin = sigma_daily * sqrt(dt / trading day)."""
    market, _ = market_case
    assert market.sigma_bin == pytest.approx(
        market.params.sigma * math.sqrt(market.dt / TRADING_HOURS_PER_DAY), rel=1e-15
    )


@pytest.mark.parametrize("bad", [-1.0, 0.0])
def test_non_positive_order_sizes_are_rejected(market_case, bad):
    """Silently returning an empty schedule for a zero order is not acceptable."""
    market, _ = market_case
    with pytest.raises(ValueError, match="order_size must be positive"):
        twap_trajectory(market, bad)
    with pytest.raises(ValueError, match="order_size must be positive"):
        linearised_eta(market, bad)
