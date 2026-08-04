"""Task 0 — a certificate that ``optimal_trajectory`` is the optimum.

M0 established that the oracle carries two kappas and pinned the lambda-rescaling
that maps one onto the other. That proves they are one sinh family. It does *not*
prove either is optimal: a family can be self-consistent and wrong, and from M2
onward the rediscovery claim is graded against ``optimal_*``, so "is it actually
the minimiser?" stops being a rhetorical question.

This module answers it without using a kappa anywhere. It reassembles the frozen
objective from the units in ``oracle/model.py`` as a quadratic in the interior
holdings, and then does what one does with a quadratic:

* **positive-definite** — Cholesky succeeds, so the stationary point is the unique
  global minimum rather than a saddle or a maximum;
* **stationarity** — a generic dense solve of ``H y = -b`` reproduces
  ``optimal_trajectory``;
* **perturbation** — 200 random interior directions at two scales all move the
  objective up;
* **monotonicity** — ``x*`` is a sell-only schedule, which is what makes the
  spread term a constant that drops out of the gradient.

Neither ``optimal_kappa`` nor ``objective_curvature`` is imported. If someone
rewrites the closed form, this test still knows what the right answer is.

The quadratic
-------------
With ``x_0 = X`` and ``x_N = 0`` fixed, the frozen objective (bps) is

.. code::

    U(x) = A * sum_{i<N} (x_i - x_{i+1})^2  +  B * sum_{i<N} x_i^2  +  const
    A    = eta_tilde * BPS / (X * dt)          temporary impact, per share^2
    B    = lambda * (sigma_bin * BPS)^2 / X^2  the inventory penalty

``const`` is permanent cost plus the half-spread: both are ``schedule``-invariant
for a monotone schedule that fully liquidates, so they shift ``U`` without moving
its minimiser. The Hessian in the interior variables is
``2A * tridiag(-1, 2, -1) + 2B * I`` — the second difference in ``eta_tilde/tau``
plus ``lambda * sigma^2 * tau`` on the diagonal.
"""

from __future__ import annotations

import numpy as np
import pytest

from temper.oracle import BPS, linearised_eta, optimal_trajectory, schedule_moments, trades

from .conftest import M1_CONFIG, case_by_id

VARIATIONAL = M1_CONFIG["variational"]
CASES = [case_by_id(case_id) for case_id in VARIATIONAL["cases"]]
SOLVE_RTOL = float(VARIATIONAL["solve_rtol"])
TOLERANCE_REL = float(VARIATIONAL["tolerance_rel"])
DIRECTIONS = int(VARIATIONAL["directions"])
SCALES = [float(scale) for scale in VARIATIONAL["scales"]]


def _coefficients(case) -> tuple[float, float]:
    """``(A, B)`` in the units of ``oracle/model.py``, from the parameters only."""
    market, order_size = case.market, case.order_size
    a = linearised_eta(market, order_size) * BPS / (order_size * market.dt)
    b = case.lambda_risk * (market.sigma_bin * BPS) ** 2 / order_size**2
    return a, b


def _objective(x: np.ndarray, case) -> float:
    """``U(x)`` — the quadratic part of the frozen objective, in bps."""
    a, b = _coefficients(case)
    return float(a * np.sum(np.diff(x) ** 2) + b * np.sum(x[:-1] ** 2))


def _hessian_and_gradient_offset(case) -> tuple[np.ndarray, np.ndarray]:
    """``H`` and ``b`` with ``grad U(y) = H y + b`` over the interior holdings."""
    a, b = _coefficients(case)
    size = case.market.n_bins - 1

    second_difference = np.zeros((size, size))
    np.fill_diagonal(second_difference, 2.0)
    off = np.arange(size - 1)
    second_difference[off, off + 1] = -1.0
    second_difference[off + 1, off] = -1.0

    hessian = 2.0 * a * second_difference + 2.0 * b * np.eye(size)
    # x_0 = X enters through the (x_0 - x_1)^2 term only; x_N = 0 contributes
    # nothing, which is why the offset has a single non-zero entry.
    offset = np.zeros(size)
    offset[0] = -2.0 * a * case.order_size
    return hessian, offset


def _stationary_trajectory(case) -> np.ndarray:
    """Solve the stationarity system generically: no kappa, no sinh, no oracle."""
    hessian, offset = _hessian_and_gradient_offset(case)
    interior = np.linalg.solve(hessian, -offset)
    return np.concatenate(([case.order_size], interior, [0.0]))


@pytest.fixture(params=CASES, ids=str)
def case(request):
    """The 3 x 3 golden grid the config names for the certificate."""
    return request.param


# ---------------------------------------------------------------------------
# The quadratic really is the oracle's objective
# ---------------------------------------------------------------------------


def test_the_quadratic_reproduces_the_oracles_objective(case):
    """U(x) + constants == schedule_moments(x).objective(lambda).

    Without this the rest of the module would certify a functional of its own
    invention. The constants are permanent cost and the half-spread, both fixed
    for any monotone schedule that fully liquidates.
    """
    market, order_size = case.market, case.order_size
    constant = (
        market.params.gamma * market.params.sigma * BPS * order_size
        / (2.0 * market.v_hourly)
        + market.params.half_spread
    )
    schedules = [optimal_trajectory(market, order_size, case.lambda_risk)]
    schedules.append(order_size * (1.0 - market.times / market.horizon_hours))  # TWAP
    schedules.append(np.asarray(case.ac["trajectory"], dtype=float))

    for schedule in schedules:
        assert np.all(trades(schedule, market) >= -1e-9 * order_size)
        oracle = schedule_moments(schedule, market).objective(case.lambda_risk)
        assert _objective(schedule, case) + constant == pytest.approx(oracle, rel=1e-12)


# ---------------------------------------------------------------------------
# (a) positive definite, (b) stationarity, (c) perturbation, (d) monotone
# ---------------------------------------------------------------------------


def test_the_hessian_is_positive_definite(case):
    """Cholesky succeeds: the stationary point is the unique global minimum.

    A quadratic with a positive-definite Hessian has exactly one stationary
    point and it is the minimum — no line search, no local-optimum caveat, and
    no need to trust the closed form to know it.
    """
    hessian, _ = _hessian_and_gradient_offset(case)
    factor = np.linalg.cholesky(hessian)  # raises LinAlgError if not PD
    assert np.all(np.diag(factor) > 0.0)
    assert np.allclose(factor @ factor.T, hessian, rtol=1e-12, atol=0.0)


def test_the_generic_solve_matches_the_closed_form(case):
    """The dense stationarity solve reproduces `optimal_trajectory`."""
    solved = _stationary_trajectory(case)
    closed = optimal_trajectory(case.market, case.order_size, case.lambda_risk)
    worst = float(np.max(np.abs(solved - closed))) / case.order_size
    assert worst <= SOLVE_RTOL, (
        f"stationarity solve and closed form differ by {worst:.3e} of X "
        f"(band {SOLVE_RTOL:.0e})"
    )


def test_no_perturbation_of_the_optimum_lowers_the_objective(case):
    """200 random interior directions at two scales, all uphill.

    The scales are fractions of X rather than of the local inventory: a
    front-loaded optimum has late bins of a few shares, and a perturbation scaled
    to *those* moves the objective by less than a double can represent, which
    would make this test pass on rounding noise instead of on convexity.
    """
    optimum = optimal_trajectory(case.market, case.order_size, case.lambda_risk)
    best = _objective(optimum, case)
    floor = -TOLERANCE_REL * abs(best)

    rng = np.random.default_rng(int(VARIATIONAL["seed"]))
    interior = case.market.n_bins - 1
    for scale in SCALES:
        directions = rng.normal(size=(DIRECTIONS, interior))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        directions *= scale * case.order_size

        worst = np.inf
        for direction in directions:
            perturbed = optimum.copy()
            perturbed[1:-1] += direction
            worst = min(worst, _objective(perturbed, case) - best)
        assert worst >= floor, (
            f"a perturbation at |d| = {scale:g}X lowered the objective by "
            f"{-worst:.3e} bps (floor {floor:.3e})"
        )


def test_the_optimum_is_monotone_so_the_spread_term_is_constant(case):
    """(d), stated rather than assumed.

    The spread charge is ``half_spread * sum_k |n_k| / X``. That is the constant
    ``half_spread`` only while every ``n_k >= 0``; a schedule that bought back
    would pay the spread on the round trip and the term would re-enter the
    gradient, breaking the quadratic above. The optimum is monotone, so it does
    not — and this test fails if that ever stops being true.
    """
    optimum = optimal_trajectory(case.market, case.order_size, case.lambda_risk)
    executed = trades(optimum, case.market)
    assert np.all(executed >= 0.0), "the optimum buys back; the spread term is not constant"
    assert np.all(np.diff(optimum) <= 0.0), "inventory is not monotone"
    assert executed.sum() == pytest.approx(case.order_size, rel=1e-12)

    spread = case.market.params.half_spread * np.sum(np.abs(executed)) / case.order_size
    assert spread == pytest.approx(case.market.params.half_spread, rel=1e-12)
