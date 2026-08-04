"""Inventory trajectories: TWAP, the vendored AC schedule, and the exact optimum.

Two Almgren–Chriss decay rates live in this module and they are **not the same
number**. Both are correct; they answer different questions, and conflating them
would quietly void the rediscovery claim.

``ac_kappa`` — the vendored convention
    ``kappa^2 = lambda * sigma_bin^2 / eta_tilde``, exactly as FrontierView
    computes it. This is the schedule the goldens pin, and the schedule the
    frontier in FrontierView's UI traces. It is a continuum-limit expression
    with the order size and bin length absorbed into the risk-aversion scale.

``optimal_kappa`` — the exact discrete minimiser
    The stationary point of the frozen objective
    :meth:`~temper.oracle.cost.CostMoments.objective` under linear temporary
    impact, solved on the discrete grid rather than in the continuum::

        cosh(kappa * dt) = 1 + mu / 2,
        mu = lambda * sigma_bin^2 * BPS * dt / (X * eta_tilde)

    Writing the linearised objective as ``A * sum (x_i - x_{i+1})^2 + B * sum
    x_i^2`` with ``A = eta_tilde * BPS / (X * dt)`` and ``B = lambda *
    (sigma_bin * BPS)^2 / X^2``, the first-order conditions give the three-term
    recurrence ``x_{i-1} - 2 x_i + x_{i+1} = (B / A) x_i``, whose solution under
    ``x_0 = X``, ``x_N = 0`` is the sinh trajectory at the kappa above. That is a
    genuine discrete AC solve, not a sampled continuum solution.

The two agree in shape — both are ``X * sinh(kappa (T - t)) / sinh(kappa T)`` —
but not in rate: ``optimal_kappa^2 / ac_kappa^2 -> BPS / (X * dt)`` in the
small-``kappa*dt`` limit, so the same lambda front-loads by different amounts
under the two conventions. See ``ARCHITECTURE.md`` §9 (2026-08-04) for why both
are kept: the vendored one because invariant 2 requires matching the goldens,
the exact one because "the agent converges to the optimum" is only well-posed
against a trajectory that actually is the optimum.

All functions here return trajectories: ``n_bins + 1`` inventory levels in
shares, starting at the parent order size.
"""

from __future__ import annotations

import math

import numpy as np

from .impact import linearised_eta
from .model import BPS, Market

#: ``sinh(kappa * T)`` overflows a float64 near ``kappa * T = 710``; above this
#: threshold the vendored implementation switches to the ``exp(-kappa t)``
#: asymptote, which is the correct large-``kappa*T`` limit. Pinned by the
#: `sinh-overflow-asymptote` golden.
SINH_OVERFLOW_KT = 500.0

#: Floor on ``kappa^2`` in the vendored convention, guarding ``kappa = 0``.
KAPPA2_FLOOR = 1e-12


def twap_trajectory(market: Market, order_size: float) -> np.ndarray:
    """Flat participation: inventory runs down linearly to zero."""
    if order_size <= 0.0:
        raise ValueError(f"order_size must be positive, got {order_size}")
    fraction = 1.0 - np.arange(market.n_bins + 1, dtype=float) / market.n_bins
    return order_size * fraction


def sinh_trajectory(market: Market, order_size: float, kappa: float) -> np.ndarray:
    """``X * sinh(kappa (T - t)) / sinh(kappa T)`` on the grid.

    Degenerate rates are handled explicitly: ``kappa = 0`` is the TWAP limit
    (``(T - t) / T``), and above :data:`SINH_OVERFLOW_KT` the ratio is replaced
    by its ``exp(-kappa t)`` asymptote before ``sinh(kappa T)`` overflows.
    """
    if order_size <= 0.0:
        raise ValueError(f"order_size must be positive, got {order_size}")
    if kappa < 0.0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")

    horizon, times = market.horizon_hours, market.times
    if kappa == 0.0:
        return order_size * (1.0 - times / horizon)
    if kappa * horizon > SINH_OVERFLOW_KT:
        return order_size * np.exp(-kappa * times)
    return order_size * np.sinh(kappa * (horizon - times)) / math.sinh(kappa * horizon)


# ---------------------------------------------------------------------------
# The vendored (FrontierView) convention — golden-pinned
# ---------------------------------------------------------------------------


def ac_kappa(market: Market, order_size: float, lambda_risk: float) -> float:
    """FrontierView's decay rate: ``sqrt(lambda * sigma_bin^2 / eta_tilde)``.

    Floored at ``sqrt(KAPPA2_FLOOR)``. Reproduced verbatim because invariant 2
    makes the goldens normative for this quantity.
    """
    if lambda_risk < 0.0:
        raise ValueError(f"lambda_risk must be non-negative, got {lambda_risk}")
    eta_tilde = linearised_eta(market, order_size)
    return math.sqrt(max(lambda_risk * market.sigma_bin**2 / eta_tilde, KAPPA2_FLOOR))


def ac_trajectory(market: Market, order_size: float, lambda_risk: float) -> np.ndarray:
    """The vendored Almgren–Chriss trajectory at `lambda_risk`."""
    return sinh_trajectory(market, order_size, ac_kappa(market, order_size, lambda_risk))


# ---------------------------------------------------------------------------
# The exact discrete optimum of the frozen objective
# ---------------------------------------------------------------------------


def objective_curvature(market: Market, order_size: float, lambda_risk: float) -> float:
    """``mu = B / A``, the curvature ratio of the linearised objective.

    ``mu = lambda * sigma_bin^2 * BPS * dt / (X * eta_tilde)``. It is the only
    way lambda, the grid and the order size enter the optimal schedule.
    """
    if lambda_risk < 0.0:
        raise ValueError(f"lambda_risk must be non-negative, got {lambda_risk}")
    eta_tilde = linearised_eta(market, order_size)
    return lambda_risk * market.sigma_bin**2 * BPS * market.dt / (order_size * eta_tilde)


def optimal_kappa(market: Market, order_size: float, lambda_risk: float) -> float:
    """Exact discrete AC decay rate: ``arccosh(1 + mu / 2) / dt``.

    Evaluated as ``log1p(u + sqrt(u (u + 2))) / dt`` with ``u = mu / 2``, which
    stays accurate as ``mu -> 0`` where ``arccosh(1 + mu / 2)`` would cancel to
    zero, and degrades gracefully to ``log(mu) / dt`` when ``mu`` is large enough
    that ``1 + mu / 2`` overflows.
    """
    u = objective_curvature(market, order_size, lambda_risk) / 2.0
    if not math.isfinite(u):
        raise ValueError("objective curvature overflowed; lambda_risk is too large")
    inner = u + math.sqrt(u * (u + 2.0))
    arccosh = math.log(inner) if inner > 1e15 else math.log1p(inner)
    return arccosh / market.dt


def optimal_trajectory(
    market: Market, order_size: float, lambda_risk: float
) -> np.ndarray:
    """The trajectory minimising ``E + lambda * V`` under linear temporary impact.

    This is the reference answer the RL agent is graded against from M2 onward.
    """
    return sinh_trajectory(
        market, order_size, optimal_kappa(market, order_size, lambda_risk)
    )


def optimal_trajectory_by_solve(
    market: Market, order_size: float, lambda_risk: float
) -> np.ndarray:
    """The same optimum, obtained by solving the first-order conditions directly.

    Builds the tridiagonal system that the stationarity conditions impose on the
    interior inventory levels and solves it with dense linear algebra. Slower and
    less elegant than :func:`optimal_trajectory`, and completely independent of
    it — which is the point: it is the differential check on the closed form.
    """
    n = market.n_bins
    mu = objective_curvature(market, order_size, lambda_risk)
    if n == 1:
        return np.array([order_size, 0.0])

    # (2 + mu) x_i - x_{i-1} - x_{i+1} = 0 for i = 1 .. n-1, with x_0 = X, x_n = 0.
    size = n - 1
    matrix = np.zeros((size, size))
    np.fill_diagonal(matrix, 2.0 + mu)
    idx = np.arange(size - 1)
    matrix[idx, idx + 1] = -1.0
    matrix[idx + 1, idx] = -1.0

    rhs = np.zeros(size)
    rhs[0] = order_size  # the known x_0 moved to the right-hand side

    interior = np.linalg.solve(matrix, rhs)
    return np.concatenate(([order_size], interior, [0.0]))
