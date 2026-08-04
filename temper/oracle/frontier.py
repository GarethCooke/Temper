"""The mean–variance efficient frontier: (E, V) points traced over lambda.

M3's hero figure overlays RL-traced points on this locus, so the frontier is
oracle surface, not eval-harness surface: one implementation, used by the
figure, the tolerance table and the tests alike.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .cost import cost_moments
from .model import Market
from .schedules import ac_kappa, ac_trajectory, optimal_kappa, optimal_trajectory

#: FrontierView's own lambda sweep: 1e-9 to 1e-1 on a log-half-decade grid.
VENDOR_LAMBDA_GRID: tuple[float, ...] = tuple(10 ** (i * 0.5 - 9) for i in range(17))


@dataclass(frozen=True)
class FrontierPoint:
    """One (expected cost, variance) pair and the schedule that produced it."""

    lambda_risk: float
    kappa: float
    expected_cost: float
    variance: float


def _point(market, order_size, lambda_risk, trajectory_fn, kappa_fn) -> FrontierPoint:
    moments = cost_moments(trajectory_fn(market, order_size, lambda_risk), market)
    return FrontierPoint(
        lambda_risk=lambda_risk,
        kappa=kappa_fn(market, order_size, lambda_risk),
        expected_cost=moments.expected,
        variance=moments.variance,
    )


def ac_frontier_point(
    market: Market, order_size: float, lambda_risk: float
) -> FrontierPoint:
    """(E, V) for the vendored AC schedule at `lambda_risk` — golden-pinned."""
    return _point(market, order_size, lambda_risk, ac_trajectory, ac_kappa)


def optimal_frontier_point(
    market: Market, order_size: float, lambda_risk: float
) -> FrontierPoint:
    """(E, V) for the exact discrete optimum at `lambda_risk`."""
    return _point(market, order_size, lambda_risk, optimal_trajectory, optimal_kappa)


def ac_frontier(
    market: Market,
    order_size: float,
    lambdas: Iterable[float] = VENDOR_LAMBDA_GRID,
) -> Sequence[FrontierPoint]:
    """The vendored AC frontier over a lambda grid, ascending in lambda."""
    return [ac_frontier_point(market, order_size, lam) for lam in sorted(lambdas)]


def optimal_frontier(
    market: Market,
    order_size: float,
    lambdas: Iterable[float] = VENDOR_LAMBDA_GRID,
) -> Sequence[FrontierPoint]:
    """The exact-optimum frontier over a lambda grid, ascending in lambda."""
    return [optimal_frontier_point(market, order_size, lam) for lam in sorted(lambdas)]
