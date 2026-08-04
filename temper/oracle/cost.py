"""Expected cost, its decomposition, and shortfall variance for a schedule.

A schedule is represented by its **inventory trajectory**: ``n_bins + 1`` share
counts, inventory remaining at each bin boundary, starting at the parent order
size. Trades are its negated first difference. Everything else — participation
rates, execution weights — is derived, so there is exactly one representation of
a schedule in the oracle and no way for two of them to disagree.

Cost model (bps of notional), with ``w_i = n_i / X`` the execution weight of
bin ``i``:

* **temporary** ``sum_i h(p_i) * w_i`` — power law, or linear under
  :func:`linear_cost_moments`.
* **permanent**  midpoint accumulation: each bin pays the drift left by every
  earlier bin plus half its own. For linear ``g`` this telescopes to a closed
  form independent of schedule shape (see :func:`permanent_cost_bps`).
* **spread**     ``half_spread * sum_i w_i`` — the half-spread on every share.
* **variance**   ``sigma_bin^2 * sum_{i<N} (x_i / X)^2`` in bps², the variance
  of execution shortfall while the order is worked. Inter-bin correlation is
  ignored, as in the vendored model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .impact import linearised_eta, temporary_impact_bps
from .model import BPS, Market


@dataclass(frozen=True)
class CostMoments:
    """Cost decomposition and shortfall variance for one schedule, in bps."""

    temporary: float
    permanent: float
    spread: float
    variance: float

    @property
    def expected(self) -> float:
        """Total expected cost in bps: the three components summed."""
        return self.temporary + self.permanent + self.spread

    def objective(self, lambda_risk: float) -> float:
        """The frozen mean–variance objective ``E + lambda * V``.

        Constitution invariant 7: the training reward, the evaluation metric and
        the oracle all optimise *this* functional and nothing else.
        """
        return self.expected + lambda_risk * self.variance


def _decompose(trajectory, market: Market):
    """Split a trajectory into (inventory, trades, X, weights, participation)."""
    x = np.asarray(trajectory, dtype=float)
    if x.ndim != 1 or x.size != market.n_bins + 1:
        raise ValueError(
            f"trajectory must have {market.n_bins + 1} points for an "
            f"{market.n_bins}-bin grid, got shape {x.shape}"
        )
    order_size = x[0]
    if order_size <= 0.0:
        raise ValueError(f"trajectory must start at a positive size, got {order_size}")
    executed = x[:-1] - x[1:]
    weights = executed / order_size
    rates = executed / (market.dt * market.v_hourly)
    return x, executed, order_size, weights, rates


def trades(trajectory, market: Market) -> np.ndarray:
    """Shares executed in each bin: the negated first difference of inventory."""
    return _decompose(trajectory, market)[1]


def participation(trajectory, market: Market) -> np.ndarray:
    """Per-bin participation rate: shares-per-hour over `v_hourly`."""
    return _decompose(trajectory, market)[4]


def permanent_cost_bps(trajectory, market: Market) -> float:
    """Midpoint-rule permanent cost in bps — closed form where one exists.

    Each bin's own drift is ``a_i = gamma * sigma * |p_i| * BPS * dt``, which for
    a sell schedule is ``c * n_i`` with ``c = gamma * sigma * BPS / v_hourly``.
    The midpoint accumulation is then

    .. code::

        sum_i (c * sum_{j<i} n_j + c * n_i / 2) * n_i / X
          = (c / X) * [ sum_i n_i sum_{j<i} n_j + (1/2) sum_i n_i^2 ]
          = c * (sum_i n_i)^2 / (2 X)

    — the telescoping identity behind FrontierView's schedule-invariance claim,
    here generalised to a schedule that does not fully liquidate (the large-κ
    asymptotic branch leaves an exponentially small residue). Because permanent
    impact is charged on ``|p_i|``, the telescoping needs monotone inventory; a
    non-monotone path falls back to the explicit accumulation.
    """
    _, executed, order_size, weights, _ = _decompose(trajectory, market)
    c = market.params.gamma * market.params.sigma * BPS / market.v_hourly

    if np.all(executed >= 0.0):
        return float(c * executed.sum() ** 2 / (2.0 * order_size))

    own = c * np.abs(executed)
    prior = np.concatenate(([0.0], np.cumsum(own)[:-1]))
    return float(np.sum((prior + own / 2.0) * weights))


def shortfall_variance_bps2(trajectory, market: Market) -> float:
    """Variance of execution shortfall in bps², over inventory *before* each bin."""
    x, _, order_size, _, _ = _decompose(trajectory, market)
    return float((market.sigma_bin * BPS) ** 2 * np.sum((x[:-1] / order_size) ** 2))


def cost_moments(trajectory, market: Market) -> CostMoments:
    """Cost moments under the power-law temporary impact model.

    This is the vendored model: what FrontierView charges, and what the goldens
    pin to 1e-6 relative.
    """
    _, _, _, weights, participation = _decompose(trajectory, market)
    temporary = float(np.sum(temporary_impact_bps(participation, market) * weights))
    return CostMoments(
        temporary=temporary,
        permanent=permanent_cost_bps(trajectory, market),
        spread=float(market.params.half_spread * weights.sum()),
        variance=shortfall_variance_bps2(trajectory, market),
    )


def linear_cost_moments(
    trajectory, market: Market, eta_tilde: float
) -> CostMoments:
    """Cost moments under *linear* temporary impact — the closed form's model.

    Replaces the power law with its tangent ``eta_tilde * v``, giving

    .. code::

        temporary = eta_tilde * BPS / (dt * X) * sum_i n_i^2

    Permanent, spread and variance are unchanged: only the temporary term
    differs between the two regimes. This is the functional whose exact discrete
    minimiser is the sinh trajectory
    (:func:`~temper.oracle.schedules.optimal_trajectory`).
    """
    _, executed, order_size, weights, _ = _decompose(trajectory, market)
    temporary = float(
        eta_tilde * BPS * np.sum(executed**2) / (market.dt * order_size)
    )
    return CostMoments(
        temporary=temporary,
        permanent=permanent_cost_bps(trajectory, market),
        spread=float(market.params.half_spread * weights.sum()),
        variance=shortfall_variance_bps2(trajectory, market),
    )


def schedule_moments(
    trajectory, market: Market, *, order_size: float | None = None
) -> CostMoments:
    """Phase-1 moments of an **arbitrary** deterministic schedule.

    The named schedules (TWAP, ``ac_*``, ``optimal_*``) have their moments pinned
    by the M0 goldens. This is the same computation for a schedule nobody wrote
    down in advance — a realised trajectory out of :class:`ExecutionEnv
    <temper.env.ExecutionEnv>`, say, possibly one whose tail was force-liquidated
    — which is what M1's Monte-Carlo differential standardises against.

    Phase 1 is the linearised world end-to-end (``ARCHITECTURE.md`` §9,
    2026-08-04), so this is :func:`linear_cost_moments` at the tangent slope
    :func:`~temper.oracle.impact.linearised_eta`. The tangent is taken at the
    *parent* order's TWAP participation, not the realised schedule's, because
    ``eta_tilde`` is a property of the order the env was configured with and is
    frozen for the episode; pass `order_size` explicitly when the trajectory does
    not start at the parent size.
    """
    x = np.asarray(trajectory, dtype=float)
    reference = float(x[0]) if order_size is None else order_size
    return linear_cost_moments(x, market, linearised_eta(market, reference))
