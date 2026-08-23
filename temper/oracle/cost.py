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


def cost_moments(trajectory, market: Market, *, liquidity=None) -> CostMoments:
    """Cost moments under the power-law temporary impact model.

    This is the vendored model: what FrontierView charges, and what the goldens
    pin to 1e-6 relative.

    `liquidity` is M4b's per-bin multiplier on ``v_hourly``: participation becomes
    ``p_k = n_k / (dt v_hourly L_k)`` and **nothing else in the model changes**.
    Passing a realised path makes this ``E[cost | L]`` *exactly* rather than
    approximately — the price shock enters realised cost only through M1a's affine
    term and a graded policy never sees a price, so conditioning on the liquidity
    path removes all of the price randomness analytically and what is left is a
    closed form. That is the route by which an M4b agent is graded, and the
    per-step assertion that licenses it is
    :func:`~temper.eval.grading.deterministic_schedule`.

    ``liquidity=None`` is the deterministic world and is **bit-identical** to what
    this function computed before the argument existed, which
    ``tests/test_m4b_conditional_grading.py`` pins: no M4a or earlier number is
    permitted to move because a later milestone widened a signature.

    The encoding is unchanged. Liquidity randomises the *market*, not the cost
    functional — the charge is still ``eta sigma p**beta`` — so §9's *A metric
    grades the world that charges it* is untouched and this stays a ``power_law``
    metric.
    """
    _, _, _, weights, participation = _decompose(trajectory, market)
    if liquidity is not None:
        multiplier = np.asarray(liquidity, dtype=float)
        if multiplier.shape not in {(), (market.n_bins,)}:
            raise ValueError(
                f"liquidity must be a scalar or {market.n_bins} per-bin "
                f"multipliers, got shape {multiplier.shape}"
            )
        if np.any(multiplier <= 0.0):
            raise ValueError("liquidity multipliers must be strictly positive")
        participation = participation / multiplier
    temporary = float(np.sum(temporary_impact_bps(participation, market) * weights))
    return CostMoments(
        temporary=temporary,
        permanent=permanent_cost_bps(trajectory, market),
        spread=float(market.params.half_spread * weights.sum()),
        variance=shortfall_variance_bps2(trajectory, market),
    )


def expected_cost_moments(trajectory, market: Market, law) -> CostMoments:
    """A **fixed** schedule's ``E[cost]`` under a liquidity law — a closed form.

    A fixed schedule's weights are not random, so the expectation passes straight
    through the multiplier and lands on
    :meth:`~temper.oracle.liquidity.LiquidityLaw.inverse_power_moment`:

    .. code::

        E[ sum_k h(n_k / (dt v L_k)) w_k ] = E[L^-beta] * sum_k h(n_k / (dt v)) w_k

    Permanent impact, the half-spread and the shortfall variance are untouched —
    ``V`` is *price*-shortfall variance and liquidity dispersion enters ``E[cost]``
    through Jensen, not ``lambda V``, which is why M4b needs no amendment to
    invariant 7.

    Closed form rather than an average over sampled paths, and that is the whole
    point: the two static rungs this prices differ by ~0.002 bps, and differencing
    two *simulated* levels turns that into noise. The level shift is the gate that
    decides whether M4b's headline is adaptivity or a re-solve, so it is computed
    where it can be computed exactly.
    """
    base = cost_moments(trajectory, market)
    return CostMoments(
        temporary=base.temporary * law.inverse_power_moment(market.temp_exponent),
        permanent=base.permanent,
        spread=base.spread,
        variance=base.variance,
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
