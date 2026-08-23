"""The power-law world's own optimum — solved, not written down.

FrontierView charges temporary impact as ``eta * sigma * p ** 0.6``. That admits
no sinh closed form, which is why the vendored library linearises at the tangent
``eta_tilde`` before it solves anything, and why every Phase-1 milestone graded
against :func:`~temper.oracle.schedules.optimal_trajectory` — the exact minimiser
of the *tangent's* objective. From M4a the power law is the world, so the
reference answer has to be the minimiser of the power law's own objective, and
this module is where it is computed.

Why it can be certified rather than trusted
-------------------------------------------
On the reachable set — sell-only and fully liquidating, which
:class:`~temper.env.ExecutionEnv`'s clip to ``[0, remaining]`` makes the *only*
set — the objective in the trade weights ``w_i = n_i / X`` is

.. code::

    J(w) = A * sum_i w_i**(1 + beta)
         + lambda * B * sum_{k<N} (1 - sum_{i<k} w_i)**2      + const
    A = eta * sigma * BPS * (X / (dt * v_hourly))**beta
    B = (sigma_bin * BPS)**2

``const`` is permanent cost plus the half-spread: both are schedule-invariant for
a monotone full liquidation (:func:`~temper.oracle.cost.permanent_cost_bps`), so
they shift ``J`` without moving its minimiser — the same constants M1's
variational certificate drops. ``w**1.6`` is strictly convex on ``w >= 0`` and
the inventory term is a convex quadratic, so ``J`` is strictly convex on the
simplex and its stationary point is the **unique global minimum**. There is no
sinh and there does not need to be one.

Two solvers, deliberately
-------------------------
:func:`optimum_for_charge` is a damped Newton iteration on the stationarity
conditions in the interior holdings — the same "generic linear algebra, no
formula" discipline M1 task 0 used, in dense numpy, reaching a relative KKT
residual of 1e-15 or better in a handful of iterations. :func:`optimum_by_shooting`
solves the *equal-marginal-cost* condition by bisecting on the first bin's
weight and touches no matrix at all. Slow and inelegant on purpose: it is the
differential check on the Newton solve, exactly as
:func:`~temper.oracle.schedules.optimal_trajectory_by_solve` is on the closed
form. ``tests/test_power_law_certificate.py`` requires them to agree to 1e-10 of
``X``.

No scipy. The optimum is dense numpy and a scalar bisection; a dependency for
one solve would be a dependency on every host this repo has to run on.

The exponent is a parameter
---------------------------
:class:`TemporaryCharge` carries ``(scale, exponent, encoding)`` and every
routine here takes one, so the *same* machinery solves the tangent world at
``exponent = 1`` — where it must return :func:`optimal_trajectory` to float
precision. That is part (e) of the certificate, and it is what makes "the two
worlds are the same problem at different exponents" a test rather than a remark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .impact import linearised_eta
from .model import BPS, LINEAR_ENCODING, POWER_LAW_ENCODING, Market
from .schedules import twap_trajectory

#: The certificate's pre-stated bar on the relative KKT residual, and the
#: solver's own contract: it iterates to the fixed point and refuses to return a
#: schedule outside this. Measured on the vendored grid the power-law solve lands
#: between 1.0e-15 and 6.9e-18 — three orders inside the bar at its worst — in
#: well under twenty iterations.
KKT_TOLERANCE = 1e-12

#: Iteration caps. Both are runaway guards rather than working limits — the
#: Newton solve converges in single digits and the bisection in ~60 halvings of
#: a bracket that starts one unit wide.
MAX_NEWTON_STEPS = 100
MAX_BISECTION_STEPS = 200

#: An iterate is accepted only while every bin still trades a strictly positive
#: amount: ``w ** (beta - 1)`` is undefined at zero and negative weights are
#: outside the reachable set anyway. Strictly positive and nothing more — an
#: absolute floor here (1e-15 was the first attempt) rejects the *true* optimum
#: at the top of the lambda grid, where the last bins legitimately trade 1e-14 of
#: the order, and the solve then damps its way to a schedule three orders less
#: stationary than the one it refused.
WEIGHT_FLOOR = 0.0


@dataclass(frozen=True)
class TemporaryCharge:
    """Temporary impact as it enters the objective, in the trade weights.

    ``temporary cost (bps) = scale * sum_i w_i ** (1 + exponent)``.

    Two constructors build the two worlds, and nothing else in the oracle needs
    to know which one it is holding: :func:`power_law_charge` is FrontierView's
    0.6-power law, :func:`tangent_charge` the linear model the Almgren–Chriss
    closed form is derived at. `encoding` travels with the pair so a solved
    optimum can say which world it is the optimum *of* — M4a's rule is that a
    metric grades the world that charges it, and an optimum is the reference
    half of that pairing.
    """

    scale: float
    exponent: float
    encoding: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"charge scale must be finite and positive, got {self.scale}")
        if not 0.0 < self.exponent <= 1.0:
            raise ValueError(
                f"charge exponent must be in (0, 1], got {self.exponent}"
            )

    def cost_bps(self, weights) -> float:
        """``scale * sum w ** (1 + exponent)`` — the temporary term alone, in bps."""
        w = np.asarray(weights, dtype=float)
        return float(self.scale * np.sum(w ** (1.0 + self.exponent)))

    def marginal_bps(self, weights):
        """``d(cost)/dw`` per bin: ``scale * (1 + exponent) * w ** exponent``."""
        w = np.asarray(weights, dtype=float)
        return self.scale * (1.0 + self.exponent) * w**self.exponent

    def curvature_bps(self, weights):
        """``d2(cost)/dw2`` per bin. Diverges as ``w -> 0`` for ``exponent < 1``."""
        w = np.asarray(weights, dtype=float)
        return (
            self.scale
            * (1.0 + self.exponent)
            * self.exponent
            * w ** (self.exponent - 1.0)
        )

    def weight_at_marginal(self, marginal):
        """The weight whose marginal cost is `marginal` — the inverse of the above.

        Undefined for a non-positive marginal, which is the shooting solver's
        signal that its trial first bin was too small (see
        :func:`optimum_by_shooting`).
        """
        m = np.asarray(marginal, dtype=float)
        return (m / (self.scale * (1.0 + self.exponent))) ** (1.0 / self.exponent)


def power_law_charge(market: Market, order_size: float) -> TemporaryCharge:
    """FrontierView's charge, expressed in the trade weights.

    ``sum_i h(p_i) w_i`` with ``h(p) = eta sigma p**beta`` and
    ``p_i = w_i X / (dt v_hourly)`` collapses to ``A sum_i w_i**(1+beta)`` with
    ``A = eta sigma BPS (X / (dt v_hourly))**beta``. Nothing is approximated in
    that step: it is the same arithmetic :func:`~temper.oracle.cost.cost_moments`
    performs bin by bin, rearranged so the optimiser can see it.
    """
    if order_size <= 0.0:
        raise ValueError(f"order_size must be positive, got {order_size}")
    beta = market.temp_exponent
    scale = (
        market.params.eta
        * market.params.sigma
        * BPS
        * (order_size / (market.dt * market.v_hourly)) ** beta
    )
    return TemporaryCharge(scale=scale, exponent=beta, encoding=POWER_LAW_ENCODING)


def tangent_charge(market: Market, order_size: float) -> TemporaryCharge:
    """The Phase-1 charge: the tangent to that power law, at exponent 1.

    ``eta_tilde BPS sum_i n_i**2 / (dt X)`` is ``A sum_i w_i**2`` with
    ``A = eta_tilde BPS X / dt``. Feeding this to the same solver must return
    :func:`~temper.oracle.schedules.optimal_trajectory`, which is the
    certificate's part (e) and the reason the exponent is a parameter.
    """
    if order_size <= 0.0:
        raise ValueError(f"order_size must be positive, got {order_size}")
    scale = linearised_eta(market, order_size) * BPS * order_size / market.dt
    return TemporaryCharge(scale=scale, exponent=1.0, encoding=LINEAR_ENCODING)


def charge_for(encoding: str, market: Market, order_size: float) -> TemporaryCharge:
    """The charge an encoding names. One place that maps the two worlds."""
    if encoding == POWER_LAW_ENCODING:
        return power_law_charge(market, order_size)
    if encoding == LINEAR_ENCODING:
        return tangent_charge(market, order_size)
    raise ValueError(f"unknown cost encoding {encoding!r}")


def inventory_penalty_scale(market: Market) -> float:
    """``B = (sigma_bin * BPS)**2`` — the inventory term's coefficient, bps²."""
    return float((market.sigma_bin * BPS) ** 2)


def schedule_invariant_bps(market: Market, order_size: float) -> float:
    """Permanent cost plus the half-spread — the part no schedule can move.

    The constant :func:`varying_objective_bps` drops, named rather than spelled
    out at each call site. On a monotone full liquidation
    :func:`~temper.oracle.cost.permanent_cost_bps` telescopes to
    ``gamma sigma BPS X / (2 v_hourly)`` and the spread is ``half_spread`` on
    weights that sum to one, so both are the same number for every schedule the
    env can realise — which is exactly why the optimiser may ignore them and why
    anything comparing a *varying* objective with a
    :class:`~temper.oracle.cost.CostMoments` one has to add them back.
    """
    if order_size <= 0.0:
        raise ValueError(f"order_size must be positive, got {order_size}")
    permanent = (
        market.params.gamma * market.params.sigma * BPS * order_size
        / (2.0 * market.v_hourly)
    )
    return float(permanent + market.params.half_spread)


# ---------------------------------------------------------------------------
# The objective, its gradient and its Hessian, in the interior holdings
# ---------------------------------------------------------------------------


def _weights(trajectory: np.ndarray, order_size: float) -> np.ndarray:
    return -np.diff(trajectory) / order_size


def varying_objective_bps(
    trajectory, market: Market, order_size: float, lambda_risk: float, charge: TemporaryCharge
) -> float:
    """The schedule-*varying* part of ``E + lambda V``, in bps.

    Permanent cost and the half-spread are omitted: constant on the reachable
    set, so they shift this without moving the minimiser. Add
    ``gamma sigma BPS X / (2 v_hourly) + half_spread`` to recover the full
    objective, which is what :func:`~temper.oracle.cost.cost_moments` reports and
    what ``tests/test_power_law_certificate.py`` checks this against.
    """
    x = np.asarray(trajectory, dtype=float)
    holdings = x[:-1] / order_size
    return charge.cost_bps(_weights(x, order_size)) + float(
        lambda_risk * inventory_penalty_scale(market) * np.sum(holdings**2)
    )


def charge_gradient(
    trajectory, market: Market, order_size: float, lambda_risk: float, charge: TemporaryCharge
) -> np.ndarray:
    """``dJ/dx`` over the interior holdings ``x_1..x_{N-1}``, bps per share."""
    x = np.asarray(trajectory, dtype=float)
    marginal = charge.marginal_bps(_weights(x, order_size)) / order_size
    penalty = inventory_penalty_scale(market)
    return (
        marginal[1:]
        - marginal[:-1]
        + 2.0 * lambda_risk * penalty * x[1:-1] / order_size**2
    )


def charge_hessian(
    trajectory, market: Market, order_size: float, lambda_risk: float, charge: TemporaryCharge
) -> np.ndarray:
    """``d2J/dx2`` over the interior holdings, bps per share².

    Tridiagonal, because bin ``i``'s trade touches only holdings ``i`` and
    ``i+1``, and positive definite wherever every bin trades: it is a weighted
    graph Laplacian of the path graph — with the endpoint weights ``curv_0`` and
    ``curv_{N-1}`` grounded by the fixed ``x_0`` and ``x_N`` — plus a positive
    multiple of the identity.

    Unlike Phase 1's, this matrix is **not constant in x**: ``w**(beta-1)``
    depends on the schedule. Every statement derived from it — the curvature
    floor, the trajectory band — is therefore a statement *at the point it was
    assembled*, and the milestone says so rather than inheriting Phase 1's
    global bound by habit.
    """
    x = np.asarray(trajectory, dtype=float)
    curvature = charge.curvature_bps(_weights(x, order_size)) / order_size**2
    size = x.size - 2
    if size < 1:
        return np.zeros((0, 0))

    hessian = np.zeros((size, size))
    index = np.arange(size)
    hessian[index, index] = curvature[:-1] + curvature[1:]
    off = np.arange(size - 1)
    hessian[off, off + 1] = -curvature[1:-1]
    hessian[off + 1, off] = -curvature[1:-1]
    hessian[index, index] += (
        2.0 * lambda_risk * inventory_penalty_scale(market) / order_size**2
    )
    return hessian


def _marginal_terms(
    trajectory: np.ndarray,
    market: Market,
    order_size: float,
    lambda_risk: float,
    charge: TemporaryCharge,
) -> tuple[np.ndarray, np.ndarray]:
    """The two halves of ``dJ/dw_i``: what trading costs, and what waiting saves."""
    holdings = trajectory[:-1] / order_size
    # sum_{k = i+1}^{N-1} of the holding fractions, per bin i.
    tail = np.concatenate((np.cumsum(holdings[:0:-1])[::-1], [0.0]))
    impact = charge.marginal_bps(_weights(trajectory, order_size))
    risk = 2.0 * lambda_risk * inventory_penalty_scale(market) * tail
    return impact, risk


def marginal_costs(
    trajectory, market: Market, order_size: float, lambda_risk: float, charge: TemporaryCharge
) -> np.ndarray:
    """``dJ/dw_i`` for each bin — equal across bins at an interior optimum.

    ``A (1+beta) w_i**beta - 2 lambda B sum_{k>i} x_k / X``. This is the KKT
    condition the certificate reads: on the simplex with no active bound, the
    stationarity condition *is* "every bin's marginal cost is the same number".
    Selling a share now pays the impact of a bigger bin; holding it pays one more
    bin of variance on everything still outstanding, and the optimum is where
    those two prices agree everywhere.
    """
    impact, risk = _marginal_terms(
        np.asarray(trajectory, dtype=float), market, order_size, lambda_risk, charge
    )
    return impact - risk


def kkt_residual(
    trajectory, market: Market, order_size: float, lambda_risk: float, charge: TemporaryCharge
) -> float:
    """Spread of the per-bin marginal costs, relative to the terms that form them.

    Zero exactly when every bin's marginal cost agrees, which on the interior of
    the simplex is stationarity. Dimensionless, so the certificate's ``<= 1e-12``
    means the same thing at every lambda and in either world.

    **The denominator is the size of the two terms, not of their difference.**
    A bin's marginal is ``impact - risk``, and at high lambda those two are large
    and nearly equal: at ``lambda = 10^-2.5`` in the tangent world they cancel to
    about four digits, so a spread that is 1e-16 of the *terms* is 1e-11 of the
    *mean* and a mean-relative residual would report a float-exact solve as a
    failure. Normalising by ``max_i(|impact_i| + |risk_i|)`` asks the question
    the arithmetic can answer — how stationary is this, relative to the numbers
    being subtracted — and floors at machine precision at every lambda instead of
    at the local cancellation.
    """
    x = np.asarray(trajectory, dtype=float)
    impact, risk = _marginal_terms(x, market, order_size, lambda_risk, charge)
    scale = float(np.max(np.abs(impact) + np.abs(risk)))
    spread = float(np.ptp(impact - risk))
    return spread / scale if scale > 0.0 else spread


# ---------------------------------------------------------------------------
# Solver 1 — damped Newton on the stationarity conditions
# ---------------------------------------------------------------------------


def optimum_for_charge(
    market: Market, order_size: float, lambda_risk: float, charge: TemporaryCharge
) -> np.ndarray:
    """The unique minimiser of `charge`'s objective, by damped Newton from TWAP.

    Newton on a strictly convex objective, with the step halved until it both
    keeps every bin trading and does not raise the objective beyond float noise.
    The damping is not decoration: the Hessian's ``w**(beta-1)`` blows up as a
    bin empties, so an undamped step from TWAP at large lambda can overshoot into
    an infeasible schedule that the next iterate cannot evaluate at all.

    The descent test carries a slack of a few ulps of the objective, and the loop
    also stops once an accepted step no longer moves the schedule. Both exist for
    the same reason: the last Newton step lands within rounding of the minimum,
    where ``J`` is flat to the last bit and a strict ``value < best`` is a test
    the arithmetic cannot pass. Without the slack the line search halves eighty
    times and the solve reports failure at an iterate whose KKT residual is 1e-15
    — a converged answer thrown away for being *too* converged.

    Returns the full ``n_bins + 1`` inventory trajectory, ``x_0 = X``,
    ``x_N = 0``.
    """
    if order_size <= 0.0:
        raise ValueError(f"order_size must be positive, got {order_size}")
    if lambda_risk < 0.0:
        raise ValueError(f"lambda_risk must be non-negative, got {lambda_risk}")
    if market.n_bins < 2:
        return np.array([order_size, 0.0])

    x = twap_trajectory(market, order_size)
    best = varying_objective_bps(x, market, order_size, lambda_risk, charge)
    answer = x
    residual = kkt_residual(x, market, order_size, lambda_risk, charge)
    previous = None

    for _ in range(MAX_NEWTON_STEPS):
        gradient = charge_gradient(x, market, order_size, lambda_risk, charge)
        hessian = charge_hessian(x, market, order_size, lambda_risk, charge)
        step = np.linalg.solve(hessian, -gradient)
        slack = 8.0 * np.finfo(float).eps * abs(best)

        length = 1.0
        for _ in range(80):
            trial = x.copy()
            trial[1:-1] += length * step
            if np.all(_weights(trial, order_size) > WEIGHT_FLOOR):
                value = varying_objective_bps(
                    trial, market, order_size, lambda_risk, charge
                )
                if value <= best + slack:
                    break
            length *= 0.5
        else:
            raise RuntimeError(
                f"the line search failed at lambda = {lambda_risk:.6e}; no damped "
                "Newton step kept every bin trading and lowered the objective"
            )

        # Run to the fixed point rather than to a threshold: what this returns is
        # then the most stationary schedule the arithmetic admits, and
        # `KKT_TOLERANCE` is a bar it is *checked* against rather than a precision
        # it settles for. Stopping at the threshold would hand the certificate a
        # 1e-12 answer where a 1e-15 one was two more 13x13 solves away.
        #
        # "Fixed point" means the iterate stops moving, or starts alternating
        # between two adjacent floats — both happen at the float floor, and the
        # residual is *not* monotone before it, so the loop cannot stop at the
        # first step that fails to improve. From TWAP at lambda = 1e-3 the very
        # first damped step raises the residual before Newton's quadratic phase
        # takes it to 1e-15, and an "improved or stop" rule returned that first
        # step: a schedule three orders from stationary, reported as the optimum.
        settled = np.array_equal(trial, x) or (
            previous is not None and np.array_equal(trial, previous)
        )
        previous, x, best = x, trial, min(value, best)
        current = kkt_residual(x, market, order_size, lambda_risk, charge)
        if current < residual:
            answer, residual = x, current
        if settled:
            break

    if residual > KKT_TOLERANCE:
        raise RuntimeError(
            f"the Newton solve reached a relative KKT residual of {residual:.3e}, "
            f"outside the certified {KKT_TOLERANCE:g}, at lambda = "
            f"{lambda_risk:.6e}"
        )
    return answer


def power_law_optimum(
    market: Market, order_size: float, lambda_risk: float
) -> np.ndarray:
    """The certified optimum of FrontierView's power-law world — M4a's reference.

    What an M4a agent is graded against. The Almgren–Chriss schedule is derived
    at the tangent to this world's impact function and therefore does *not* solve
    it: at the reference case the two schedules sit 16 878 shares apart, which is
    the mis-specification M4a's advantage is measured over.
    """
    return optimum_for_charge(
        market, order_size, lambda_risk, power_law_charge(market, order_size)
    )


# ---------------------------------------------------------------------------
# Solver 2 — bisection on the equal-marginal-cost condition
# ---------------------------------------------------------------------------


def _shoot(
    first_weight: float,
    market: Market,
    order_size: float,
    lambda_risk: float,
    charge: TemporaryCharge,
) -> tuple[np.ndarray, float]:
    """Propagate the stationarity recursion from a trial first-bin weight.

    Differencing the equal-marginal condition between neighbouring bins gives
    ``m_j = m_{j-1} - 2 lambda B x_j / X`` on the per-weight marginals, so the
    whole schedule is determined by ``w_0`` alone: take the marginal, walk it
    down bin by bin, invert it for the next weight. The residual is the
    inventory left at the horizon, which the true optimum drives to zero.

    Monotone by construction — a larger ``w_0`` leaves less inventory, which
    lowers every later marginal decrement and so raises every later weight — and
    that is what makes bisection sound rather than merely convergent.
    """
    penalty = 2.0 * lambda_risk * inventory_penalty_scale(market)
    x = np.empty(market.n_bins + 1)
    x[0] = order_size
    marginal = float(charge.marginal_bps(np.array([first_weight]))[0])

    for index in range(market.n_bins):
        if index > 0:
            marginal -= penalty * x[index] / order_size
            if marginal <= 0.0:
                # No positive weight has this marginal: w_0 was too small.
                return x, math.inf
            weight = float(charge.weight_at_marginal(np.array([marginal]))[0])
        else:
            weight = first_weight
        x[index + 1] = x[index] - weight * order_size
        if x[index + 1] <= 0.0:
            # Already over-liquidated, so the residual's sign is settled and the
            # bisection has what it needs. Continuing would keep raising the
            # marginal off negative inventory until ``w = m ** (1/beta)``
            # overflows — a float warning from a branch whose answer is known.
            x[index + 2 :] = x[index + 1]
            return x, float(x[index + 1])

    return x, float(x[-1])


def optimum_by_shooting(
    market: Market, order_size: float, lambda_risk: float, charge: TemporaryCharge
) -> np.ndarray:
    """The same optimum, by bisecting the equal-marginal condition. No matrices.

    Deliberately unlike :func:`optimum_for_charge`: no Hessian, no linear solve,
    no line search — a scalar bisection on the first bin's weight against a
    residual that is monotone in it. Independent enough that agreeing to 1e-10 of
    ``X`` is evidence about the answer rather than about a shared bug, which is
    part (d) of the certificate and the same role
    :func:`~temper.oracle.schedules.optimal_trajectory_by_solve` plays for the
    closed form.
    """
    if market.n_bins < 2:
        return np.array([order_size, 0.0])

    low, high = 0.0, 1.0
    for _ in range(MAX_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:  # the bracket is float-adjacent
            break
        _, residual = _shoot(middle, market, order_size, lambda_risk, charge)
        if residual > 0.0:
            low = middle
        else:
            high = middle

    # The bracket straddles the root; return whichever end lands closer to a
    # clean liquidation rather than whichever half the loop happened to exit on.
    candidates = [
        _shoot(end, market, order_size, lambda_risk, charge)
        for end in (low, high)
        if end > 0.0
    ]
    trajectory, _ = min(candidates, key=lambda pair: abs(pair[1]))
    return trajectory


def power_law_optimum_by_shooting(
    market: Market, order_size: float, lambda_risk: float
) -> np.ndarray:
    """:func:`optimum_by_shooting` on FrontierView's power law."""
    return optimum_by_shooting(
        market, order_size, lambda_risk, power_law_charge(market, order_size)
    )


# ---------------------------------------------------------------------------
# The local curvature floor, and the band it implies
# ---------------------------------------------------------------------------


def local_curvature_floor(
    trajectory, market: Market, order_size: float, lambda_risk: float, charge: TemporaryCharge
) -> float:
    """``lambda_min`` of the Hessian **at this schedule** — a local statement.

    Phase 1's :func:`~temper.oracle.schedules.objective_curvature_floor` is a
    property of the whole problem, because there the Hessian is constant. Here it
    is not, so this is the curvature of the bowl at the point it was measured,
    and the band derived from it holds in a neighbourhood rather than globally.
    M4a validates that band by direct evaluation at its own radius instead of
    asserting the quadratic inequality — which is the honest version of the same
    check, and the only one that survives the Hessian moving.
    """
    hessian = charge_hessian(
        trajectory, market, order_size, lambda_risk, charge
    )
    if hessian.size == 0:
        raise ValueError(f"an interior holding needs n_bins >= 2, got {market.n_bins}")
    return float(np.linalg.eigvalsh(hessian)[0])
