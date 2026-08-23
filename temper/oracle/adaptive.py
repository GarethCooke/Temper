"""The adaptive optimum of the stochastic-liquidity world — converged, and bracketed.

M4a could **certify** its optimum: the objective was strictly convex on the
reachable set, the Hessian was Cholesky-positive-definite, and the KKT residual
came out at 1.2e-15. A stochastic dynamic program has no such certificate, and
calling this one certified would be the first dishonest number in the repo. What
is available instead is stronger than "it converged", and it is two-sided:

* **A feasible upper bound.** :meth:`AdaptiveOptimum.greedy_weights` is a *real
  policy* — it is what an agent could actually execute — so its mean conditional
  cost is an unbiased estimate of an attainable value, and therefore an upper
  bound on the optimum. It is only a *useful* bound if the stage problem is solved
  properly: snapping the action to a grid node degrades it by an order of
  magnitude, which is why :func:`_stage_minimum` interpolates and searches.
* **A rigorous lower bound.** :func:`clairvoyant_trajectories` hands the optimiser
  the whole liquidity path in advance and solves the deterministic convex problem
  per path — M4a's Newton system with per-bin coefficients ``A L_k^-beta``,
  batched. More information cannot cost more, so the average is a
  perfect-information lower bound on the adaptive optimum.

The bracket is too loose to grade against and exactly tight enough for the thing
that matters: **the red-flag test becomes rigorous**. No adapted policy can beat
perfect information, so an agent below the clairvoyant bound is a defect with a
proof rather than a discovery. Where M4a's red flag rested on an algebraic
certificate, M4b's rests on a relaxation.

The problem, in the trade weights
---------------------------------
Everything M4a's :mod:`~temper.oracle.powerlaw` established carries over with one
factor added. Participation is ``p_k = n_k / (dt v_hourly L_k)``, so the temporary
charge on bin ``k`` is

.. code::

    h(p_k) w_k = A * L_k^-beta * w_k^(1 + beta)
    A = eta sigma BPS (X / (dt v_hourly))^beta          (power_law_charge)

and nothing else in the world moves: the price shock, permanent impact and the
half-spread are Phase 1's, so the schedule-invariant constant
(:func:`~temper.oracle.powerlaw.schedule_invariant_bps`) and the inventory penalty
``lambda B sum h^2`` are M4a's verbatim. **The frozen objective is untouched** —
``V`` is still price-shortfall variance and liquidity enters ``E[cost]`` through
Jensen, so invariant 7 needs no amendment.

Three optima, and only the middle one is the milestone's denominator
--------------------------------------------------------------------
====================  ======================================================
:func:`static_optimum`  the best *fixed* schedule that knows the liquidity
                        **law**. A fixed schedule pays
                        ``A E[L^-beta] sum w^(1+beta)``, which is M4a's problem
                        at an inflated coefficient — so this is a closed form
                        (a certified Newton solve), not a simulation.
``power_law_optimum``   M4a's schedule, which knows no liquidity at all. Its
                        excess over the static optimum is the **level shift**: a
                        constant any static solver picks up for free, and *not*
                        the agent's to be credited with.
:func:`adaptive_optimum` the optimum over all adapted policies. ``(k, x_k, L_k)``
                        is a sufficient statistic under i.i.d. liquidity, so the
                        DP over that state is the optimum over every policy, not
                        merely over the ones with that observation.
====================  ======================================================

The denominator of M4b's headline is ``J_static* - J_DP``. Differencing the two
*static* rungs from simulations instead of closed forms would turn a 0.002 bps
quantity into noise, and that difference is what decides whether the milestone's
headline is adaptivity or a re-solve.

No scipy, as everywhere else in this package: the DP is `np.interp` and a
vectorised golden-section search, and the clairvoyant solve is a batched Newton
on stacked tridiagonal systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .liquidity import LiquidityLaw
from .model import Market
from .powerlaw import (
    TemporaryCharge,
    inventory_penalty_scale,
    optimum_for_charge,
    power_law_charge,
    schedule_invariant_bps,
)
from .schedules import twap_trajectory

#: Inventory grid resolution for the value functions. The brief pre-states
#: ">= 1601 points, Richardson residual reported"; the error is second order in
#: the spacing (linear interpolation of a convex value function), so the residual
#: is a measured quantity rather than a hope — see
#: :func:`richardson_residual`.
DEFAULT_GRID_POINTS = 1601

#: Gauss–Hermite nodes over ``log L``. Measured to converge by five and pinned at
#: fifteen: the quadrature is the cheap axis (the grid dominates the cost by two
#: orders), so sitting three times past the knee costs nothing and removes the
#: node count from the list of things a later session has to re-argue.
DEFAULT_QUADRATURE_NODES = 15

#: Golden-section iterations per stage problem. The bracket starts one order wide
#: and shrinks by 0.618 each step, so ninety takes it below float resolution
#: (0.618**90 ~ 1e-19) with margin; it is a runaway guard, not a working limit.
STAGE_SEARCH_STEPS = 90

#: Damped-Newton iterations for the batched clairvoyant solve. M4a's scalar solver
#: converges in single digits; sixty is the same runaway guard one axis wider.
MAX_BATCH_NEWTON_STEPS = 60

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


# ---------------------------------------------------------------------------
# The static rung: M4a's problem at an inflated coefficient
# ---------------------------------------------------------------------------


def liquidity_charge(
    market: Market, order_size: float, law: LiquidityLaw
) -> TemporaryCharge:
    """The charge a **fixed** schedule pays under `law`, in the trade weights.

    ``E[A L_k^-beta w_k^(1+beta)] = A E[L^-beta] w_k^(1+beta)`` because a fixed
    schedule's weights are not random, so the expectation passes straight through
    the multiplier and lands on the closed-form moment
    (:meth:`~temper.oracle.liquidity.LiquidityLaw.inverse_power_moment`). The
    result is M4a's charge at a larger scale and the *same* exponent, which is why
    the static optimum is a certified Newton solve rather than a simulation — and
    why the level shift can be computed to full precision instead of being
    differenced out of two Monte-Carlo levels.

    ``E[L^-beta] >= 1`` by Jensen at ``E[L] = 1``: dispersion in liquidity costs a
    fixed schedule something even though the mean liquidity is unchanged.
    """
    base = power_law_charge(market, order_size)
    return TemporaryCharge(
        scale=base.scale * law.inverse_power_moment(market.temp_exponent),
        exponent=base.exponent,
        encoding=base.encoding,
    )


def static_optimum(
    market: Market, order_size: float, lambda_risk: float, law: LiquidityLaw
) -> np.ndarray:
    """The best fixed schedule that knows the liquidity *law* — the denominator's top.

    M4b's advantage is measured from here, **not** from M4a's schedule: the gap
    between the two is a level shift any static solver picks up for free by
    re-solving at the inflated coefficient, and crediting it to the agent would
    make a re-solve look like adaptivity.
    """
    return optimum_for_charge(
        market, order_size, lambda_risk, liquidity_charge(market, order_size, law)
    )


# ---------------------------------------------------------------------------
# Conditional cost — the graded quantity, vectorised over paths
# ---------------------------------------------------------------------------


def path_objective_bps(
    weights,
    multipliers,
    market: Market,
    order_size: float,
    lambda_risk: float,
) -> np.ndarray:
    """``E[cost | L] + lambda V`` in bps, per path — the full frozen objective.

    Exact, not sampled: the price shock enters realised cost only through M1a's
    affine term and the policy never sees a price, so conditioning on the
    liquidity path removes *all* of the price randomness analytically. What is
    left is a deterministic function of ``(weights, L)``, and the only Monte-Carlo
    error anywhere downstream is liquidity dispersion.

    `weights` is ``(paths, n_bins)`` or ``(n_bins,)`` and broadcasts against
    `multipliers` of the same shape, so a *fixed* schedule priced on many paths
    and a *policy's* per-path schedules go through one function. This is the
    vectorised twin of :func:`~temper.oracle.cost.cost_moments` with its
    ``liquidity`` argument; ``tests/test_m4b_adaptive_oracle.py`` pins the two
    against each other, which is what stops the fast route drifting from the one
    the grader uses.
    """
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    liquidity = np.atleast_2d(np.asarray(multipliers, dtype=float))
    w, liquidity = np.broadcast_arrays(w, liquidity)
    if w.shape[-1] != market.n_bins:
        raise ValueError(
            f"weights must carry {market.n_bins} bins, got {w.shape[-1]}"
        )

    beta = market.temp_exponent
    scale = power_law_charge(market, order_size).scale
    holdings = 1.0 - np.cumsum(w, axis=-1) + w  # inventory *before* each bin, /X
    temporary = np.sum(scale * liquidity ** (-beta) * w ** (1.0 + beta), axis=-1)
    penalty = lambda_risk * inventory_penalty_scale(market) * np.sum(
        holdings**2, axis=-1
    )
    return temporary + penalty + schedule_invariant_bps(market, order_size)


# ---------------------------------------------------------------------------
# The dynamic program
# ---------------------------------------------------------------------------


def _stage_minimum(
    inventory: np.ndarray,
    coefficient: np.ndarray,
    grid: np.ndarray,
    continuation: np.ndarray,
    exponent: float,
    order_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``min_{0<=n<=x} c (n/X)^(1+beta) + W(x - n)``, vectorised, by golden section.

    Both terms are convex in ``n`` — the charge because ``1 + beta > 1``, the
    continuation because a linear interpolant of convex samples is convex — so the
    stage objective is unimodal and golden section is sound rather than merely
    convergent.

    **Interpolate, do not snap.** The obvious alternative is to restrict the
    action to the inventory grid and take the best node; it costs an order of
    magnitude in the feasible upper bound, which is the difference between a
    reference that can carry a 10 % tolerance and one that cannot. The brief
    pre-states a 2 %-of-advantage band on that bound precisely so this choice is
    checked rather than trusted.
    """
    low = np.zeros_like(inventory)
    high = inventory.copy()

    def value(trade: np.ndarray) -> np.ndarray:
        return coefficient * (trade / order_size) ** (
            1.0 + exponent
        ) + np.interp(inventory - trade, grid, continuation)

    left = high - _GOLDEN * (high - low)
    right = low + _GOLDEN * (high - low)
    f_left, f_right = value(left), value(right)
    for _ in range(STAGE_SEARCH_STEPS):
        lower = f_left < f_right
        high = np.where(lower, right, high)
        low = np.where(lower, low, left)
        left = high - _GOLDEN * (high - low)
        right = low + _GOLDEN * (high - low)
        f_left, f_right = value(left), value(right)

    trade = 0.5 * (low + high)
    return trade, value(trade)


@dataclass(frozen=True)
class AdaptiveOptimum:
    """The DP's value, and the policy it implies — converged, **not** certified.

    :attr:`objective_bps` is the optimum over all adapted policies, including the
    schedule-invariant constant so it is directly comparable with a
    :class:`~temper.oracle.cost.CostMoments` objective. The word "certified" is
    deliberately absent everywhere this class is reported: M4a earned that word
    with a KKT residual and a Cholesky factorisation, and this milestone's honest
    claim is "converged, and bracketed by a perfect-information relaxation".
    """

    lambda_risk: float
    objective_bps: float
    grid_points: int
    quadrature_nodes: int
    sigma_log_name: str
    market: Market = field(repr=False)
    order_size: float = field(repr=False)
    #: ``continuations[k]`` is ``E_L[V_k(y, L)]`` on :attr:`grid` — the value of
    #: *arriving* at bin ``k`` holding ``y``, before that bin's multiplier is
    #: revealed. ``continuations[0](X)`` is the answer.
    continuations: tuple[np.ndarray, ...] = field(repr=False, default=())
    grid: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def greedy_weights(self, multipliers) -> np.ndarray:
        """The DP's own policy, rolled out on given liquidity paths.

        A **real policy**: it observes ``(k, x_k, L_k)`` and nothing else, solves
        the same stage problem the backward pass solved, and force-liquidates at
        the final bin exactly as :class:`~temper.env.ExecutionEnv` does. Its mean
        conditional cost is therefore an unbiased estimate of an attainable value,
        which is what makes it a *feasible upper bound* on the optimum rather than
        a second opinion about it.

        Returns trade weights, ``(paths, n_bins)``, summing to one per path.
        """
        liquidity = np.atleast_2d(np.asarray(multipliers, dtype=float))
        n_bins = self.market.n_bins
        if liquidity.shape[-1] != n_bins:
            raise ValueError(
                f"liquidity paths must carry {n_bins} bins, got {liquidity.shape[-1]}"
            )
        beta = self.market.temp_exponent
        scale = power_law_charge(self.market, self.order_size).scale
        coefficients = scale * liquidity ** (-beta)

        inventory = np.full(liquidity.shape[0], self.order_size)
        weights = np.empty((liquidity.shape[0], n_bins))
        for bin_index in range(n_bins):
            if bin_index == n_bins - 1:
                # The terminal constraint x_N = 0, the env's force-liquidation.
                trade = inventory
            else:
                trade, _ = _stage_minimum(
                    inventory,
                    coefficients[:, bin_index],
                    self.grid,
                    self.continuations[bin_index + 1],
                    beta,
                    self.order_size,
                )
            weights[:, bin_index] = trade / self.order_size
            inventory = inventory - trade
        return weights

    def as_dict(self) -> dict:
        return {
            "objective_bps": self.objective_bps,
            "grid_points": self.grid_points,
            "quadrature_nodes": self.quadrature_nodes,
            "certified": False,
            "reference_kind": "converged and bracketed",
        }


def adaptive_optimum(
    market: Market,
    order_size: float,
    lambda_risk: float,
    law: LiquidityLaw,
    *,
    points: int = DEFAULT_GRID_POINTS,
    nodes: int = DEFAULT_QUADRATURE_NODES,
) -> AdaptiveOptimum:
    """Backward value iteration over ``(bin, inventory, multiplier)``.

    ``V_k(x, L) = lambda B (x/X)^2 + min_n [ A L^-beta (n/X)^(1+beta) + W_{k+1}(x-n) ]``
    with ``W_k(y) = E_L[V_k(y, L)]`` and the last bin force-liquidating, so
    ``W_{n_bins}`` exists only at ``y = 0`` and never has to be represented.

    Because ``L`` is i.i.d., the continuation ``W`` does not depend on the
    multiplier that was just observed, so exactly one array per bin is stored and
    the quadrature collapses onto it. That is the same sufficiency the milestone
    checks by re-running this on an augmented state: if carrying ``L_{k-1}``
    *improves* the value, the process implementation is not i.i.d. — a bug in the
    env, not a discovery about markets.

    The value returned is the full objective, constant included.
    """
    if order_size <= 0.0:
        raise ValueError(f"order_size must be positive, got {order_size}")
    if lambda_risk < 0.0:
        raise ValueError(f"lambda_risk must be non-negative, got {lambda_risk}")
    if market.n_bins < 2:
        raise ValueError(f"the dynamic program needs n_bins >= 2, got {market.n_bins}")
    if points < 3:
        raise ValueError(f"the inventory grid needs at least three points, got {points}")

    beta = market.temp_exponent
    scale = power_law_charge(market, order_size).scale
    penalty = lambda_risk * inventory_penalty_scale(market)
    grid = np.linspace(0.0, order_size, points)
    multipliers, quadrature = law.quadrature(nodes)
    coefficients = scale * multipliers ** (-beta)
    n_nodes = coefficients.size

    holdings = (grid / order_size)[:, None]

    # The last bin: the terminal constraint leaves no choice, so the stage value
    # is the charge on the whole remaining position.
    stage = coefficients[None, :] * holdings ** (1.0 + beta)
    continuation = (penalty * holdings**2 + stage) @ quadrature

    continuations: list[np.ndarray] = [np.empty(0)] * market.n_bins
    continuations[market.n_bins - 1] = continuation

    flat_inventory = np.repeat(grid, n_nodes)
    flat_coefficients = np.tile(coefficients, points)
    flat_holdings = flat_inventory / order_size
    for bin_index in range(market.n_bins - 2, -1, -1):
        _, best = _stage_minimum(
            flat_inventory,
            flat_coefficients,
            grid,
            continuation,
            beta,
            order_size,
        )
        values = (penalty * flat_holdings**2 + best).reshape(points, n_nodes)
        continuation = values @ quadrature
        continuations[bin_index] = continuation

    varying = float(np.interp(order_size, grid, continuation))
    return AdaptiveOptimum(
        lambda_risk=lambda_risk,
        objective_bps=varying + schedule_invariant_bps(market, order_size),
        grid_points=points,
        quadrature_nodes=n_nodes,
        sigma_log_name=law.name,
        market=market,
        order_size=order_size,
        continuations=tuple(continuations),
        grid=grid,
    )


@dataclass(frozen=True)
class AugmentedOptimum:
    """The same DP on a state that also carries the *previous* multiplier.

    M4b's whole reference rests on ``(k, x_k, L_k)`` being a sufficient statistic:
    if it is, the dynamic program over that state is the optimum over **all**
    adapted policies rather than merely over the ones with that observation, and
    "the agent could have done better with a richer observation" is answerable
    instead of arguable. Under i.i.d. liquidity it is sufficient, because past
    liquidity carries no information about future liquidity.

    That is checked, not asserted. This solve carries ``L_{k-1}`` as a genuine
    state coordinate — the continuation is a value **per (inventory, previous
    node)** and the expectation is taken against
    :meth:`~temper.oracle.liquidity.LiquidityLaw.transition_quadrature`'s
    conditional weights — so a law whose transition depended on the previous draw
    would produce columns that differ and a strictly better value.

    A value that *improves* is therefore not a discovery about markets: it means
    the process implementation is not i.i.d., which is a bug in the env.
    :attr:`column_spread` is the direct measurement of the same thing, and it is
    the sharper of the two — it says the continuations are equal, not merely that
    two scalars agreed.
    """

    objective_bps: float
    #: Largest spread across the previous-multiplier columns of any continuation,
    #: in bps. Zero to float precision exactly when liquidity is memoryless.
    column_spread: float
    grid_points: int
    quadrature_nodes: int


def augmented_optimum(
    market: Market,
    order_size: float,
    lambda_risk: float,
    law: LiquidityLaw,
    *,
    points: int = DEFAULT_GRID_POINTS,
    nodes: int = DEFAULT_QUADRATURE_NODES,
) -> AugmentedOptimum:
    """:func:`adaptive_optimum` with ``L_{k-1}`` in the state — task 1(e)'s check."""
    if market.n_bins < 2:
        raise ValueError(f"the dynamic program needs n_bins >= 2, got {market.n_bins}")

    beta = market.temp_exponent
    scale = power_law_charge(market, order_size).scale
    penalty = lambda_risk * inventory_penalty_scale(market)
    grid = np.linspace(0.0, order_size, points)
    multipliers, transition = law.transition_quadrature(nodes)
    coefficients = scale * multipliers ** (-beta)
    n_nodes = coefficients.size

    holdings = (grid / order_size)[:, None]

    # V_{N-1}(x, L_j): the terminal constraint leaves no choice.
    values = penalty * holdings**2 + coefficients[None, :] * holdings ** (1.0 + beta)
    # W_{N-1}(y, previous p) = sum_j Q[p, j] V(y, L_j) — one column per previous node.
    continuation = values @ transition.T
    spread = float(np.ptp(continuation, axis=1).max())

    flat_inventory = np.repeat(grid, n_nodes)
    flat_coefficients = np.tile(coefficients, points)
    flat_holdings = flat_inventory / order_size
    for _ in range(market.n_bins - 2, -1, -1):
        # State (x_i, current node j) looks up the continuation column indexed by
        # *its own* node, because at the next bin this draw is the previous one.
        best = np.empty(points * n_nodes)
        for node in range(n_nodes):
            select = slice(node, None, n_nodes)
            _, best[select] = _stage_minimum(
                flat_inventory[select],
                flat_coefficients[select],
                grid,
                continuation[:, node],
                beta,
                order_size,
            )
        values = (penalty * flat_holdings**2 + best).reshape(points, n_nodes)
        continuation = values @ transition.T
        spread = max(spread, float(np.ptp(continuation, axis=1).max()))

    return AugmentedOptimum(
        objective_bps=float(np.interp(order_size, grid, continuation[:, 0]))
        + schedule_invariant_bps(market, order_size),
        column_spread=spread,
        grid_points=points,
        quadrature_nodes=n_nodes,
    )


def richardson_residual(coarse: float, fine: float) -> tuple[float, float]:
    """Richardson-extrapolate two second-order values and report the residual.

    Linear interpolation of a convex value function converges from *above* at
    second order in the grid spacing, so with ``fine`` at half the spacing of
    ``coarse`` the extrapolant is ``fine + (fine - coarse) / 3`` and the residual
    — how far the fine grid still is from the limit — is ``|fine - coarse| / 3``.
    Reported rather than asserted: it is the numerical uncertainty in a reference
    the milestone is explicit about not having certified.
    """
    correction = (fine - coarse) / 3.0
    return fine + correction, abs(correction)


# ---------------------------------------------------------------------------
# The clairvoyant relaxation — M4a's Newton, per-bin coefficients, batched
# ---------------------------------------------------------------------------


def clairvoyant_trajectories(
    market: Market,
    order_size: float,
    lambda_risk: float,
    multipliers,
) -> np.ndarray:
    """The perfect-information optimum per liquidity path — a **rigorous** bound.

    Give the optimiser the whole path in advance and the problem separates into
    one deterministic convex solve per path: M4a's objective with the single
    coefficient ``A`` replaced by the per-bin vector ``A L_k^-beta``. Strict
    convexity, the tridiagonal Hessian, the damped Newton step and the descent
    slack are all :func:`~temper.oracle.powerlaw.optimum_for_charge`'s, one axis
    wider; ``tests/test_m4b_adaptive_oracle.py`` pins the batched solver against
    the scalar one on a constant path, where the two must agree to 1e-10 of ``X``.

    More information cannot cost more, so ``mean(J_clairvoyant) <= J_adaptive``
    for *any* adapted policy — including the true optimum and including a trained
    agent. That inequality is what makes M4b's red-flag test rigorous where M4a's
    rested on an algebraic certificate: an agent below this bound is a defect with
    a proof, not a discovery.

    Returns inventory trajectories, ``(paths, n_bins + 1)``.
    """
    liquidity = np.atleast_2d(np.asarray(multipliers, dtype=float))
    n_bins = market.n_bins
    if liquidity.shape[-1] != n_bins:
        raise ValueError(
            f"liquidity paths must carry {n_bins} bins, got {liquidity.shape[-1]}"
        )
    if n_bins < 2:
        return np.tile(np.array([order_size, 0.0]), (liquidity.shape[0], 1))

    beta = market.temp_exponent
    scales = power_law_charge(market, order_size).scale * liquidity ** (-beta)
    penalty = lambda_risk * inventory_penalty_scale(market)
    paths = scales.shape[0]
    interior = n_bins - 1

    def weights_of(trajectory: np.ndarray) -> np.ndarray:
        return -np.diff(trajectory, axis=1) / order_size

    def objective(trajectory: np.ndarray) -> np.ndarray:
        w = weights_of(trajectory)
        holdings = trajectory[:, :-1] / order_size
        return np.sum(scales * w ** (1.0 + beta), axis=1) + penalty * np.sum(
            holdings**2, axis=1
        )

    x = np.tile(twap_trajectory(market, order_size), (paths, 1))
    best = objective(x)
    index = np.arange(interior)
    off = np.arange(interior - 1)

    for _ in range(MAX_BATCH_NEWTON_STEPS):
        w = weights_of(x)
        marginal = scales * (1.0 + beta) * w**beta / order_size
        gradient = (
            marginal[:, 1:]
            - marginal[:, :-1]
            + 2.0 * penalty * x[:, 1:-1] / order_size**2
        )
        curvature = (
            scales * (1.0 + beta) * beta * w ** (beta - 1.0) / order_size**2
        )
        hessian = np.zeros((paths, interior, interior))
        hessian[:, index, index] = (
            curvature[:, :-1] + curvature[:, 1:] + 2.0 * penalty / order_size**2
        )
        hessian[:, off, off + 1] = -curvature[:, 1:-1]
        hessian[:, off + 1, off] = -curvature[:, 1:-1]
        step = np.linalg.solve(hessian, -gradient[..., None])[..., 0]

        # Per-path damping. A path whose Newton step overshoots into an infeasible
        # schedule halves its own length while the rest keep theirs, which is the
        # scalar solver's line search with the loop moved inside the array.
        slack = 8.0 * np.finfo(float).eps * np.abs(best)
        length = np.ones(paths)
        accepted = np.zeros(paths, dtype=bool)
        moved = x.copy()
        for _ in range(80):
            trial = x.copy()
            trial[:, 1:-1] += length[:, None] * step
            feasible = np.all(weights_of(trial) > 0.0, axis=1)
            # Evaluate only the feasible rows: `w ** beta` on a negative weight is
            # a NaN, and `filterwarnings = error` makes that a test failure rather
            # than a quiet nan. Infeasible rows are scored at their current point.
            guarded = np.where(feasible[:, None], trial, x)
            value = np.where(feasible, objective(guarded), np.inf)
            good = feasible & (value <= best + slack) & ~accepted
            moved[good] = trial[good]
            accepted |= good
            if accepted.all():
                break
            length = np.where(accepted, length, 0.5 * length)

        settled = np.array_equal(moved, x)
        x = moved
        best = np.minimum(best, objective(x))
        if settled:
            break

    return x
