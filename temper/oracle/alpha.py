"""The alpha-aware optimum — a dynamic program over ``(inventory, signal)``.

M5's reference, and the third kind of confidence this repo has had to name. M4a
could **certify** its optimum: strictly convex on the reachable set, Cholesky-PD
Hessian, relative KKT residual 1.2e-15. M4b's dynamic program could not, and said
so — *converged, and bracketed by a perfect-information relaxation*. This module
is M4b's successor and inherits the same word discipline, with one difference that
matters: **M5 uses both kinds at once**, and every report has to say which is
which.

* :func:`alpha_optimum` is **converged**, not certified. It is a dynamic program.
* :func:`execution_floor_bps` is M4a's **certified** optimum, used here as a
  rigorous lower bound on the half of the objective the signal cannot touch.

The world, and the one term that is new
---------------------------------------
M4a's: FrontierView's 0.6-power temporary impact, deterministic liquidity.
**Not** M4b's stochastic liquidity — bundled, a red result could not be attributed,
and the two adaptivities respond to different randomness and would compete for the
same schedule shape. What is added is an observation:
:class:`~temper.oracle.signal.AlphaSignal` puts ``s_k`` at the decision point for
bin ``k``, with ``E[xi_{k+1} | s_k] = rho s_k``.

The env's realised shortfall carries the price walk as ``-w_k * price_bps`` with
``price_bps`` containing ``+walk``, so summing by shock rather than by bin gives a
noise term of ``-A sum_k h_k xi_k`` with ``h_k = x_k / X`` the inventory *before*
bin ``k`` and ``A = sigma_bin * BPS`` (:func:`alpha_coefficient`). A positive shock
is a price that rose, which is money to a seller — so a policy that sees ``s_k > 0``
should *hold* inventory through bin ``k + 1``, and one that sees ``s_k < 0`` should
get out ahead of the fall. Conditioning on the whole signal path,

.. code::

    E[cost | s] = A_pow sum_k w_k^(1+beta)          temporary impact
                + lambda B sum_k h_k^2              inventory risk
                - A rho sum_{k>=1} h_k s_{k-1}      alpha
                + schedule_invariant_bps            permanent + half-spread

which is a **closed form again** — the fourth rung of the pattern §9 has been
building since M2 (price-free, then conditional on liquidity, now conditional on
the signal): the policy's actions are deterministic given whatever it observed, so
conditioning on the observation removes all of the price randomness analytically.
There is no price sampling anywhere in this module.

``h_0 = 1`` for every schedule and ``xi_0`` is predicted by nothing, so the first
bin's shock is absent from the conditional cost entirely: the sum starts at
``k = 1``.

The advantage is a difference of two much larger numbers
---------------------------------------------------------
At ``rho = 0.01`` the optimum monetises ~0.148 bps of signal and pays ~0.067 bps
back in worse impact and risk — **45 % of the gross effect is given back** — and
the ~0.081 bps that survives is the milestone's headline. A single capture
fraction against that net number cannot tell a policy that captures 0.15 and pays
0.07 from one that captures 0.25 and pays 0.17, so :class:`AlphaOptimum` carries
the decomposition as first-class fields rather than leaving it to a caller, and
**the identity ``J = impact + risk + alpha + invariant`` is asserted at every node
of every stage** (:attr:`AlphaOptimum.node_identity_residual_bps`) rather than
checked once at the root.

The decomposition is exact, not sampled. The backward pass carries three companion
value functions through the *same* interpolation the value iteration uses, so each
term is an expectation under the same quadrature measure the objective is, and the
only thing separating their sum from the value is the order three linear
interpolations are added in — 1e-15 bps, measured and reported.

The red-flag test moves to the certified half
----------------------------------------------
M4b's hard red flag was the perfect-information relaxation. Here it dies:
:func:`clairvoyant_price_values` measures it at ~-89 bps against an advantage of
0.081, which is **1 100x too loose to ever fire**. Its replacement is rigorous,
tight, and certified. Impact and risk are convex in the trade weights and involve
no signal at all, so for **any** policy, adapted or not,

.. code::

    E[impact + risk] >= min over deterministic schedules = J_M4a_varying

by Jensen — with equality only at M4a's optimum, which is the one object in this
milestone that carries a certificate. An agent reporting execution cheaper than
that has a defect with a proof behind it.

No scipy, as everywhere else in this package: the dynamic program is ``np.interp``
and a vectorised golden-section search, and the clairvoyant relaxation is the same
search over a batched uniform-grid interpolant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .model import BPS, Market
from .powerlaw import (
    inventory_penalty_scale,
    power_law_charge,
    power_law_optimum,
    schedule_invariant_bps,
    varying_objective_bps,
)
from .signal import AlphaSignal

#: Inventory grid resolution for the value functions — M4b's, unchanged, so the
#: two milestones' references are converged to the same standard. The error is
#: second order in the spacing (linear interpolation of a convex value function),
#: so the residual is measured rather than hoped for; see
#: :func:`~temper.oracle.adaptive.richardson_residual`, which M5 reuses.
DEFAULT_SIGNAL_GRID_POINTS = 1601

#: Gauss–Hermite nodes over the signal. M4b's count, and for the same reason: the
#: quadrature is the cheap axis and the inventory grid dominates the cost by two
#: orders, so sitting well past the knee removes the node count from the list of
#: things a later session has to re-argue.
DEFAULT_SIGNAL_QUADRATURE_NODES = 15

#: Golden-section iterations per stage problem. ``0.618**90 ~ 1e-19`` takes the
#: bracket below float resolution with margin; a runaway guard, not a working
#: limit. M4b's constant, kept rather than re-derived.
STAGE_SEARCH_STEPS = 90

#: The clairvoyant relaxation's inventory grid, and the paths it averages over.
#: Both are far coarser than the reference DP's, and for different reasons. The
#: grid is coarse because it can be: doubling it moves the value by 3e-7 bps,
#: because the clairvoyant value function is dominated by a term *linear* in
#: inventory and linear interpolation is exact on one — measured in task 0 rather
#: than assumed, and reported there. The path count is small because the
#: relaxation exists in this milestone **to be retired**: its own half-width is
#: ~9 bps against a quantity three orders smaller than the gap it is being asked
#: to resolve, so a fourth digit would not change a word of the conclusion.
CLAIRVOYANT_GRID_POINTS = 401
CLAIRVOYANT_PATHS = 400

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


# ---------------------------------------------------------------------------
# The two coefficients, and the certified floor
# ---------------------------------------------------------------------------


def alpha_coefficient(market: Market) -> float:
    """``A = sigma_bin * BPS`` — one unit of shock, in bps of notional.

    The same number ``ExecutionEnv`` scales its walk by and the square root of
    :func:`~temper.oracle.powerlaw.inventory_penalty_scale`. Named here because
    M5 is the first milestone in which it appears in ``E[cost]`` rather than only
    in ``V``, and a coefficient re-derived at three call sites is the pattern that
    put a derived-quantities object on FrontierView's own backlog.

    At the reference case it is 42.99 bps — **18x the whole objective**, which is
    the single fact that fixes this milestone's shape.
    """
    return float(market.sigma_bin * BPS)


def execution_floor_bps(
    market: Market, order_size: float, lambda_risk: float
) -> float:
    """``J_M4a_varying`` — the **certified** floor under ``E[impact + risk]``.

    M4a's certified optimum, evaluated on the schedule-varying part of the
    objective. Impact is strictly convex in the trade weights and risk is a convex
    quadratic in the holdings, which are linear in the weights, so their sum is
    strictly convex on the simplex; a policy induces a *random* weight vector
    there, and Jensen gives

    .. code::

        E[impact + risk] >= (impact + risk)(E[w]) >= min over the simplex

    for **any** policy, with equality only when the policy plays M4a's optimum
    almost surely. The signal appears nowhere in either term, which is what makes
    the bound rigorous rather than merely plausible, and the minimiser carries a
    Cholesky factorisation and a 1.2e-15 KKT residual
    (``tests/test_power_law_certificate.py``), which is what makes it *certified*
    rather than converged.

    This is M5's red-flag test, and it replaces an inherited one that could never
    fire — see :func:`clairvoyant_price_values`.
    """
    trajectory = power_law_optimum(market, order_size, lambda_risk)
    return varying_objective_bps(
        trajectory,
        market,
        order_size,
        lambda_risk,
        power_law_charge(market, order_size),
    )


# ---------------------------------------------------------------------------
# Conditional cost — the graded quantity, vectorised over signal paths
# ---------------------------------------------------------------------------


def signal_path_objective_bps(
    weights,
    signals,
    market: Market,
    order_size: float,
    lambda_risk: float,
    signal: AlphaSignal,
) -> np.ndarray:
    """``E[cost | s] + lambda V`` in bps, per path — the full frozen objective.

    Exact, not sampled. The price shock enters realised cost only through M1a's
    affine term, so conditioning on the signal path replaces every shock by its
    conditional mean ``rho s_{k-1}`` and removes **all** of the price randomness
    analytically. What is left is a deterministic function of ``(weights, s)``, and
    the only Monte-Carlo error anywhere downstream is signal dispersion.

    The twin of :func:`~temper.oracle.adaptive.path_objective_bps` one rung along:
    same shapes, same broadcasting, same schedule-invariant constant. ``weights``
    is ``(paths, n_bins)`` or ``(n_bins,)``, so a *fixed* schedule priced on many
    signal paths and a *policy's* per-path schedules go through one function.

    **The objective is unchanged** (invariant 7): ``V`` is still price-shortfall
    variance and the signal moves only ``E[cost]``, through a term that used to be
    zero. No amendment.
    """
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    s = np.atleast_2d(np.asarray(signals, dtype=float))
    w, s = np.broadcast_arrays(w, s)
    if w.shape[-1] != market.n_bins:
        raise ValueError(f"weights must carry {market.n_bins} bins, got {w.shape[-1]}")

    beta = market.temp_exponent
    scale = power_law_charge(market, order_size).scale
    holdings = 1.0 - np.cumsum(w, axis=-1) + w  # inventory *before* each bin, /X
    temporary = np.sum(scale * w ** (1.0 + beta), axis=-1)
    penalty = lambda_risk * inventory_penalty_scale(market) * np.sum(
        holdings**2, axis=-1
    )
    # h_0 is 1 for every schedule and xi_0 is predicted by nothing, so at the
    # model's lag the alpha sum starts at the second bin and reads the signal that
    # preceded it. At lag 0 — task 1's already-landed instrument — every bin's
    # holding pairs with its own signal, and none of those holdings is a decision
    # the signal could have influenced, which is the whole point.
    lag = signal.lag
    alpha = -alpha_coefficient(market) * signal.correlation() * np.sum(
        holdings[..., lag:] * s[..., : market.n_bins - lag], axis=-1
    )
    return temporary + penalty + alpha + schedule_invariant_bps(market, order_size)


def expected_alpha_bps(
    trajectory, market: Market, order_size: float, signal: AlphaSignal
) -> float:
    """The alpha term of a **fixed** schedule's expected cost. Exactly zero.

    ``-A rho sum_{k>=1} h_k E[s]`` with ``E[s] = 0``
    (:meth:`~temper.oracle.signal.AlphaSignal.mean`) — so this returns a float
    zero, and adding it to an objective is the identity operation on every finite
    float.

    Written out and called rather than reasoned about at the call sites, because
    the whole of "M5 needs no new reading of the lambda rule" rests on it. M4b
    needed a *third reading* of the selection rule and a recorded decision between
    two candidates that disagreed, because ``E[L^-beta] > 1`` moved every fixed
    schedule's objective; here nothing moves, and
    ``tools/m5_reference_table.py`` asserts the resulting table is **bit-identical**
    to M4a's rather than merely agreeing with it. A function that could have
    returned something else is what makes that an assertion.
    """
    x = np.asarray(trajectory, dtype=float)
    holdings = x[:-1] / order_size
    return float(
        -alpha_coefficient(market)
        * signal.correlation()
        * np.sum(holdings[signal.lag :] * signal.mean())
    )


# ---------------------------------------------------------------------------
# The dynamic program
# ---------------------------------------------------------------------------


def _stage_minimum(
    inventory: np.ndarray,
    coefficient: float,
    linear: np.ndarray,
    grid: np.ndarray,
    continuation: np.ndarray,
    exponent: float,
    order_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``min_{0<=n<=x} c (n/X)^(1+beta) + l (x-n) + W(x-n)``, by golden section.

    :func:`~temper.oracle.adaptive._stage_minimum` with one term added: ``l`` is
    the alpha price of the inventory carried *out* of this bin, ``-A rho s_k / X``
    per share, and it is the only place the signal enters the stage problem at
    all. A linear term cannot break unimodality — the charge is convex because
    ``1 + beta > 1``, the continuation is convex because a linear interpolant of
    convex samples is convex, and a linear function is both — so golden section
    stays sound rather than merely convergent.

    **Interpolate, do not snap**, for M4b's reason unchanged: restricting the
    action to the inventory grid costs an order of magnitude in the value.

    Vectorised over ``(inventory, signal)`` pairs in one call rather than looped
    per node, which M4b's augmented solve had to do: there the *impact*
    coefficient varied by node so each node needed its own continuation, and here
    the continuation is shared and only the linear coefficient moves.
    """
    low = np.zeros_like(inventory)
    high = inventory.copy()

    def value(trade: np.ndarray) -> np.ndarray:
        remaining = inventory - trade
        return (
            coefficient * (trade / order_size) ** (1.0 + exponent)
            + linear * remaining
            + np.interp(remaining, grid, continuation)
        )

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
class AlphaOptimum:
    """The alpha-aware optimum, its decomposition, and the policy it implies.

    **Converged, not certified** — same word discipline as
    :class:`~temper.oracle.adaptive.AdaptiveOptimum`, and :meth:`as_dict` carries
    ``certified: False`` and a ``reference_kind`` for the same reason. What is
    different from M4b is that the *bracket* is gone from the reference's name: a
    perfect-information relaxation is 1 100x too loose here to say anything, and
    what replaces it bounds only :attr:`impact_bps` + :attr:`risk_bps`. So this is
    "converged, with a certified floor under its execution half", and no phrase
    that suggests the whole value is bracketed appears anywhere.

    Every term is an expectation under the same quadrature the objective is, and
    they close: ``objective_bps == impact_bps + risk_bps + alpha_bps +
    invariant_bps`` to :attr:`identity_residual_bps`, and the same identity holds
    at every node of every stage to :attr:`node_identity_residual_bps`.
    """

    lambda_risk: float
    objective_bps: float
    #: ``E[A_pow sum w^(1+beta)]`` — what the policy pays to trade.
    impact_bps: float
    #: ``lambda B E[sum h^2]`` — what it pays to wait.
    risk_bps: float
    #: ``-A rho E[sum h_k s_{k-1}]`` — what the signal is worth. Negative when the
    #: policy monetises it, and the *gross* effect: the milestone's headline is
    #: this plus the execution premium paid to collect it, which is why one
    #: fraction cannot grade a policy here.
    alpha_bps: float
    #: Permanent cost plus the half-spread — schedule-invariant, so it moves no
    #: minimiser and is carried only so this is comparable with a `CostMoments`
    #: objective.
    invariant_bps: float
    #: ``|objective - (impact + risk + alpha + invariant)|`` at the root, bps.
    identity_residual_bps: float
    #: The largest violation of the same identity at *any* node of *any* stage.
    #: The sharper of the two: it says the decomposition holds everywhere the
    #: value function does, not merely where the answer is read off.
    node_identity_residual_bps: float
    grid_points: int
    quadrature_nodes: int
    signal_name: str
    rho: float
    market: Market = field(repr=False)
    order_size: float = field(repr=False)
    #: Bins ahead the signal pointed. 1 is the model; 0 is task 1's already-landed
    #: instrument, where the alpha term is a *state* cost no action can move and
    #: the whole advantage must therefore collapse to zero.
    bins_ahead: int = 1
    #: ``continuations[k]`` is ``E_s[V_k(y, s)]`` on :attr:`grid` — the value of
    #: *arriving* at bin ``k`` holding ``y``, before that bin's signal is
    #: revealed. ``continuations[0](X)`` is the answer.
    continuations: tuple[np.ndarray, ...] = field(repr=False, default=())
    grid: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def execution_bps(self) -> float:
        """``impact + risk`` — the half of the objective the signal cannot touch.

        What :func:`execution_floor_bps` bounds from below, rigorously, for any
        policy at all.
        """
        return self.impact_bps + self.risk_bps

    def greedy_weights(self, signals) -> np.ndarray:
        """The DP's own policy, rolled out on given signal paths.

        A **real policy**: it observes ``(k, x_k, s_k)`` and nothing else, solves
        the same stage problem the backward pass solved, and force-liquidates at
        the final bin exactly as :class:`~temper.env.ExecutionEnv` does. Its mean
        conditional cost is an unbiased estimate of an attainable value and so a
        *feasible upper bound* on the optimum — the one half of M4b's bracket that
        survives into M5, because it is a statement about a policy rather than
        about information.

        Returns trade weights, ``(paths, n_bins)``, summing to one per path.
        """
        s = np.atleast_2d(np.asarray(signals, dtype=float))
        n_bins = self.market.n_bins
        if s.shape[-1] != n_bins:
            raise ValueError(
                f"signal paths must carry {n_bins} bins, got {s.shape[-1]}"
            )
        beta = self.market.temp_exponent
        coefficient = power_law_charge(self.market, self.order_size).scale
        # At lag 0 the alpha term prices inventory the *previous* decision fixed,
        # so it is constant in this bin's action and the policy correctly ignores
        # the signal entirely. That is what makes the advantage collapse.
        alpha_scale = (
            alpha_coefficient(self.market) * self.rho / self.order_size
            if self.bins_ahead == 1
            else 0.0
        )

        inventory = np.full(s.shape[0], self.order_size)
        weights = np.empty((s.shape[0], n_bins))
        for bin_index in range(n_bins):
            if bin_index == n_bins - 1:
                # The terminal constraint x_N = 0, the env's force-liquidation.
                trade = inventory
            else:
                trade, _ = _stage_minimum(
                    inventory,
                    coefficient,
                    -alpha_scale * s[:, bin_index],
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
            "impact_bps": self.impact_bps,
            "risk_bps": self.risk_bps,
            "alpha_bps": self.alpha_bps,
            "invariant_bps": self.invariant_bps,
            "execution_bps": self.execution_bps,
            "identity_residual_bps": self.identity_residual_bps,
            "node_identity_residual_bps": self.node_identity_residual_bps,
            "grid_points": self.grid_points,
            "quadrature_nodes": self.quadrature_nodes,
            "signal": self.signal_name,
            "rho": self.rho,
            "bins_ahead": self.bins_ahead,
            "certified": False,
            "reference_kind": "converged, with a certified floor under its execution half",
        }


def alpha_optimum(
    market: Market,
    order_size: float,
    lambda_risk: float,
    signal: AlphaSignal,
    *,
    points: int = DEFAULT_SIGNAL_GRID_POINTS,
    nodes: int = DEFAULT_SIGNAL_QUADRATURE_NODES,
) -> AlphaOptimum:
    """Backward value iteration over ``(bin, inventory, signal)``.

    .. code::

        V_k(x, s) = lambda B (x/X)^2
                  + min_n [ A_pow (n/X)^(1+beta)
                            - A rho s (x-n)/X
                            + W_{k+1}(x-n) ]
        W_k(y)    = E_s[ V_k(y, s) ]

    with the last bin force-liquidating, so ``W_{n_bins}`` exists only at ``y = 0``
    and never has to be represented. The alpha term is priced on the inventory
    carried *out* of bin ``k``, because that is the holding ``s_k``'s shock lands
    on; ``h_0 = 1`` for every policy and ``xi_0`` is predicted by nothing, so the
    first bin contributes no alpha to anybody and drops out of the comparison.

    Because ``s`` is i.i.d. the continuation does not depend on the signal that
    was just observed, so exactly one array per bin is stored and the quadrature
    collapses onto it — the same sufficiency M4b checked by re-solving on an
    augmented state, and the same failure mode if it is wrong: a value that
    *improves* under a richer state means the process is not i.i.d., which is a
    bug in the seam rather than a discovery about markets.

    Three companion value functions ride the same recursion so the decomposition
    is **exact rather than sampled**: impact, risk and alpha are each accumulated
    through the same interpolation, under the same policy, against the same
    quadrature. Their sum is checked against the value at every node — that is
    :attr:`AlphaOptimum.node_identity_residual_bps`, and it is the identity
    asserted rather than assumed.

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
    coefficient = power_law_charge(market, order_size).scale
    penalty = lambda_risk * inventory_penalty_scale(market)
    lag = signal.lag
    # Two places the alpha term can be priced, and which one it is *is* the
    # milestone. At lag 1 it lands on the inventory carried **out** of the bin, so
    # it is a function of the action and the policy can move it. At lag 0 it lands
    # on the inventory carried **in**, which the previous decision already fixed:
    # a state cost, constant in the action, mean zero over the signal, and worth
    # exactly nothing to anybody.
    action_scale = (
        alpha_coefficient(market) * signal.correlation() / order_size
        if lag == 1
        else 0.0
    )
    state_scale = (
        0.0 if lag == 1 else alpha_coefficient(market) * signal.correlation()
    )
    grid = np.linspace(0.0, order_size, points)
    signal_nodes, quadrature = signal.quadrature(nodes)
    n_nodes = signal_nodes.size

    holdings = (grid / order_size)[:, None]

    # The last bin: the terminal constraint leaves no choice, so the stage value
    # is the charge on the whole remaining position. At the model's lag no alpha
    # is paid or earned on it because there is no inventory left to carry; at lag
    # 0 the bin still pays its own state cost, which is the same nothing.
    impact = np.repeat(coefficient * holdings ** (1.0 + beta), n_nodes, axis=1)
    risk = np.repeat(penalty * holdings**2, n_nodes, axis=1)
    alpha = -state_scale * holdings * signal_nodes[None, :] * np.ones((points, n_nodes))
    value = impact + risk + alpha
    node_residual = 0.0

    continuation = value @ quadrature
    impact_continuation = impact @ quadrature
    risk_continuation = risk @ quadrature
    alpha_continuation = alpha @ quadrature

    continuations: list[np.ndarray] = [np.empty(0)] * market.n_bins
    continuations[market.n_bins - 1] = continuation

    flat_inventory = np.repeat(grid, n_nodes)
    flat_holdings = flat_inventory / order_size
    flat_nodes = np.tile(signal_nodes, points)
    linear = -action_scale * flat_nodes
    state_alpha = -state_scale * flat_holdings * flat_nodes
    for bin_index in range(market.n_bins - 2, -1, -1):
        trade, best = _stage_minimum(
            flat_inventory,
            coefficient,
            linear,
            grid,
            continuation,
            beta,
            order_size,
        )
        remaining = flat_inventory - trade
        impact = (
            coefficient * (trade / order_size) ** (1.0 + beta)
            + np.interp(remaining, grid, impact_continuation)
        ).reshape(points, n_nodes)
        risk = (
            penalty * flat_holdings**2
            + np.interp(remaining, grid, risk_continuation)
        ).reshape(points, n_nodes)
        alpha = (
            state_alpha
            + linear * remaining
            + np.interp(remaining, grid, alpha_continuation)
        ).reshape(points, n_nodes)
        value = (penalty * flat_holdings**2 + state_alpha + best).reshape(
            points, n_nodes
        )

        # The identity, at every node of every stage. The three companions were
        # accumulated through three separate interpolations of arrays that sum to
        # the one the value used, so the only thing between them is the order the
        # additions happen in — float noise, and it is measured rather than
        # assumed away.
        node_residual = max(
            node_residual, float(np.abs(value - (impact + risk + alpha)).max())
        )

        continuation = value @ quadrature
        impact_continuation = impact @ quadrature
        risk_continuation = risk @ quadrature
        alpha_continuation = alpha @ quadrature
        continuations[bin_index] = continuation

    def at_full_size(values: np.ndarray) -> float:
        return float(np.interp(order_size, grid, values))

    varying = at_full_size(continuation)
    impact_bps = at_full_size(impact_continuation)
    risk_bps = at_full_size(risk_continuation)
    alpha_bps = at_full_size(alpha_continuation)
    invariant = schedule_invariant_bps(market, order_size)
    return AlphaOptimum(
        lambda_risk=lambda_risk,
        objective_bps=varying + invariant,
        impact_bps=impact_bps,
        risk_bps=risk_bps,
        alpha_bps=alpha_bps,
        invariant_bps=invariant,
        identity_residual_bps=abs(varying - (impact_bps + risk_bps + alpha_bps)),
        node_identity_residual_bps=node_residual,
        grid_points=points,
        quadrature_nodes=n_nodes,
        signal_name=signal.name,
        rho=signal.correlation(),
        bins_ahead=lag,
        market=market,
        order_size=order_size,
        continuations=tuple(continuations),
        grid=grid,
    )


# ---------------------------------------------------------------------------
# Sufficiency — the same solve on a state that carries the spent signal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentedAlphaOptimum:
    """The same dynamic program on a state that also carries ``s_{k-1}``.

    M5's reference rests on ``(k, x_k, s_k)`` being a sufficient statistic: if it
    is, the dynamic program over that state is the optimum over **all** adapted
    policies rather than merely over the ones with that observation, and "the
    agent could have done better with a richer observation" is answerable instead
    of arguable. Under an i.i.d. one-step-ahead signal it is sufficient, for a
    reason worth spelling out because it is *not* the liquidity one: ``s_{k-1}``
    predicted ``xi_k``, that shock has already landed, and the inventory it was
    charged on was fixed by the previous decision. The information is not merely
    stale — it has been **spent**.

    That is checked, not asserted. This solve carries ``s_{k-1}`` as a genuine
    state coordinate — the continuation is a value **per (inventory, previous
    node)** and the expectation is taken against
    :meth:`~temper.oracle.signal.AlphaSignal.transition_quadrature`'s conditional
    weights — so a signal whose transition depended on the previous draw would
    produce columns that differ and a strictly better value.

    A value that *improves* is therefore not a discovery about markets: it means
    the seam leaks. :attr:`column_spread` is the direct measurement of the same
    thing and is the sharper of the two — it says the continuations are equal for
    every previous signal, not merely that two scalars agreed at one point.

    **The bar on both is float noise, not a tolerance.** A leak shows up as a
    systematic improvement rather than as scatter, so a loose bar would hide
    exactly the failure the check exists for.
    """

    objective_bps: float
    #: Largest spread across the previous-signal columns of any continuation, in
    #: bps. Zero to float precision exactly when the signal is memoryless *and*
    #: points at a shock that has not yet landed.
    column_spread: float
    grid_points: int
    quadrature_nodes: int
    bins_ahead: int

    def as_dict(self) -> dict:
        return {
            "objective_bps": self.objective_bps,
            "column_spread_bps": self.column_spread,
            "grid_points": self.grid_points,
            "quadrature_nodes": self.quadrature_nodes,
            "bins_ahead": self.bins_ahead,
        }


def augmented_alpha_optimum(
    market: Market,
    order_size: float,
    lambda_risk: float,
    signal: AlphaSignal,
    *,
    points: int = DEFAULT_SIGNAL_GRID_POINTS,
    nodes: int = DEFAULT_SIGNAL_QUADRATURE_NODES,
) -> AugmentedAlphaOptimum:
    """:func:`alpha_optimum` with ``s_{k-1}`` in the state — task 1's check."""
    if market.n_bins < 2:
        raise ValueError(f"the dynamic program needs n_bins >= 2, got {market.n_bins}")
    if points < 3:
        raise ValueError(f"the inventory grid needs at least three points, got {points}")

    beta = market.temp_exponent
    coefficient = power_law_charge(market, order_size).scale
    penalty = lambda_risk * inventory_penalty_scale(market)
    lag = signal.lag
    action_scale = (
        alpha_coefficient(market) * signal.correlation() / order_size
        if lag == 1
        else 0.0
    )
    state_scale = (
        0.0 if lag == 1 else alpha_coefficient(market) * signal.correlation()
    )
    grid = np.linspace(0.0, order_size, points)
    signal_nodes, transition = signal.transition_quadrature(nodes)
    n_nodes = signal_nodes.size

    holdings = (grid / order_size)[:, None]

    # V_{N-1}(x, s_j): the terminal constraint leaves no choice.
    values = (
        penalty * holdings**2
        + coefficient * holdings ** (1.0 + beta)
        - state_scale * holdings * signal_nodes[None, :]
    ) * np.ones((points, n_nodes))
    # W_{N-1}(y, previous p) = sum_j Q[p, j] V(y, s_j) — one column per previous
    # node.
    continuation = values @ transition.T
    spread = float(np.ptp(continuation, axis=1).max())

    flat_inventory = np.repeat(grid, n_nodes)
    flat_holdings = flat_inventory / order_size
    flat_nodes = np.tile(signal_nodes, points)
    linear = -action_scale * flat_nodes
    state_alpha = -state_scale * flat_holdings * flat_nodes
    for _ in range(market.n_bins - 2, -1, -1):
        # State (x_i, current node j) looks up the continuation column indexed by
        # *its own* node, because at the next bin this draw is the previous one.
        best = np.empty(points * n_nodes)
        for node in range(n_nodes):
            select = slice(node, None, n_nodes)
            _, best[select] = _stage_minimum(
                flat_inventory[select],
                coefficient,
                linear[select],
                grid,
                continuation[:, node],
                beta,
                order_size,
            )
        values = (penalty * flat_holdings**2 + state_alpha + best).reshape(
            points, n_nodes
        )
        continuation = values @ transition.T
        spread = max(spread, float(np.ptp(continuation, axis=1).max()))

    return AugmentedAlphaOptimum(
        objective_bps=float(np.interp(order_size, grid, continuation[:, 0]))
        + schedule_invariant_bps(market, order_size),
        column_spread=spread,
        grid_points=points,
        quadrature_nodes=n_nodes,
        bins_ahead=lag,
    )


# ---------------------------------------------------------------------------
# The price-clairvoyant relaxation — computed in order to be retired
# ---------------------------------------------------------------------------


def _batched_interp(
    query: np.ndarray, values: np.ndarray, spacing: float, points: int
) -> np.ndarray:
    """Linear interpolation on a shared uniform grid, one continuation per row.

    ``np.interp`` takes a single ``fp``; the clairvoyant solve needs one value
    function *per path*, because each path knows a different future. The grid is
    uniform, so the bracketing index is arithmetic rather than a search and the
    whole thing is two gathers and a lerp.
    """
    index = np.clip((query / spacing).astype(np.int64), 0, points - 2)
    fraction = query / spacing - index
    lower = np.take_along_axis(values, index, axis=1)
    upper = np.take_along_axis(values, index + 1, axis=1)
    return lower + fraction * (upper - lower)


def clairvoyant_price_values(
    market: Market,
    order_size: float,
    lambda_risk: float,
    shocks,
    *,
    points: int = CLAIRVOYANT_GRID_POINTS,
) -> np.ndarray:
    """The perfect-information optimum per **price** path — M4b's red flag, retired.

    Hand the optimiser the whole shock path in advance and the problem separates
    into one deterministic convex solve per path: M4a's objective with a per-bin
    linear term ``-A xi_k h_k`` added. More information cannot cost more, so the
    average is a lower bound on the value of *any* adapted policy, exactly as it
    was in M4b.

    **And it is useless here, which is the finding.** Price clairvoyance at the
    reference case is worth about -89 bps against an advantage of 0.081 — roughly
    1 100x — because per-bin volatility is 18x the objective and a clairvoyant
    trader is not executing an order but front-running the tape with it. A bound
    that loose can never fire, so M5 retires the test and replaces it with
    :func:`execution_floor_bps`, which is rigorous, tight and certified. The
    number is computed anyway, because retiring an inherited test needs evidence
    rather than a sentence.

    **Why a grid rather than M4b's batched Newton.** With ``A`` three orders above
    the impact scale the unconstrained optimum leaves the reachable set on most
    paths — a favourable shock makes the optimiser want to buy — and the sell-only
    constraint binds. ``w**beta`` has *zero* marginal at ``w = 0`` for
    ``beta < 1``, so there is no interior barrier to keep a Newton iterate feasible
    and an active set would have to be tracked. A per-path dynamic program over
    inventory enforces ``0 <= n <= x`` by construction and reuses the search this
    module already has.

    **The remaining uncertainty is Monte Carlo, not the grid.** Doubling the
    inventory grid on the same paths moves the value by ~3e-7 bps: unlike the
    reference DP's, this value function is dominated by a term *linear* in
    inventory, and linear interpolation is exact on one. What error the
    interpolation does carry runs the conservative way — a linear interpolant of a
    convex function lies above it, so this over-estimates the clairvoyant value
    and therefore *under*-states its looseness. Both facts are measured in task 0
    rather than argued.

    Returns per-path objectives in bps, constant included.
    """
    z = np.atleast_2d(np.asarray(shocks, dtype=float))
    if z.shape[-1] != market.n_bins:
        raise ValueError(f"shock paths must carry {market.n_bins} bins, got {z.shape[-1]}")
    if points < 3:
        raise ValueError(f"the inventory grid needs at least three points, got {points}")

    beta = market.temp_exponent
    coefficient = power_law_charge(market, order_size).scale
    penalty = lambda_risk * inventory_penalty_scale(market)
    amplitude = alpha_coefficient(market)
    paths = z.shape[0]

    grid = np.linspace(0.0, order_size, points)
    spacing = order_size / (points - 1)
    holdings = np.tile(grid / order_size, (paths, 1))
    inventory = np.tile(grid, (paths, 1))

    # The last bin: force-liquidation, so the stage value is the charge on the
    # whole position plus that bin's risk and its now-known shock.
    continuation = (
        coefficient * holdings ** (1.0 + beta)
        + penalty * holdings**2
        - amplitude * z[:, -1:] * holdings
    )

    for bin_index in range(market.n_bins - 2, -1, -1):
        low = np.zeros_like(inventory)
        high = inventory.copy()

        def value(trade: np.ndarray) -> np.ndarray:
            remaining = inventory - trade
            return coefficient * (trade / order_size) ** (
                1.0 + beta
            ) + _batched_interp(remaining, continuation, spacing, points)

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

        continuation = (
            penalty * holdings**2
            - amplitude * z[:, bin_index : bin_index + 1] * holdings
            + value(0.5 * (low + high))
        )

    return continuation[:, -1] + schedule_invariant_bps(market, order_size)
