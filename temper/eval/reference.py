"""M2 task 0 — the oracle-only reference table, and the rule that fixes lambda.

Nothing here knows an agent exists. Every number is a closed form evaluated on a
committed case, which is the whole point: constitution invariant 3 says success
thresholds are pre-stated, and a threshold derived from oracle surface *before*
any training code exists cannot have been chosen to fit a training curve. The
milestone's lambda, its tolerance ``epsilon`` and its trajectory band are all
functions of this module's output and of the rule in
``docs/briefs/M2-ppo-rediscovery.md``; ``configs/m2_ppo.yaml`` writes them down
and ``tests/test_m2_reference.py`` re-derives them, so the config cannot drift
away from the rule that produced it.

The three columns
-----------------
Each schedule is reported as ``E``, ``lambda * V``, and ``lambda * (V - floor)``.
The floor is real and is the M1a finding: the shock lands *before* the first bin
executes, so ``V`` cannot fall below ``sigma_bin^2 X^2`` — one bin of volatility
on the whole position, unavoidable by any schedule (``ARCHITECTURE.md`` §9, *The
shock lands before the bin executes...*). Objective *differences* between
schedules therefore live entirely in the excess over that floor, and a table that
reported only ``lambda * V`` would make the schedules look far more similar than
they are.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from temper.oracle import (
    BPS,
    DEFAULT_GRID_POINTS,
    DEFAULT_QUADRATURE_NODES,
    DEFAULT_SIGNAL_GRID_POINTS,
    DEFAULT_SIGNAL_QUADRATURE_NODES,
    LINEAR_ENCODING,
    POWER_LAW_ENCODING,
    VENDOR_LAMBDA_GRID,
    AlphaOptimum,
    AlphaSignal,
    Market,
    ac_trajectory,
    adaptive_optimum,
    alpha_optimum,
    charge_for,
    clairvoyant_trajectories,
    cost_moments,
    execution_floor_bps,
    expected_alpha_bps,
    expected_cost_moments,
    local_curvature_floor,
    objective_curvature_floor,
    optimal_kappa,
    optimal_trajectory,
    path_objective_bps,
    power_law_optimum,
    schedule_moments,
    signal_path_objective_bps,
    static_optimum,
    trades,
    twap_trajectory,
)
from temper.seeding import M4B_REFERENCE_POOL, M5_REFERENCE_POOL, pool_rng

#: The schedules every table and figure carries, per world (invariant 4). Order
#: is the reporting order and is part of the committed contract.
#:
#: ``optimal`` always names **the certified optimum of the world the row is in**,
#: so nothing downstream needs a branch to know what it is being graded against.
#: In the power-law world that is :func:`~temper.oracle.powerlaw.power_law_optimum`
#: and the sinh gets its own key, ``tangent`` — which is the honest name for it
#: there: :func:`~temper.oracle.schedules.optimal_trajectory` is the exact
#: minimiser of the *tangent's* objective, and evaluating it under the power law
#: is precisely the mis-specification M4a measures. Its excess over ``optimal``
#: is the milestone's available advantage.
REFERENCE_SCHEDULES: dict[str, tuple[str, ...]] = {
    LINEAR_ENCODING: ("twap", "ac", "optimal"),
    POWER_LAW_ENCODING: ("twap", "ac", "tangent", "optimal"),
}


def variance_floor_bps2(market: Market) -> float:
    """``sigma_bin^2`` in bps² — the lowest ``V`` any schedule can reach.

    ``V = sigma_bin^2 * sum_k (x_k / X)^2`` over inventory *before* each bin, and
    ``x_0 = X`` for every schedule, so the ``k = 0`` term alone is
    ``sigma_bin^2`` and the sum can never be smaller. Immediate liquidation
    attains it exactly.
    """
    return float((market.sigma_bin * BPS) ** 2)


@dataclass(frozen=True)
class ScheduleReference:
    """One schedule's row: the trajectory and what the oracle charges it."""

    name: str
    trajectory: np.ndarray
    expected: float          # E[cost], bps
    variance: float          # V[cost], bps^2
    risk: float              # lambda * V, bps
    excess_risk: float       # lambda * (V - floor), bps
    objective: float         # E + lambda * V, bps
    max_bin_fraction: float  # largest single-bin trade, as a fraction of X

    def as_dict(self) -> dict:
        """A JSON-safe view, for `results/` and for the brief's table."""
        return {
            "name": self.name,
            "expected_bps": self.expected,
            "variance_bps2": self.variance,
            "risk_bps": self.risk,
            "excess_risk_bps": self.excess_risk,
            "objective_bps": self.objective,
            "max_bin_fraction": self.max_bin_fraction,
            "trajectory": [float(x) for x in self.trajectory],
        }


@dataclass(frozen=True)
class ReferenceRow:
    """The world's schedules at one lambda, plus what the selection rule reads."""

    lambda_risk: float
    schedules: dict[str, ScheduleReference]
    variance_floor: float
    kappa_horizon: float
    #: Which cost functional every number in this row was charged under. Travels
    #: with the row so a grade cannot be computed at one encoding against an
    #: optimum solved at another (invariant 7) — the same reason `lambda_risk` is
    #: here rather than passed alongside.
    encoding: str = LINEAR_ENCODING

    @property
    def twap(self) -> ScheduleReference:
        return self.schedules["twap"]

    @property
    def ac(self) -> ScheduleReference:
        return self.schedules["ac"]

    @property
    def optimal(self) -> ScheduleReference:
        """The certified optimum **of this row's world** — what an agent is graded on."""
        return self.schedules["optimal"]

    @property
    def tangent(self) -> ScheduleReference | None:
        """The sinh derived at the tangent, where it is not this world's optimum.

        ``None`` in the linearised world, because there it *is* ``optimal`` and a
        second name for one schedule would invite the two to drift apart.
        """
        return self.schedules.get("tangent")

    @property
    def twap_gap(self) -> float:
        """``(J_twap - J_optimal) / J_optimal`` — how discriminative the testbed is.

        If this is small then "the agent got within epsilon of optimal" is a
        claim TWAP also satisfies, and the milestone measures nothing. Condition
        (i) of the selection rule is a floor under it.
        """
        return (self.twap.objective - self.optimal.objective) / self.optimal.objective

    @property
    def available_advantage(self) -> float | None:
        """``J_tangent - J_optimal`` in bps — what there was to be beaten.

        M4a's denominator, and ``None`` where there is nothing to beat: in the
        linearised world the closed form already *is* the optimum, and §1.1 names
        an agent that appears to beat it a red flag rather than a result.

        In bps rather than as a fraction because it is small — 0.0367 bps at the
        reference case — and because the whole reason M3's tolerance cannot be
        reused in this world is that 5 % of that lambda's TWAP gap is 1.8-2.0x
        this entire number. A milestone graded to the TWAP gap here would pass an
        agent that captured none of the mis-specification.
        """
        tangent = self.tangent
        if tangent is None:
            return None
        return tangent.objective - self.optimal.objective

    @property
    def advantage_fraction(self) -> float | None:
        """The available advantage as a fraction of ``J_optimal`` — task 0's gate.

        The gate is >= 1 %: below that the training point is not worth an evening
        and the milestone leads with M4b instead.
        """
        advantage = self.available_advantage
        if advantage is None:
            return None
        return advantage / self.optimal.objective

    def as_dict(self) -> dict:
        return {
            "lambda": self.lambda_risk,
            "encoding": self.encoding,
            "twap_gap": self.twap_gap,
            "available_advantage_bps": self.available_advantage,
            "advantage_fraction": self.advantage_fraction,
            "kappa_horizon": self.kappa_horizon,
            "variance_floor_bps2": self.variance_floor,
            "schedules": {
                name: schedule.as_dict() for name, schedule in self.schedules.items()
            },
        }


def schedule_moments_for(
    encoding: str, trajectory: np.ndarray, market: Market, order_size: float
):
    """The moments of a schedule under `encoding`. One place that maps the worlds.

    The linear branch keeps ``schedule_moments``' `order_size` argument, which
    matters for a realised trajectory whose first point is not the parent size:
    ``eta_tilde`` is a property of the order the env was configured with. The
    power law needs no such care — it is the same function whatever size is
    worked through it, which is exactly why it has no tangent to be taken in the
    wrong place.
    """
    if encoding == LINEAR_ENCODING:
        return schedule_moments(trajectory, market, order_size=order_size)
    if encoding == POWER_LAW_ENCODING:
        return cost_moments(trajectory, market)
    raise ValueError(f"unknown cost encoding {encoding!r}")


def _schedule_reference(
    name: str,
    trajectory: np.ndarray,
    market: Market,
    order_size: float,
    lambda_risk: float,
    floor: float,
    encoding: str,
) -> ScheduleReference:
    moments = schedule_moments_for(encoding, trajectory, market, order_size)
    return ScheduleReference(
        name=name,
        trajectory=trajectory,
        expected=moments.expected,
        variance=moments.variance,
        risk=lambda_risk * moments.variance,
        excess_risk=lambda_risk * (moments.variance - floor),
        objective=moments.objective(lambda_risk),
        max_bin_fraction=float(np.max(trades(trajectory, market))) / order_size,
    )


def reference_trajectories(
    encoding: str, market: Market, order_size: float, lambda_risk: float
) -> dict[str, np.ndarray]:
    """The world's reference schedules at one lambda, by name.

    Every world carries TWAP and the vendored AC schedule (invariant 4) and an
    ``optimal`` that is its *own* certified optimum. The power-law world carries
    one more: the tangent-derived sinh, which is what the closed form actually
    produces there and what the advantage is measured against.
    """
    common = {
        "twap": twap_trajectory(market, order_size),
        "ac": ac_trajectory(market, order_size, lambda_risk),
    }
    if encoding == LINEAR_ENCODING:
        return common | {
            "optimal": optimal_trajectory(market, order_size, lambda_risk)
        }
    if encoding == POWER_LAW_ENCODING:
        return common | {
            "tangent": optimal_trajectory(market, order_size, lambda_risk),
            "optimal": power_law_optimum(market, order_size, lambda_risk),
        }
    raise ValueError(f"unknown cost encoding {encoding!r}")


def reference_row(
    market: Market,
    order_size: float,
    lambda_risk: float,
    *,
    encoding: str = LINEAR_ENCODING,
) -> ReferenceRow:
    """The world's reference schedules at one lambda, priced by that world.

    `encoding` defaults to the linearised world, which is Phase 1's and every
    committed result's. A Phase-2 world is never inherited — the caller names it
    (constitution §4), and ``tests/test_repo_invariants.py`` pins that the
    default here is the Phase-1 one.
    """
    floor = variance_floor_bps2(market)
    trajectories = reference_trajectories(
        encoding, market, order_size, lambda_risk
    )
    return ReferenceRow(
        lambda_risk=lambda_risk,
        schedules={
            name: _schedule_reference(
                name,
                trajectories[name],
                market,
                order_size,
                lambda_risk,
                floor,
                encoding,
            )
            for name in REFERENCE_SCHEDULES[encoding]
        },
        variance_floor=floor,
        kappa_horizon=(
            optimal_kappa(market, order_size, lambda_risk) * market.horizon_hours
        ),
        encoding=encoding,
    )


def reference_table(
    market: Market,
    order_size: float,
    lambdas: Iterable[float] = VENDOR_LAMBDA_GRID,
    *,
    encoding: str = LINEAR_ENCODING,
) -> list[ReferenceRow]:
    """The whole table, ascending in lambda. M0's 17-point grid by default."""
    return [
        reference_row(market, order_size, lam, encoding=encoding)
        for lam in sorted(lambdas)
    ]


# ---------------------------------------------------------------------------
# The selection rule (task 0), applied to the table and to nothing else
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LambdaRule:
    """The brief's rule for fixing the milestone lambda.

    *The smallest lambda on the grid satisfying* ``twap_gap >= min_twap_gap``
    *and* ``max_bin_fraction <= max_bin_fraction`` *for the optimal schedule.*

    Condition (i) keeps the testbed discriminative — "within epsilon of optimal"
    says nothing where TWAP is already near-optimal. Condition (ii) rejects
    degenerate near-immediate liquidation, where the optimal schedule is a single
    large bin and the control problem is trivial. Both are read off the oracle
    before any agent exists.
    """

    min_twap_gap: float = 0.20
    max_bin_fraction: float = 0.50

    def admits(self, row: ReferenceRow) -> bool:
        return (
            row.twap_gap >= self.min_twap_gap
            and row.optimal.max_bin_fraction <= self.max_bin_fraction
        )


class NoAdmissibleLambda(ValueError):
    """No grid point satisfies the rule — the *case* must change, not the rule.

    Raised rather than returning the closest miss on purpose. The brief's
    instruction when the grid is empty is to change the order size or horizon and
    re-run task 0 *now*, from oracle numbers, recording why — never to relax a
    condition after seeing a training curve.
    """


def select_lambda(
    table: Sequence[ReferenceRow], rule: LambdaRule = LambdaRule()
) -> ReferenceRow:
    """The row the rule selects. Deterministic, and a pure function of the table."""
    for row in sorted(table, key=lambda row: row.lambda_risk):
        if rule.admits(row):
            return row
    raise NoAdmissibleLambda(
        f"no lambda on the {len(table)}-point grid has twap_gap >= "
        f"{rule.min_twap_gap:g} and max bin fraction <= {rule.max_bin_fraction:g}; "
        "change the case (order size or horizon) and re-run task 0"
    )


# ---------------------------------------------------------------------------
# The derived trajectory band
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryBand:
    """How far a schedule may sit from the optimum while costing at most ``delta``.

    Not a chosen tolerance: it is what the objective's own curvature implies.
    ``U`` is quadratic in the interior holdings with Hessian ``H``, so
    ``U(x* + d) - U(x*) = d' H d / 2 >= lambda_min(H) |d|^2 / 2`` and an
    objective excess of ``delta`` bps confines ``d`` to

    .. code::

        |d|_2 <= sqrt(2 * delta / lambda_min(H))     shares

    The bound is attained along the flattest eigenvector, so it is tight rather
    than merely valid. Reporting it beside the observed deviation is the point:
    the objective is flat near its minimum by exactly this much, and an
    independently chosen trajectory tolerance would be either vacuous or
    unmeetable for reasons that have nothing to do with the agent.

    **Over which schedules.** ``U`` is exactly quadratic only while the schedule
    is sell-only: permanent impact is charged on ``|n_k|``, so its telescoping to
    a schedule-invariant constant needs monotone inventory (the same assumption
    ``tests/test_variational_certificate.py`` states and checks for the optimum).
    That is not a caveat in practice — it is a property of the reachable set.
    :class:`~temper.env.ExecutionEnv` clips every action to ``[0, remaining]``,
    so no policy can buy back and every trajectory the agent can *possibly*
    realise is monotone. The band is therefore exact on exactly the schedules M2
    grades, which ``tests/test_m2_reference.py`` pins from both ends: tight where
    the quadratic holds, and the env unable to leave that set.

    **Global in the linearised world, local in the power-law one.** Everything
    above holds because ``H`` does not depend on ``x``. Under the power law it
    does — the curvature of ``w ** 1.6`` is ``w ** -0.4``, so a schedule that
    concentrates differently is in a differently-shaped bowl — and the bound
    becomes a statement *at the optimum it was assembled at*. :attr:`local` says
    which kind a band is, and M4a validates the local one by direct evaluation on
    random directions at the band radius rather than by asserting the quadratic
    inequality it no longer satisfies globally.
    """

    delta_objective: float   # the objective excess allowed, bps
    curvature_floor: float   # lambda_min(H), bps per share^2
    bound_shares: float      # the implied |d|_2 bound, shares
    order_size: float        # the parent order the bound is a fraction of
    #: Whether the Hessian this came from is the whole problem's or one point's.
    local: bool = False
    #: The world the curvature was measured in.
    encoding: str = LINEAR_ENCODING

    @property
    def bound_fraction(self) -> float:
        """The bound as a fraction of the parent order — how it reads on a chart."""
        return self.bound_shares / self.order_size

    def as_dict(self) -> dict:
        return {
            "delta_objective_bps": self.delta_objective,
            "curvature_floor_bps_per_share2": self.curvature_floor,
            "bound_shares": self.bound_shares,
            "bound_fraction_of_X": self.bound_fraction,
            "local": self.local,
            "encoding": self.encoding,
        }


def trajectory_band(
    market: Market,
    order_size: float,
    lambda_risk: float,
    delta_objective: float,
    *,
    encoding: str = LINEAR_ENCODING,
) -> TrajectoryBand:
    """The band an objective excess of `delta_objective` bps implies.

    The linearised world takes its curvature from
    :func:`~temper.oracle.schedules.objective_curvature_floor` — the closed form,
    for the reason that function's docstring gives: this is a number a tolerance
    is *divided* by, and an iterative eigensolver a few ulps low would loosen a
    pre-stated band by exactly the amount nobody would notice. The power-law
    world has no such closed form, so its floor is ``eigvalsh`` of the Hessian at
    the certified optimum, and the band it returns is marked
    :attr:`~TrajectoryBand.local`.
    """
    if delta_objective < 0.0:
        raise ValueError(
            f"delta_objective is an objective excess and must be >= 0, "
            f"got {delta_objective}"
        )
    if encoding == LINEAR_ENCODING:
        floor = objective_curvature_floor(market, order_size, lambda_risk)
        local = False
    elif encoding == POWER_LAW_ENCODING:
        optimum = power_law_optimum(market, order_size, lambda_risk)
        floor = local_curvature_floor(
            optimum,
            market,
            order_size,
            lambda_risk,
            charge_for(encoding, market, order_size),
        )
        local = True
    else:
        raise ValueError(f"unknown cost encoding {encoding!r}")
    return TrajectoryBand(
        delta_objective=delta_objective,
        curvature_floor=floor,
        bound_shares=math.sqrt(2.0 * delta_objective / floor),
        order_size=order_size,
        local=local,
        encoding=encoding,
    )


def trajectory_deviation(trajectory, optimum) -> float:
    """``|x - x*|_2`` over the interior holdings, in shares.

    The endpoints are excluded because they are not free: every schedule starts
    at ``X`` and the env's terminal force-liquidation puts every schedule at
    zero. Including them would add two identically-zero terms and quietly make
    every deviation look smaller than it is.
    """
    x = np.asarray(trajectory, dtype=float)
    reference = np.asarray(optimum, dtype=float)
    if x.shape != reference.shape:
        raise ValueError(
            f"trajectories must have the same shape, got {x.shape} and "
            f"{reference.shape}"
        )
    return float(np.linalg.norm(x[1:-1] - reference[1:-1]))


# ---------------------------------------------------------------------------
# M4b — the liquidity world's reference row
# ---------------------------------------------------------------------------

#: The schedules M4b's table carries, in reporting order. The first three are
#: every world's (invariant 4). ``m4a`` is the power-law optimum solved *without*
#: liquidity — where M4a leaves off — and ``static`` is the best fixed schedule
#: that knows the liquidity law. Their difference is the **level shift**, and it
#: is reported as its own line everywhere because it is a constant any static
#: solver picks up for free by re-solving at the inflated coefficient. Crediting
#: it to the agent would make a re-solve look like adaptivity.
LIQUIDITY_SCHEDULES: tuple[str, ...] = ("twap", "ac", "tangent", "m4a", "static")

#: Paths drawn for the two Monte-Carlo bounds in the reference table. The brief
#: pre-states M = 20 000 for the graded evaluation; the table uses the same count
#: so the half-width it reports is the one grading will achieve.
REFERENCE_BOUND_PATHS = 20_000


@dataclass(frozen=True)
class PathBound:
    """A Monte-Carlo bound on the adaptive optimum, **paired** against a closed form.

    The level of any single policy under sampled liquidity has a standard
    deviation of ~0.18 bps, which is three times the whole effect M4b measures:
    an unpaired estimate of a bound is worthless here. But the *static* optimum's
    expectation is a closed form
    (:func:`~temper.oracle.cost.expected_cost_moments`), so scoring every sampled
    policy against the static schedule on the **same** liquidity paths turns a
    level estimate into a difference estimate with a known offset —

    .. code::

        J_policy = J_static* + E[ C_policy(L) - C_static(L) ]

    — which is unbiased, and whose sampling error is ~9x smaller in variance
    because the two share their liquidity. That is the same common-random-numbers
    rule the milestone grades an *agent* under; applying it to the reference's own
    bounds is consistency, not a shortcut. :attr:`paired_sd_bps` and
    :attr:`unpaired_sd_bps` are both reported so the reader can see which it is.
    """

    name: str
    value_bps: float
    half_width_bps: float
    paired_sd_bps: float
    unpaired_sd_bps: float
    paths: int
    paired_against: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value_bps": self.value_bps,
            "half_width_bps": self.half_width_bps,
            "paired_sd_bps": self.paired_sd_bps,
            "unpaired_sd_bps": self.unpaired_sd_bps,
            "paths": self.paths,
            "paired_against": self.paired_against,
        }


def _paired_bound(
    name: str, policy_bps: np.ndarray, static_bps: np.ndarray, static_level: float
) -> PathBound:
    difference = policy_bps - static_bps
    return PathBound(
        name=name,
        value_bps=static_level + float(difference.mean()),
        half_width_bps=float(
            1.96 * difference.std(ddof=1) / math.sqrt(difference.size)
        ),
        paired_sd_bps=float(difference.std(ddof=1)),
        unpaired_sd_bps=float(policy_bps.std(ddof=1)),
        paths=int(difference.size),
        paired_against="static",
    )


@dataclass(frozen=True)
class LiquidityReferenceRow:
    """One lambda in the stochastic-liquidity world: five schedules and three optima.

    **Liquidity is not a cost encoding.** The functional is unchanged — the charge
    is still ``eta sigma p**beta`` — and what M4b randomises is the *market*, so
    :attr:`encoding` stays ``power_law`` and §9's *A metric grades the world that
    charges it* needs no amendment. What changes is that a schedule is no longer
    the only kind of answer: :attr:`adaptive_bps` is the value of a *policy*, and
    the two Monte-Carlo bounds say how well that value is known.

    **Which optimum the lambda rule reads.** :attr:`optimal` is the *static*
    optimum, so :meth:`LambdaRule.admits` reads this row exactly as it reads every
    earlier one — a schedule's TWAP gap and a schedule's largest bin. That is the
    recorded decision, and :attr:`adaptive_twap_gap` carries the other reading
    beside it so the choice is visible rather than implied. The reasons are in
    ``tools/m4b_reference_table.py``'s gate 1 and, in one line: the static reading
    is a closed form and it misses at 10^-4 by 3.94 percentage points, where the
    adaptive reading *clears* the same bar there by 0.011 — a selection that would
    turn on the fifth digit of a numerically-solved value function.
    """

    lambda_risk: float
    schedules: dict[str, ScheduleReference]
    variance_floor: float
    liquidity: dict
    encoding: str = POWER_LAW_ENCODING
    #: The adaptive half, absent on a row built by :func:`static_liquidity_row`.
    #: The lambda rule reads only the *static* rungs (the class docstring says
    #: why), and those are five certified solves where the dynamic program and its
    #: two sampled bounds are minutes — so the rule stays checkable at the top of
    #: every run, which is where ``verify_lambda_rule`` needs it, instead of
    #: becoming a thing a session skips because it is slow.
    adaptive_bps: float | None = None
    adaptive: dict | None = None
    clairvoyant: PathBound | None = None
    feasible: PathBound | None = None
    mean_schedule_max_bin: float | None = None

    def _adaptive(self) -> float:
        if self.adaptive_bps is None:
            raise ValueError(
                f"the row at lambda = {self.lambda_risk:.6e} was built without the "
                "dynamic program; build it with liquidity_reference_row to read an "
                "adaptive quantity"
            )
        return self.adaptive_bps

    @property
    def twap(self) -> ScheduleReference:
        return self.schedules["twap"]

    @property
    def ac(self) -> ScheduleReference:
        return self.schedules["ac"]

    @property
    def tangent(self) -> ScheduleReference:
        return self.schedules["tangent"]

    @property
    def m4a(self) -> ScheduleReference:
        """M4a's optimum, re-priced here — the schedule that knows no liquidity."""
        return self.schedules["m4a"]

    @property
    def static(self) -> ScheduleReference:
        """The best fixed schedule that knows the liquidity *law*."""
        return self.schedules["static"]

    @property
    def optimal(self) -> ScheduleReference:
        """What the lambda rule reads — the static optimum. See the class docstring."""
        return self.static

    @property
    def twap_gap(self) -> float:
        """``(J_twap - J_static*) / J_static*`` — the rule's condition (i)."""
        return (self.twap.objective - self.static.objective) / self.static.objective

    @property
    def adaptive_twap_gap(self) -> float:
        """The same gap read against the DP — the reading *not* used, recorded."""
        adaptive = self._adaptive()
        return (self.twap.objective - adaptive) / adaptive

    @property
    def adaptive_advantage(self) -> float:
        """``J_static* - J_DP`` in bps — **the milestone's denominator**.

        Not ``J_M4a - J_DP``: 3.8 % of that at the trained sigma_L is a level
        shift a static solver gets for free, and the whole claim of M4b is that
        the remainder is something no fixed schedule can capture at all.
        """
        return self.static.objective - self._adaptive()

    @property
    def advantage_fraction(self) -> float:
        """The adaptive advantage as a fraction of ``J_DP`` — task 0's gate 2."""
        return self.adaptive_advantage / self._adaptive()

    @property
    def level_shift(self) -> float:
        """``J_M4a - J_static*`` in bps — the constant, reported on its own line."""
        return self.m4a.objective - self.static.objective

    @property
    def level_shift_fraction(self) -> float:
        """The level shift as a fraction of the adaptive advantage — gate 3.

        The gate that matters most. If the constant is a large fraction of the
        advantage then most of what looks like adaptivity is a re-solve, and the
        milestone's headline has to be restated *before* any training rather than
        caveated afterwards.
        """
        return self.level_shift / self.adaptive_advantage

    @property
    def bracket_bps(self) -> float:
        """Feasible upper minus clairvoyant lower — the reference's uncertainty."""
        if self.feasible is None or self.clairvoyant is None:
            raise ValueError(
                f"the row at lambda = {self.lambda_risk:.6e} carries no sampled "
                "bounds; build it with liquidity_reference_row"
            )
        return self.feasible.value_bps - self.clairvoyant.value_bps

    @property
    def bracket_fraction(self) -> float:
        """The bracket as a fraction of the advantage — task 0's gate 4."""
        return self.bracket_bps / self.adaptive_advantage

    def as_dict(self) -> dict:
        document = {
            "lambda": self.lambda_risk,
            "encoding": self.encoding,
            "liquidity": self.liquidity,
            "twap_gap": self.twap_gap,
            "level_shift_bps": self.level_shift,
            "variance_floor_bps2": self.variance_floor,
            "schedules": {
                name: schedule.as_dict() for name, schedule in self.schedules.items()
            },
        }
        if self.adaptive_bps is not None:
            document |= {
                "adaptive_twap_gap": self.adaptive_twap_gap,
                "adaptive_bps": self.adaptive_bps,
                "adaptive": self.adaptive,
                "adaptive_advantage_bps": self.adaptive_advantage,
                "advantage_fraction": self.advantage_fraction,
                "level_shift_fraction": self.level_shift_fraction,
                "clairvoyant": self.clairvoyant.as_dict(),
                "feasible_upper": self.feasible.as_dict(),
                "bracket_bps": self.bracket_bps,
                "bracket_fraction": self.bracket_fraction,
                "mean_schedule_max_bin": self.mean_schedule_max_bin,
            }
        return document


def liquidity_trajectories(
    market: Market, order_size: float, lambda_risk: float, law
) -> dict[str, np.ndarray]:
    """The liquidity world's five reference schedules, by name.

    All five are *fixed* schedules and every one of them is a closed form or a
    certified solve — no simulation anywhere. That is deliberate: ``m4a`` and
    ``static`` differ by ~0.002 bps at the trained point, and differencing two
    Monte-Carlo levels would turn the milestone's most load-bearing gate into
    noise.
    """
    return {
        "twap": twap_trajectory(market, order_size),
        "ac": ac_trajectory(market, order_size, lambda_risk),
        "tangent": optimal_trajectory(market, order_size, lambda_risk),
        "m4a": power_law_optimum(market, order_size, lambda_risk),
        "static": static_optimum(market, order_size, lambda_risk, law),
    }


def _liquidity_schedule_reference(
    name: str,
    trajectory: np.ndarray,
    market: Market,
    order_size: float,
    lambda_risk: float,
    floor: float,
    law,
) -> ScheduleReference:
    """A fixed schedule's row, priced by the *law* rather than by one sampled path."""
    moments = expected_cost_moments(trajectory, market, law)
    return ScheduleReference(
        name=name,
        trajectory=trajectory,
        expected=moments.expected,
        variance=moments.variance,
        risk=lambda_risk * moments.variance,
        excess_risk=lambda_risk * (moments.variance - floor),
        objective=moments.objective(lambda_risk),
        max_bin_fraction=float(np.max(trades(trajectory, market))) / order_size,
    )


def static_liquidity_row(
    market: Market, order_size: float, lambda_risk: float, law
) -> LiquidityReferenceRow:
    """The liquidity world's five *fixed* rungs at one lambda. No DP, no sampling.

    This is what the lambda selection rule is applied to, and it is the whole of
    M4b's answer to "how is the rule applied in a world that is not a new
    encoding": the liquidity world's static problem is M4a's problem at the
    coefficient ``A E[L^-beta]``, a monotone rescaling, so the rule reads a
    *schedule's* TWAP gap and a *schedule's* largest bin exactly as it has since
    M2. It is also five certified solves rather than minutes of dynamic
    programming, which is what lets ``Experiment.verify_lambda_rule`` keep
    checking it at the top of every run.
    """
    floor = variance_floor_bps2(market)
    trajectories = liquidity_trajectories(market, order_size, lambda_risk, law)
    return LiquidityReferenceRow(
        lambda_risk=lambda_risk,
        schedules={
            name: _liquidity_schedule_reference(
                name, trajectories[name], market, order_size, lambda_risk, floor, law
            )
            for name in LIQUIDITY_SCHEDULES
        },
        variance_floor=floor,
        liquidity=law.as_dict(),
    )


def static_liquidity_table(
    market: Market,
    order_size: float,
    law,
    lambdas: Iterable[float] = VENDOR_LAMBDA_GRID,
) -> list[LiquidityReferenceRow]:
    """The static liquidity table over a grid — what :func:`select_lambda` reads."""
    return [
        static_liquidity_row(market, order_size, lam, law) for lam in sorted(lambdas)
    ]


def liquidity_reference_row(
    market: Market,
    order_size: float,
    lambda_risk: float,
    law,
    *,
    root_seed: int,
    stream_index: int = 0,
    paths: int = REFERENCE_BOUND_PATHS,
    grid_points: int = DEFAULT_GRID_POINTS,
    quadrature_nodes: int = DEFAULT_QUADRATURE_NODES,
) -> LiquidityReferenceRow:
    """One row of M4b's table: five fixed schedules, the DP, and its two bounds.

    The Monte-Carlo paths are drawn from :data:`~temper.seeding.M4B_REFERENCE_POOL`
    — the oracle's own pool. Spending ``eval`` streams on a reference table would
    burn addresses a trained result is reported at, on a computation that has no
    agent in it (the same argument that gave M1's differential its own pool).
    """
    static = static_liquidity_row(market, order_size, lambda_risk, law)
    trajectories = {name: row.trajectory for name, row in static.schedules.items()}
    schedules = static.schedules

    optimum = adaptive_optimum(
        market,
        order_size,
        lambda_risk,
        law,
        points=grid_points,
        nodes=quadrature_nodes,
    )

    rng = pool_rng(root_seed, M4B_REFERENCE_POOL, stream_index)
    multipliers = law.draw(rng, (paths, market.n_bins))
    static_weights = trades(trajectories["static"], market) / order_size
    static_cost = path_objective_bps(
        static_weights, multipliers, market, order_size, lambda_risk
    )

    greedy = optimum.greedy_weights(multipliers)
    feasible = _paired_bound(
        "feasible_upper",
        path_objective_bps(greedy, multipliers, market, order_size, lambda_risk),
        static_cost,
        schedules["static"].objective,
    )
    clairvoyant_x = clairvoyant_trajectories(
        market, order_size, lambda_risk, multipliers
    )
    clairvoyant = _paired_bound(
        "clairvoyant_lower",
        path_objective_bps(
            -np.diff(clairvoyant_x, axis=1) / order_size,
            multipliers,
            market,
            order_size,
            lambda_risk,
        ),
        static_cost,
        schedules["static"].objective,
    )

    return LiquidityReferenceRow(
        lambda_risk=lambda_risk,
        schedules=schedules,
        variance_floor=static.variance_floor,
        liquidity=law.as_dict(),
        adaptive_bps=optimum.objective_bps,
        adaptive=optimum.as_dict(),
        clairvoyant=clairvoyant,
        feasible=feasible,
        mean_schedule_max_bin=float(np.max(greedy.mean(axis=0))),
    )


def liquidity_reference_table(
    market: Market,
    order_size: float,
    law,
    lambdas: Iterable[float] = VENDOR_LAMBDA_GRID,
    *,
    root_seed: int,
    paths: int = REFERENCE_BOUND_PATHS,
    grid_points: int = DEFAULT_GRID_POINTS,
    quadrature_nodes: int = DEFAULT_QUADRATURE_NODES,
) -> list[LiquidityReferenceRow]:
    """The whole liquidity table, ascending in lambda. M0's 17-point grid by default.

    Each row gets its **own** stream index, so adding a lambda cannot move the
    liquidity paths an already-committed row was measured on.
    """
    ordered = sorted(lambdas)
    return [
        liquidity_reference_row(
            market,
            order_size,
            lam,
            law,
            root_seed=root_seed,
            stream_index=index,
            paths=paths,
            grid_points=grid_points,
            quadrature_nodes=quadrature_nodes,
        )
        for index, lam in enumerate(ordered)
    ]
# ---------------------------------------------------------------------------
# M5 — the alpha-aware world: one more reading, and a second kind of confidence
# ---------------------------------------------------------------------------

#: The schedules M5's table carries, in reporting order. Identical to the
#: power-law world's, and that is the milestone's first result rather than an
#: omission: a zero-mean signal leaves every *fixed* schedule exactly where M4a
#: left it, so there is no fifth rung here the way ``static`` was M4b's. What the
#: signal adds is not a schedule but a policy, and a policy has no row in a table
#: of trajectories.
SIGNAL_SCHEDULES: tuple[str, ...] = REFERENCE_SCHEDULES[POWER_LAW_ENCODING]

#: Signal paths drawn for the feasible upper bound. Matches M4b's committed count
#: so the half-width it reports is the one grading will achieve.
SIGNAL_BOUND_PATHS = 20_000


def _signal_schedule_reference(
    name: str,
    trajectory: np.ndarray,
    market: Market,
    order_size: float,
    lambda_risk: float,
    floor: float,
    signal: AlphaSignal,
) -> ScheduleReference:
    """A *fixed* schedule's row in the signal world — M4a's, plus exactly zero.

    The alpha term is computed rather than reasoned away
    (:func:`~temper.oracle.alpha.expected_alpha_bps`): a deterministic schedule's
    holdings do not depend on ``s``, so its expected alpha is
    ``-A rho sum h_k E[s]`` and ``E[s]`` is a float zero. Adding a float zero to
    an objective is the identity operation on every finite float, which is what
    makes "M5's static reading is **bit-identical** to M4a's" an assertion the
    reference table can make rather than a claim it has to be believed on.

    ``variance`` is untouched: the signal moves ``E[cost]`` through a term that
    used to be zero and never touches ``V``, so invariant 7 needs no amendment.
    """
    moments = cost_moments(trajectory, market)
    alpha = expected_alpha_bps(trajectory, market, order_size, signal)
    return ScheduleReference(
        name=name,
        trajectory=trajectory,
        expected=moments.expected + alpha,
        variance=moments.variance,
        risk=lambda_risk * moments.variance,
        excess_risk=lambda_risk * (moments.variance - floor),
        objective=moments.objective(lambda_risk) + alpha,
        max_bin_fraction=float(np.max(trades(trajectory, market))) / order_size,
    )


def signal_static_row(
    market: Market, order_size: float, lambda_risk: float, signal: AlphaSignal
) -> ReferenceRow:
    """The signal world's *fixed* rungs at one lambda. No dynamic program.

    What the lambda selection rule is applied to, and the whole of M5's answer to
    the question M4b had to decide: liquidity moved every fixed schedule's
    objective by ``E[L^-beta] > 1``, so the rule needed a third *reading* and a
    recorded choice between two candidates that disagreed. A zero-mean signal
    moves nothing, so this returns a row that is M4a's to the last bit and the
    selection cannot differ. ``tools/m5_reference_table.py`` asserts exactly that,
    field by field, rather than stating it.
    """
    floor = variance_floor_bps2(market)
    trajectories = reference_trajectories(
        POWER_LAW_ENCODING, market, order_size, lambda_risk
    )
    return ReferenceRow(
        lambda_risk=lambda_risk,
        schedules={
            name: _signal_schedule_reference(
                name,
                trajectories[name],
                market,
                order_size,
                lambda_risk,
                floor,
                signal,
            )
            for name in SIGNAL_SCHEDULES
        },
        variance_floor=floor,
        kappa_horizon=(
            optimal_kappa(market, order_size, lambda_risk) * market.horizon_hours
        ),
        encoding=POWER_LAW_ENCODING,
    )


def signal_static_table(
    market: Market,
    order_size: float,
    signal: AlphaSignal,
    lambdas: Iterable[float] = VENDOR_LAMBDA_GRID,
) -> list[ReferenceRow]:
    """The signal world's static table over a grid — what :func:`select_lambda` reads."""
    return [
        signal_static_row(market, order_size, lam, signal) for lam in sorted(lambdas)
    ]


@dataclass(frozen=True)
class ReferenceKind:
    """What kind of confidence a reference number carries, carried **with** it.

    M5 is the first milestone whose artefact holds two references of different
    kinds at once — M4a's **certified** execution floor and the **converged**
    dynamic program — and prose is not strong enough to keep them apart. A reader
    who takes the floor for the optimum, or the optimum for a certified object,
    gets the milestone backwards in the two ways that matter: the first makes the
    agent look like it has 0.56 bps to find where it has 0.08, and the second
    claims a certificate for a number that has a Richardson residual instead.

    So each number travels with its kind, its role, and the evidence behind the
    word. ``certified`` is a boolean because it is a boolean: M4a earned it with a
    Cholesky factorisation and a KKT residual, and a stochastic dynamic program
    has no such object no matter how well it has converged.
    """

    name: str
    value_bps: float
    #: ``"certified"`` or ``"converged"``. One of exactly two words, deliberately.
    kind: str
    certified: bool
    #: What this number *is* in the milestone — the sentence that stops a reader
    #: using it as the other one.
    role: str
    #: What earns the word. Named checks, not adjectives.
    evidence: str

    def __post_init__(self) -> None:
        if self.kind not in ("certified", "converged"):
            raise ValueError(
                f"a reference kind is 'certified' or 'converged', got {self.kind!r}"
            )
        if self.certified != (self.kind == "certified"):
            raise ValueError(
                f"kind {self.kind!r} contradicts certified={self.certified}"
            )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value_bps": self.value_bps,
            "kind": self.kind,
            "certified": self.certified,
            "role": self.role,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class AlphaReferenceRow:
    """One lambda in the alpha-aware world: four schedules, a policy, and a floor.

    **The signal is not a cost encoding**, for the reason liquidity was not one:
    the functional is unchanged — the charge is still ``eta sigma p**beta`` — and
    what M5 adds is information about the *market*. :attr:`encoding` stays
    ``power_law`` and §9's *A metric grades the world that charges it* needs no
    amendment.

    **Two kinds of confidence, in one row, and it says which is which.**
    :attr:`optimal` is M4a's **certified** optimum, which is also the best
    deterministic schedule here and therefore the denominator's top rung.
    :attr:`adaptive_bps` is a dynamic program's value — **converged**, not
    certified. :attr:`execution_floor` is certified again, and it bounds only the
    half of the objective the signal cannot touch. M5 is the first milestone to
    need all three words in one place, so nothing here is called "the optimum"
    without a qualifier.

    **Why three numbers rather than one.** The optimum monetises
    :attr:`alpha_available` of signal and pays :attr:`execution_premium` of it
    back in worse impact and risk — about 45 % at the trained ``rho``. A single
    capture fraction against :attr:`signal_advantage` scores a policy that
    captures 0.15 and pays 0.07 identically to one that captures 0.25 and pays
    0.17, at a headline that is supposed to be about execution quality and reports
    neither. So the parts are first-class here, and the brief requires them
    reported together everywhere the headline appears.
    """

    lambda_risk: float
    schedules: dict[str, ScheduleReference]
    variance_floor: float
    signal: dict
    #: M4a's certified optimum on the schedule-varying part — the rigorous lower
    #: bound under ``E[impact + risk]`` for *any* policy, by convexity.
    execution_floor: float
    encoding: str = POWER_LAW_ENCODING
    #: The adaptive half, absent on a row built by :func:`signal_static_row`. The
    #: lambda rule reads only the fixed rungs, and those are closed forms and a
    #: certified Newton solve where the dynamic program is seconds to minutes.
    optimum: AlphaOptimum | None = None
    feasible: PathBound | None = None
    mean_schedule_max_bin: float | None = None

    def _optimum(self) -> AlphaOptimum:
        if self.optimum is None:
            raise ValueError(
                f"the row at lambda = {self.lambda_risk:.6e} was built without the "
                "dynamic program; build it with alpha_reference_row to read an "
                "adaptive quantity"
            )
        return self.optimum

    @property
    def twap(self) -> ScheduleReference:
        return self.schedules["twap"]

    @property
    def ac(self) -> ScheduleReference:
        return self.schedules["ac"]

    @property
    def tangent(self) -> ScheduleReference:
        """The sinh derived at the tangent — still not this world's optimum."""
        return self.schedules["tangent"]

    @property
    def optimal(self) -> ScheduleReference:
        """M4a's **certified** optimum: the best schedule that cannot see anything.

        Both what the lambda rule reads and the top of M5's denominator. Unlike
        M4b there is no gap between "the best fixed schedule that knows the law"
        and "M4a's schedule" for a level shift to hide in: a zero-mean signal
        gives a fixed schedule nothing to re-solve for, so the two rungs are one
        rung and the whole advantage is information.
        """
        return self.schedules["optimal"]

    @property
    def adaptive_bps(self) -> float:
        """``J_DP`` — the optimum over all policies that see the signal. Converged."""
        return self._optimum().objective_bps

    @property
    def twap_gap(self) -> float:
        """``(J_twap - J_M4a) / J_M4a`` — the rule's condition (i), M4a's exactly."""
        return (self.twap.objective - self.optimal.objective) / self.optimal.objective

    @property
    def adaptive_twap_gap(self) -> float:
        """The same gap read against the DP — the reading *not* used, recorded."""
        adaptive = self.adaptive_bps
        return (self.twap.objective - adaptive) / adaptive

    @property
    def signal_advantage(self) -> float:
        """``J_M4a - J_DP`` in bps — **the milestone's denominator**, and net.

        Net of the execution premium the optimum pays to collect its alpha, which
        is why it is roughly half of :attr:`alpha_available` and why reporting it
        alone would hide the more interesting half of the result.
        """
        return self.optimal.objective - self.adaptive_bps

    @property
    def advantage_fraction(self) -> float:
        """The net advantage as a fraction of ``J_M4a`` — task 0's gate 2."""
        return self.signal_advantage / self.optimal.objective

    @property
    def alpha_available(self) -> float:
        """``-E[alpha]`` at the optimum, bps — the **gross** value of the signal."""
        return -self._optimum().alpha_bps

    @property
    def execution_premium(self) -> float:
        """``E[impact + risk] - J_M4a_varying`` at the optimum, bps.

        What the optimum pays for its alpha, measured against a floor that is
        certified. Non-negative for any policy by convexity; if this ever came
        back negative the defect would be in the machinery, not in the market.
        """
        return self._optimum().execution_bps - self.execution_floor

    @property
    def premium_fraction(self) -> float:
        """The execution premium as a fraction of the gross alpha — task 0's gate 3.

        The gate that matters most. Below 25 % the decomposition is decorative and
        one headline would do; above 75 % the advantage is a small difference of
        large numbers and the milestone is a different milestone.
        """
        return self.execution_premium / self.alpha_available

    @property
    def reference_kinds(self) -> dict[str, ReferenceKind]:
        """The row's two references, each carrying the word it earned.

        Task 1's third requirement, and the reason it is a structure rather than a
        paragraph: both numbers appear in the same report, in the same units, one
        bounding the other's execution half, and only one of them has a
        certificate.
        """
        floor = ReferenceKind(
            name="execution_floor",
            value_bps=self.execution_floor,
            kind="certified",
            certified=True,
            role=(
                "M4a's optimum on the schedule-varying objective. A rigorous lower "
                "bound on E[impact + risk] for ANY policy, adapted or not, by "
                "convexity and Jensen — the signal appears in neither term. It is "
                "M5's hard red flag and it is NOT the optimum: the objective it "
                "bounds is the whole objective less alpha and less the "
                "schedule-invariant constant."
            ),
            evidence=(
                "Cholesky-PD Hessian, relative KKT residual 1.2e-15 against a 1e-12 "
                "bar, 3 600 perturbations uphill, and an independent bisection "
                "solver agreeing to 3.1e-15 of X "
                "(tests/test_power_law_certificate.py)."
            ),
        )
        optimum = ReferenceKind(
            name="adaptive_optimum",
            value_bps=self.adaptive_bps,
            kind="converged",
            certified=False,
            role=(
                "The optimum over all policies that see the signal, and the bottom "
                "rung of the milestone's denominator. What an agent is graded "
                "against. NOT certified and not bracketed: M4b could bound its DP "
                "from below with perfect information and M5 cannot, because that "
                "relaxation is three orders too loose here."
            ),
            evidence=(
                "Grid and quadrature convergence reported with a Richardson "
                "residual; (k, x_k, s_k) sufficiency measured on an augmented state "
                "carrying s_{k-1}; the decomposition identity asserted at every "
                "node; a feasible upper bound from the DP's own greedy policy; and "
                "the value returning M4a's certified number at rho = 0."
            ),
        )
        return {floor.name: floor, optimum.name: optimum}

    def as_dict(self) -> dict:
        document = {
            "lambda": self.lambda_risk,
            "encoding": self.encoding,
            "signal": self.signal,
            "twap_gap": self.twap_gap,
            "execution_floor_bps": self.execution_floor,
            "variance_floor_bps2": self.variance_floor,
            "schedules": {
                name: schedule.as_dict() for name, schedule in self.schedules.items()
            },
        }
        if self.optimum is not None:
            document |= {
                "adaptive_bps": self.adaptive_bps,
                "adaptive_twap_gap": self.adaptive_twap_gap,
                "adaptive": self.optimum.as_dict(),
                "signal_advantage_bps": self.signal_advantage,
                "advantage_fraction": self.advantage_fraction,
                "alpha_available_bps": self.alpha_available,
                "execution_premium_bps": self.execution_premium,
                "premium_fraction": self.premium_fraction,
                "feasible_upper": (
                    None if self.feasible is None else self.feasible.as_dict()
                ),
                "mean_schedule_max_bin": self.mean_schedule_max_bin,
                "reference_kinds": {
                    name: kind.as_dict()
                    for name, kind in self.reference_kinds.items()
                },
            }
        return document


def alpha_reference_row(
    market: Market,
    order_size: float,
    lambda_risk: float,
    signal: AlphaSignal,
    *,
    root_seed: int,
    stream_index: int = 0,
    paths: int = SIGNAL_BOUND_PATHS,
    grid_points: int = DEFAULT_SIGNAL_GRID_POINTS,
    quadrature_nodes: int = DEFAULT_SIGNAL_QUADRATURE_NODES,
) -> AlphaReferenceRow:
    """One row of M5's table: four fixed schedules, the DP, and a feasible bound.

    The Monte-Carlo paths come from :data:`~temper.seeding.M5_REFERENCE_POOL` —
    the oracle's own pool — for M4b's reason unchanged: spending ``eval`` streams
    on a reference table would burn addresses a trained result is reported at, on
    a computation with no agent in it.

    Only *one* bound is sampled here, and the asymmetry is the milestone's finding
    rather than an omission. The feasible upper bound is a statement about a
    policy and survives; the perfect-information lower bound is a statement about
    *price* information and is three orders too loose to say anything
    (:func:`~temper.oracle.alpha.clairvoyant_price_values` computes it anyway, in
    task 0, because retiring an inherited test needs evidence). What replaces it
    is :attr:`AlphaReferenceRow.execution_floor`, which is certified and needs no
    sampling at all.
    """
    static = signal_static_row(market, order_size, lambda_risk, signal)
    optimum = alpha_optimum(
        market,
        order_size,
        lambda_risk,
        signal,
        points=grid_points,
        nodes=quadrature_nodes,
    )

    rng = pool_rng(root_seed, M5_REFERENCE_POOL, stream_index)
    signals = signal.draw(rng, (paths, market.n_bins))
    reference_weights = (
        trades(static.schedules["optimal"].trajectory, market) / order_size
    )
    reference_cost = signal_path_objective_bps(
        reference_weights, signals, market, order_size, lambda_risk, signal
    )
    greedy = optimum.greedy_weights(signals)
    paired = _paired_bound(
        "feasible_upper",
        signal_path_objective_bps(
            greedy, signals, market, order_size, lambda_risk, signal
        ),
        reference_cost,
        static.schedules["optimal"].objective,
    )
    # `_paired_bound` records what it was paired against, and here that is M4a's
    # certified optimum rather than M4b's static one: in this world they are the
    # same schedule, and the certified name is the honest one to write down.
    feasible = PathBound(
        name=paired.name,
        value_bps=paired.value_bps,
        half_width_bps=paired.half_width_bps,
        paired_sd_bps=paired.paired_sd_bps,
        unpaired_sd_bps=paired.unpaired_sd_bps,
        paths=paired.paths,
        paired_against="optimal",
    )

    return AlphaReferenceRow(
        lambda_risk=lambda_risk,
        schedules=static.schedules,
        variance_floor=static.variance_floor,
        signal=signal.as_dict(),
        execution_floor=execution_floor_bps(market, order_size, lambda_risk),
        optimum=optimum,
        feasible=feasible,
        mean_schedule_max_bin=float(np.max(greedy.mean(axis=0))),
    )


def alpha_reference_table(
    market: Market,
    order_size: float,
    signal: AlphaSignal,
    lambdas: Iterable[float] = VENDOR_LAMBDA_GRID,
    *,
    root_seed: int,
    paths: int = SIGNAL_BOUND_PATHS,
    grid_points: int = DEFAULT_SIGNAL_GRID_POINTS,
    quadrature_nodes: int = DEFAULT_SIGNAL_QUADRATURE_NODES,
) -> list[AlphaReferenceRow]:
    """The whole alpha table, ascending in lambda. M0's 17-point grid by default.

    Each row gets its **own** stream index, so adding a lambda cannot move the
    signal paths an already-committed row was measured on.
    """
    return [
        alpha_reference_row(
            market,
            order_size,
            lam,
            signal,
            root_seed=root_seed,
            stream_index=index,
            paths=paths,
            grid_points=grid_points,
            quadrature_nodes=quadrature_nodes,
        )
        for index, lam in enumerate(sorted(lambdas))
    ]
