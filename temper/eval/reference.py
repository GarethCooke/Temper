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
    VENDOR_LAMBDA_GRID,
    Market,
    ac_trajectory,
    objective_curvature_floor,
    optimal_kappa,
    optimal_trajectory,
    schedule_moments,
    trades,
    twap_trajectory,
)

#: The schedules every table and figure carries (constitution invariant 4).
#: Order is the reporting order and is part of the committed contract.
REFERENCE_SCHEDULES: tuple[str, ...] = ("twap", "ac", "optimal")


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
    """The three schedules at one lambda, plus what the selection rule reads."""

    lambda_risk: float
    schedules: dict[str, ScheduleReference]
    variance_floor: float
    kappa_horizon: float

    @property
    def twap(self) -> ScheduleReference:
        return self.schedules["twap"]

    @property
    def ac(self) -> ScheduleReference:
        return self.schedules["ac"]

    @property
    def optimal(self) -> ScheduleReference:
        return self.schedules["optimal"]

    @property
    def twap_gap(self) -> float:
        """``(J_twap - J_optimal) / J_optimal`` — how discriminative the testbed is.

        If this is small then "the agent got within epsilon of optimal" is a
        claim TWAP also satisfies, and the milestone measures nothing. Condition
        (i) of the selection rule is a floor under it.
        """
        return (self.twap.objective - self.optimal.objective) / self.optimal.objective

    def as_dict(self) -> dict:
        return {
            "lambda": self.lambda_risk,
            "twap_gap": self.twap_gap,
            "kappa_horizon": self.kappa_horizon,
            "variance_floor_bps2": self.variance_floor,
            "schedules": {
                name: schedule.as_dict() for name, schedule in self.schedules.items()
            },
        }


def _schedule_reference(
    name: str,
    trajectory: np.ndarray,
    market: Market,
    order_size: float,
    lambda_risk: float,
    floor: float,
) -> ScheduleReference:
    moments = schedule_moments(trajectory, market, order_size=order_size)
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


def reference_row(
    market: Market, order_size: float, lambda_risk: float
) -> ReferenceRow:
    """TWAP, the vendored AC schedule and the exact optimum at one lambda."""
    floor = variance_floor_bps2(market)
    trajectories = {
        "twap": twap_trajectory(market, order_size),
        "ac": ac_trajectory(market, order_size, lambda_risk),
        "optimal": optimal_trajectory(market, order_size, lambda_risk),
    }
    return ReferenceRow(
        lambda_risk=lambda_risk,
        schedules={
            name: _schedule_reference(
                name, trajectories[name], market, order_size, lambda_risk, floor
            )
            for name in REFERENCE_SCHEDULES
        },
        variance_floor=floor,
        kappa_horizon=(
            optimal_kappa(market, order_size, lambda_risk) * market.horizon_hours
        ),
    )


def reference_table(
    market: Market,
    order_size: float,
    lambdas: Iterable[float] = VENDOR_LAMBDA_GRID,
) -> list[ReferenceRow]:
    """The whole table, ascending in lambda. M0's 17-point grid by default."""
    return [reference_row(market, order_size, lam) for lam in sorted(lambdas)]


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
    """

    delta_objective: float   # the objective excess allowed, bps
    curvature_floor: float   # lambda_min(H), bps per share^2
    bound_shares: float      # the implied |d|_2 bound, shares
    order_size: float        # the parent order the bound is a fraction of

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
        }


def trajectory_band(
    market: Market,
    order_size: float,
    lambda_risk: float,
    delta_objective: float,
) -> TrajectoryBand:
    """The band an objective excess of `delta_objective` bps implies."""
    if delta_objective < 0.0:
        raise ValueError(
            f"delta_objective is an objective excess and must be >= 0, "
            f"got {delta_objective}"
        )
    floor = objective_curvature_floor(market, order_size, lambda_risk)
    return TrajectoryBand(
        delta_objective=delta_objective,
        curvature_floor=floor,
        bound_shares=math.sqrt(2.0 * delta_objective / floor),
        order_size=order_size,
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
