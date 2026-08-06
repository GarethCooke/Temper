"""Grade a policy analytically, through the oracle, with no Monte-Carlo error.

M1 left M2 a fact worth more than any variance-reduction trick: the observation
is ``(time remaining, inventory remaining)`` and carries no price, and inventory
evolves purely from actions. A *deterministic* policy therefore induces the same
inventory trajectory under every shock stream — an open-loop schedule — and an
open-loop schedule's exact moments are a closed form
(:func:`~temper.oracle.cost.schedule_moments`). One rollout, and the objective is
known exactly.

The alternative is worth stating because it is the obvious thing to do and it is
hopeless. At the frontier case the objective is ~2.4 bps while the per-episode
cost standard deviation is ~95 bps; resolving the milestone's tolerance by
sampling realised costs would need on the order of 10^7 episodes *per seed*, and
would still report a confidence interval where this reports a number.

So: **grade the eval schedule analytically; never estimate the agent's objective
by sampling.** :func:`deterministic_schedule` is where that claim is enforced
rather than assumed — it rolls the policy out on two unrelated shock streams and
requires the trajectories to be bitwise identical. If price ever leaks into the
observation, that assertion fails loudly and everything downstream stops being
computed rather than quietly becoming an estimate.

The number itself comes out of :data:`~temper.eval.metrics.GRADED`, not out of a
direct call to the closed form. That registry refuses to admit a metric encoded
against FrontierView's power law (invariant 7, ``ARCHITECTURE.md`` §9), and
routing M2's grade through it is what makes the refusal load-bearing instead of a
property of a module nothing on this path imports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from temper.env import ExecutionEnv
from temper.eval.metrics import GRADED
from temper.eval.reference import ReferenceRow, trajectory_deviation
from temper.eval.rollout import run_episode
from temper.oracle import Market

#: The registry entries a grade is assembled from. Routed through
#: :data:`~temper.eval.metrics.GRADED` rather than calling
#: :func:`~temper.oracle.cost.schedule_moments` directly, which is not
#: indirection for its own sake: ``register_graded`` refuses a metric encoded
#: against FrontierView's power law, so going through the registry is what makes
#: invariant 7's quarantine load-bearing for M2 rather than a fact about a module
#: nobody on this path imports. ``tests/test_m2_grading.py`` pins that the names
#: below are registered, are ``LINEAR``, and agree with the closed form.
OBJECTIVE = "objective"
EXPECTED_COST = "expected_cost"
SHORTFALL_VARIANCE = "shortfall_variance"

#: Float tolerance on the red-flag test, relative to ``|J_optimal|``. The
#: optimum is certified (M1 task 0: Cholesky PD, generic solve matching the
#: closed form to 1.4e-15 of X, 3 600 perturbations uphill), and §4 states the
#: optimum over adapted policies *is* that deterministic trajectory. So a
#: strictly lower objective is not a better agent, it is a defect — in the
#: metric, the env, or the grading path — and the only slack it gets is
#: arithmetic noise.
RED_FLAG_RTOL = 1e-9

#: The eval streams a schedule's determinism is checked across. Two unrelated
#: streams of the ``eval`` pool; any two would do, and these two are committed so
#: the check is regenerable (invariant 1).
DEFAULT_EVAL_STREAMS: tuple[int, ...] = (0, 1)


class ScheduleNotDeterministic(AssertionError):
    """A policy realised different schedules under different shock streams.

    Raised rather than returned. Analytic grading is only valid for an open-loop
    schedule, so a policy that fails this has not scored badly — it has not been
    scored at all, and every number downstream of it would be a category error.
    """


def deterministic_schedule(
    policy,
    market: Market,
    order_size: float,
    lambda_risk: float,
    *,
    root_seed: int,
    pool: str = "eval",
    streams: Sequence[int] = DEFAULT_EVAL_STREAMS,
) -> np.ndarray:
    """The inventory trajectory `policy` induces, verified shock-independent.

    Rolls the policy out once per stream through the shared
    :func:`~temper.eval.rollout.run_episode` — the same loop TWAP and both AC
    schedules go through — and requires every trajectory to be *bitwise* equal.
    Bitwise, not ``allclose``: the claim is that the shocks did not enter the
    computation at all, and a policy that let 1e-16 of price into its action
    would still be a policy whose schedule is not open-loop.
    """
    if len(streams) < 2:
        raise ValueError(
            f"determinism needs at least two shock streams, got {list(streams)}"
        )

    trajectories = []
    for stream in streams:
        env = ExecutionEnv(
            market,
            order_size,
            lambda_risk,
            root_seed=root_seed,
            pool=pool,
            stream_index=int(stream),
        )
        trajectories.append(run_episode(env, policy).trajectory)

    reference = trajectories[0]
    for stream, trajectory in zip(streams[1:], trajectories[1:]):
        if not np.array_equal(trajectory, reference):
            worst = float(np.max(np.abs(trajectory - reference)))
            raise ScheduleNotDeterministic(
                f"policy {getattr(policy, 'name', policy)!r} realised different "
                f"schedules on eval streams {streams[0]} and {stream} — worst "
                f"difference {worst:.3e} shares. The observation must carry no "
                "price, or analytic grading is invalid."
            )
    return reference


@dataclass(frozen=True)
class Grade:
    """One schedule, scored against the certified optimum.

    ``gap_fraction`` is the number the milestone's tolerance is stated in: the
    excess over the optimum, as a fraction of the excess TWAP already carries.
    Zero is the optimum, one is TWAP, and the pre-stated epsilon is 0.05.
    Expressing it this way rather than in bps is what makes the tolerance mean
    the same thing at every lambda when M3 sweeps.
    """

    name: str
    trajectory: np.ndarray
    expected: float          # E[cost], bps
    variance: float          # V[cost], bps^2
    objective: float         # E + lambda * V, bps
    excess: float            # J - J_optimal, bps
    relative_excess: float   # (J - J_optimal) / J_optimal
    gap_fraction: float      # relative_excess / (TWAP's relative excess)
    deviation: float         # |x - x*|_2 over the interior holdings, shares
    red_flag: bool           # J below the certified optimum beyond float slack

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "expected_bps": self.expected,
            "variance_bps2": self.variance,
            "objective_bps": self.objective,
            "excess_bps": self.excess,
            "relative_excess": self.relative_excess,
            "gap_fraction": self.gap_fraction,
            "deviation_shares": self.deviation,
            "red_flag": self.red_flag,
            "trajectory": [float(x) for x in self.trajectory],
        }


def grade_trajectory(
    trajectory,
    market: Market,
    order_size: float,
    reference: ReferenceRow,
    *,
    name: str = "agent",
) -> Grade:
    """Score a deterministic schedule against `reference`'s certified optimum.

    `reference` carries the lambda, so there is no way to grade at one lambda
    against an optimum computed at another — the pair travels together
    (invariant 7).
    """
    x = np.asarray(trajectory, dtype=float)
    # The registry's metrics take the schedule alone and read the parent size off
    # its first point, which is the same number for anything the env produced.
    # Checked rather than assumed, because a trajectory that did not start at X
    # would silently be graded against a different tangent `eta_tilde`.
    if abs(float(x[0]) - order_size) > 1e-9 * order_size:
        raise ValueError(
            f"trajectory starts at {x[0]:.6g} shares, not the parent order's "
            f"{order_size:.6g}; the graded metrics linearise at the parent size"
        )

    lambda_risk = reference.lambda_risk
    objective = GRADED[OBJECTIVE](x, market, lambda_risk)

    optimum = reference.optimal
    excess = objective - optimum.objective
    relative = excess / optimum.objective
    return Grade(
        name=name,
        trajectory=x,
        expected=GRADED[EXPECTED_COST](x, market, lambda_risk),
        variance=GRADED[SHORTFALL_VARIANCE](x, market, lambda_risk),
        objective=objective,
        excess=excess,
        relative_excess=relative,
        gap_fraction=relative / reference.twap_gap,
        deviation=trajectory_deviation(x, optimum.trajectory),
        red_flag=bool(excess < -RED_FLAG_RTOL * abs(optimum.objective)),
    )


def grade_policy(
    policy,
    market: Market,
    order_size: float,
    reference: ReferenceRow,
    *,
    root_seed: int,
    pool: str = "eval",
    streams: Sequence[int] = DEFAULT_EVAL_STREAMS,
    name: str | None = None,
) -> Grade:
    """Roll a policy out deterministically, then score the schedule it induced."""
    trajectory = deterministic_schedule(
        policy,
        market,
        order_size,
        reference.lambda_risk,
        root_seed=root_seed,
        pool=pool,
        streams=streams,
    )
    return grade_trajectory(
        trajectory,
        market,
        order_size,
        reference,
        name=name if name is not None else getattr(policy, "name", "policy"),
    )


# ---------------------------------------------------------------------------
# Across seeds: median and IQR, never a single run (constitution invariant 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedSummary:
    """Median and inter-quartile range of one quantity across training seeds."""

    name: str
    values: tuple[float, ...]
    median: float
    q1: float
    q3: float
    worst: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "values": list(self.values),
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "worst": self.worst,
        }


def summarise(name: str, values: Sequence[float]) -> SeedSummary:
    """Median, quartiles and the worst seed. ``worst`` is the largest value.

    Every quantity summarised here — objective excess, gap fraction, trajectory
    deviation — is a cost, so larger is worse without exception and ``worst`` can
    be ``max`` rather than a per-metric direction that would eventually get one
    of them backwards.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError(f"nothing to summarise for {name!r}")
    q1, median, q3 = (float(v) for v in np.percentile(array, [25.0, 50.0, 75.0]))
    return SeedSummary(
        name=name,
        values=tuple(float(v) for v in array),
        median=median,
        q1=q1,
        q3=q3,
        worst=float(np.max(array)),
    )
