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

**Amended by M4b, which is the first milestone that cannot obey the rule above as
written.** A liquidity-observing policy's schedule is closed-loop by design, so
there is no single trajectory to hand to a closed form and the open-loop shortcut
retires. The *assertion* does not: the price still enters cost only through M1a's
affine term and the policy still never sees a price, so conditioning on the
liquidity path removes all of the price randomness analytically and
``E[cost | L]`` is a closed form. :func:`deterministic_schedule` keeps its name
and its exception and grows one axis — pin the liquidity, vary the price, require
the trajectories bitwise equal — and :mod:`temper.eval.conditional` is where the
averaging over liquidity paths lives. Everything in *this* module continues to
serve the deterministic-schedule worlds, M0 through M4a, unchanged.

The number itself comes out of :data:`~temper.eval.metrics.GRADED`, not out of a
direct call to the closed form. That registry refuses to admit a metric encoded
against FrontierView's power law (invariant 7, ``ARCHITECTURE.md`` §9), and
routing M2's grade through it is what makes the refusal load-bearing instead of a
property of a module nothing on this path imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from temper.env import (
    DETERMINISTIC_LIQUIDITY,
    NO_SIGNAL_STREAM,
    ExecutionEnv,
    LiquidityStream,
    SignalStream,
    TemporaryImpact,
    impact_for,
)
from temper.eval.metrics import Metric, WorldMismatch, check_grades_world, metrics_for
from temper.eval.reference import ReferenceRow, trajectory_deviation
from temper.eval.rollout import run_episode
from temper.oracle import Market

#: The registry entries a grade is assembled from. Routed through
#: :func:`~temper.eval.metrics.metrics_for` rather than calling
#: :func:`~temper.oracle.cost.schedule_moments` directly, which is not
#: indirection for its own sake: the registry is keyed by *world*, so asking it
#: for the metrics that charge this env's encoding is what makes M4a's rule — a
#: metric grades the world that charges it — load-bearing on the grading path
#: rather than a fact about a module nobody here imports.
#: ``tests/test_m2_grading.py`` pins that the names below are registered and
#: agree with the closed form in the linearised world;
#: ``tests/test_objective_registry.py`` does it in both.
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
    temporary_impact: TemporaryImpact | None = None,
    liquidity: LiquidityStream | None = None,
    signal: SignalStream | None = None,
    expect_encoding: str | None = None,
) -> np.ndarray:
    """The inventory trajectory `policy` induces, verified **price**-independent.

    Rolls the policy out once per price stream through the shared
    :func:`~temper.eval.rollout.run_episode` — the same loop TWAP and both AC
    schedules go through — and requires every trajectory to be *bitwise* equal.
    Bitwise, not ``allclose``: the claim is that the shocks did not enter the
    computation at all, and a policy that let 1e-16 of price into its action
    would still be a policy whose schedule is not open-loop.

    **Generalised by M4b, not retired.** Through M4a the claim was "the schedule
    is open-loop", because the observation carried nothing that varied. A
    liquidity-observing policy's schedule is *not* open-loop — it reacts, on
    purpose, and that is the milestone — so the check grows one axis instead:
    the liquidity stream is **pinned** and only the price stream varies. What is
    asserted is therefore the narrower and still load-bearing half of the old
    claim, "the price never entered the decision", which is exactly what makes
    ``E[cost | L]`` a legitimate closed form
    (:mod:`temper.eval.conditional`). A policy that fails this has still not been
    scored badly — it has not been scored at all.

    The pin is what makes the check possible: liquidity normally follows the env's
    stream index, so varying the index would move both noise sources at once and
    the comparison would be vacuous. `liquidity` is pinned to `streams[0]`'s index
    here, so the two rollouts differ in the price and in nothing else.

    **Generalised again by M5, on the same axis and for a sharper reason.** The
    signal stream is pinned too, so what is asserted is still "the price never
    entered the decision" — now at a fixed liquidity path *and* a fixed signal
    path. Without the pin this check would be worse than vacuous in a
    signal-bearing world: an unpinned signal moves with the stream index, so the
    trajectories would differ and a correct policy would be refused; and an env
    built with no signal at all would pass trivially while rolling the policy out
    in the wrong world, which is the M4a mirror bug at the grading path. The
    caller names the signal for the same reason it names the impact model.

    `temporary_impact` names the world the policy is rolled out in; ``None`` is
    Phase 1, which is what the env itself defaults to. `liquidity` likewise
    defaults to the deterministic multiplier, which is the market M0-M4a ran in.
    `expect_encoding` is the env half of M4a's world/metric check: the caller
    states which world it believes it is grading in, and every env built here has
    to agree before a step is taken.
    """
    if len(streams) < 2:
        raise ValueError(
            f"determinism needs at least two shock streams, got {list(streams)}"
        )
    stream_pin = (
        DETERMINISTIC_LIQUIDITY if liquidity is None else liquidity
    ).pinned_to(int(streams[0]))
    signal_pin = (
        NO_SIGNAL_STREAM if signal is None else signal
    ).pinned_to(int(streams[0]))

    trajectories = []
    for stream in streams:
        env = ExecutionEnv(
            market,
            order_size,
            lambda_risk,
            temporary_impact=temporary_impact,
            liquidity=stream_pin,
            signal=signal_pin,
            root_seed=root_seed,
            pool=pool,
            stream_index=int(stream),
        )
        if expect_encoding is not None and env.cost_encoding != expect_encoding:
            raise WorldMismatch(
                f"the eval env charges the {env.cost_encoding!r} encoding but "
                f"the grade would be computed against a {expect_encoding!r} "
                "reference; the schedule and the optimum it is scored against "
                "must come from one world"
            )
        trajectories.append(run_episode(env, policy).trajectory)

    reference = trajectories[0]
    for stream, trajectory in zip(streams[1:], trajectories[1:]):
        if not np.array_equal(trajectory, reference):
            worst = float(np.max(np.abs(trajectory - reference)))
            raise ScheduleNotDeterministic(
                f"policy {getattr(policy, 'name', policy)!r} realised different "
                f"schedules on price streams {streams[0]} and {stream} at one "
                f"pinned liquidity path and one pinned signal path — worst "
                f"difference {worst:.3e} shares. The observation may carry a "
                "prediction of a shock that has not been committed; it may never "
                "carry the realised price, or neither analytic grading nor the "
                "conditional expectations E[cost | L] and E[cost | s] are valid."
            )
    return reference


@dataclass(frozen=True)
class Grade:
    """One schedule, scored against the certified optimum of one world.

    Two normalisations of the same excess, because the two milestones they serve
    ask different questions, and reporting a fraction without saying which
    denominator it has is the trap ``ARCHITECTURE.md`` §9 records twice.

    ``gap_fraction`` is the excess over the optimum as a fraction of the excess
    TWAP already carries. Zero is the optimum, one is TWAP; M2's and M3's epsilon
    is 0.05 of it. Portable across lambda by construction, and degenerate
    wherever TWAP and the optimum have nearly converged.

    ``advantage_fraction`` is the excess as a fraction of the *available
    advantage* — what the closed form leaves on the table in a world it was not
    derived for. It exists only where there is such a world (M4a), is ``None``
    elsewhere, and its complement :attr:`capture_fraction` is the number M4a
    leads with. It is not interchangeable with ``gap_fraction``: at M4a's lambda
    5 % of the TWAP gap is 1.8-2.0x the *entire* available advantage, so an agent
    graded to the first bar could capture none of the mis-specification and pass.

    The absolute :attr:`excess` in bps travels beside both of them everywhere,
    for the reason §9 gives: a fraction alone made a healthy agent look like a
    degrading one at low lambda, and the same trap runs the other way here — a
    capture fraction near 1 on an advantage of 0.037 bps is a small absolute
    claim and should read as one.
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
    encoding: str            # the world this was charged and graded in
    #: ``(J - J_optimal) / (J_tangent - J_optimal)``, where a tangent-derived
    #: closed form exists to be beaten; ``None`` in the linearised world.
    advantage_fraction: float | None = None

    @property
    def capture_fraction(self) -> float | None:
        """``(J_tangent - J_agent) / (J_tangent - J_optimal)`` — M4a's headline.

        One is the certified power-law optimum, zero is the Almgren-Chriss
        schedule the vendored library would have run, and negative is an agent
        that did worse than the closed form it was supposed to improve on.
        """
        if self.advantage_fraction is None:
            return None
        return 1.0 - self.advantage_fraction

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "encoding": self.encoding,
            "expected_bps": self.expected,
            "variance_bps2": self.variance,
            "objective_bps": self.objective,
            "excess_bps": self.excess,
            "relative_excess": self.relative_excess,
            "gap_fraction": self.gap_fraction,
            "advantage_fraction": self.advantage_fraction,
            "capture_fraction": self.capture_fraction,
            "deviation_shares": self.deviation,
            "red_flag": self.red_flag,
            "trajectory": [float(x) for x in self.trajectory],
        }


def graded_metrics(reference: ReferenceRow) -> Mapping[str, Metric]:
    """The metrics that may score `reference`'s world, checked before use.

    Two refusals, not one. :func:`~temper.eval.metrics.metrics_for` will not hand
    back metrics for a world it does not have; :func:`check_grades_world` then
    re-reads every one of their declared encodings against the reference's. The
    second looks redundant against a registry that is keyed by encoding, and is
    kept because it is the assertion that survives someone assembling a mapping
    by hand — which is how a linear metric would find its way onto a power-law
    env, the failure the flat quarantine could not have caught.
    """
    metrics = metrics_for(reference.encoding)
    check_grades_world(reference.encoding, metrics)
    return metrics


def grade_trajectory(
    trajectory,
    market: Market,
    order_size: float,
    reference: ReferenceRow,
    *,
    name: str = "agent",
) -> Grade:
    """Score a deterministic schedule against `reference`'s certified optimum.

    `reference` carries the lambda *and the world*, so there is no way to grade
    at one lambda against an optimum computed at another, or in one world against
    an optimum solved in the other — the three travel together (invariant 7).
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

    metrics = graded_metrics(reference)
    lambda_risk = reference.lambda_risk
    objective = metrics[OBJECTIVE](x, market, lambda_risk)

    optimum = reference.optimal
    excess = objective - optimum.objective
    relative = excess / optimum.objective
    advantage = reference.available_advantage
    return Grade(
        name=name,
        trajectory=x,
        expected=metrics[EXPECTED_COST](x, market, lambda_risk),
        variance=metrics[SHORTFALL_VARIANCE](x, market, lambda_risk),
        objective=objective,
        excess=excess,
        relative_excess=relative,
        gap_fraction=relative / reference.twap_gap,
        deviation=trajectory_deviation(x, optimum.trajectory),
        red_flag=bool(excess < -RED_FLAG_RTOL * abs(optimum.objective)),
        encoding=reference.encoding,
        advantage_fraction=None if advantage is None else excess / advantage,
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
    """Roll a policy out deterministically, then score the schedule it induced.

    The world comes off the reference and nowhere else: the env the policy is
    rolled out in and the optimum it is scored against are built from one string,
    so "the agent was graded in the world it was evaluated in" is a property of
    the call rather than of the caller's care.
    """
    trajectory = deterministic_schedule(
        policy,
        market,
        order_size,
        reference.lambda_risk,
        root_seed=root_seed,
        pool=pool,
        streams=streams,
        temporary_impact=impact_for(reference.encoding, market, order_size),
        expect_encoding=reference.encoding,
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


def summarise(
    name: str, values: Sequence[float], *, direction: str = "cost"
) -> SeedSummary:
    """Median, quartiles and the worst seed.

    Through M4b every quantity summarised here — objective excess, gap fraction,
    trajectory deviation — was a cost, so larger was worse without exception and
    ``worst`` could be ``max`` rather than a per-metric direction that would
    eventually get one of them backwards.

    "Eventually" was M5. It reports *captures*: fractions of an available
    advantage, where larger is BETTER. Summarised as costs their ``worst`` is the
    best seed, and M5's first sweep shipped exactly that, twice::

        alpha_capture   reported worst 1.1099, true worst 0.8959
        net_capture     reported worst 0.9559, true worst 0.8925

    So the direction is a parameter now rather than a comment. ``cost`` keeps the
    old behaviour and stays the default, so every call written before M5 still
    means what it meant; ``benefit`` takes the minimum instead. The direction
    belongs to the *reporting*, not to the number: the shuffled control
    summarises ``net_capture`` as a COST, because a control that captures alpha
    is the alarming outcome there.
    """
    if direction not in ("cost", "benefit"):
        raise ValueError(
            f"direction must be 'cost' or 'benefit', not {direction!r}"
        )
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
        worst=float(np.max(array) if direction == "cost" else np.min(array)),
    )


def median_ordinal(values: Sequence[float]) -> int:
    """Which seed *is* the median — the index of the upper central rank.

    :func:`summarise` reports the median as a number, and at an even seed count
    that number belongs to no seed: M4a's ten seeds have a median objective
    exactly halfway between two of them, and the two are 1e-17 apart in their
    distance from it, so "the seed nearest the median" is decided by float noise
    rather than by a rule. This picks by *rank* instead — sort ascending, take
    index ``n // 2`` — which is numpy's own upper-median position and is the
    **worse** of the two central seeds at even ``n``.

    That direction is the point. Anything selected out of a sweep and then
    shipped is a choice that could flatter the artefact, and the cheapest
    defence is a tie-break that can only ever cost: the exported policy is at or
    below the sweep's median, never above it. Every quantity this function is
    asked to rank is a cost (M5's captures are reported, not selected on),
    so ascending order is best-to-worst and no direction is needed here.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("no seeds to take a median of")
    return int(np.argsort(array, kind="stable")[array.size // 2])
