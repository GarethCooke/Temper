"""Two registries, both keyed by world: what may be graded, and what is context.

**A metric grades the world that charges it.** That is M4a's rule and it is what
these registries encode. :data:`GRADED` maps an *encoding* to the metrics that
may score an env charging that encoding; :func:`metrics_for` refuses to hand back
anything else, and :mod:`temper.eval.grading` calls it before it computes a
number.

What this replaced, and why the replacement is stronger
-------------------------------------------------------
Until M4a the rule was ``GRADEABLE_ENCODINGS = {LINEAR}``: a flat refusal of
FrontierView's power-law charge, which was the right rule stated in the only way
that was checkable when one world existed. Phase 1 is the linearised world end to
end — env dynamics, training reward, evaluation metric and oracle all speak
:func:`~temper.oracle.cost.schedule_moments` — because M1 measured the two
encodings differing by 12 %–54 % of expected cost, so grading against the other
one would have broken invariant 7 outright (``ARCHITECTURE.md`` §9,
*Phase 1 is the linearised world end-to-end*).

M4a makes the power law a world rather than a note, so a blanket ban would have
to be bypassed — and a bypassed check is not a check. The rule generalises
instead. It is strictly stronger than what it replaced, for a reason worth being
concrete about: the flat rule could not have caught a **linear metric grading a
power-law env**, because linear was permitted to everything equally. That is now
the live failure mode, and it is the one :func:`metrics_for` exists to refuse.
The same shape as M2's replacement of the flat seed-pool ban with a per-module
allow-list (``ARCHITECTURE.md`` §9, *Invariant 5 is enforced per module*).

:data:`CONTEXT` keeps its job unchanged: a metric that may sit *beside* a claim
and never be one. FrontierView's power-law charge lives there under the names
M1, M2 and M3 report it by, which is what let a Phase-1 result quote the vendored
number without ever being graded on it.

``tests/test_objective_registry.py`` checks the labels are true rather than
merely honest — every graded metric evaluated against *both* encodings on the
golden cases, in both worlds, and a world/metric mismatch refused by
construction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from temper.oracle import (
    ENCODINGS,
    LINEAR_ENCODING,
    POWER_LAW_ENCODING,
    Market,
    cost_moments,
    schedule_moments,
)

#: The frozen Phase-1 objective: linear temporary impact at the tangent eta_tilde.
LINEAR = LINEAR_ENCODING

#: FrontierView's power-law charge. Reporting context in Phase 1; M4a's world.
POWER_LAW = POWER_LAW_ENCODING


class WorldMismatch(AssertionError):
    """A metric was asked to grade a world it does not charge.

    An ``AssertionError`` rather than a ``ValueError``, and raised rather than
    returned, for the same reason
    :class:`~temper.eval.grading.ScheduleNotDeterministic` is: a schedule scored
    by the wrong functional has not scored badly, it has not been scored at all,
    and every number downstream of it would be a category error.
    """


@dataclass(frozen=True)
class Metric:
    """A named scalar over a deterministic schedule, and how it charges cost."""

    name: str
    encoding: str
    summary: str
    fn: Callable[[np.ndarray, Market, float], float]

    def __post_init__(self) -> None:
        if self.encoding not in ENCODINGS:
            raise ValueError(
                f"unknown cost encoding {self.encoding!r}; "
                f"expected one of {', '.join(ENCODINGS)}"
            )

    def __call__(self, trajectory, market: Market, lambda_risk: float) -> float:
        return self.fn(trajectory, market, lambda_risk)


#: Metrics a reported claim may be made on, by the encoding they charge.
GRADED: dict[str, dict[str, Metric]] = {encoding: {} for encoding in ENCODINGS}

#: Metrics that may only ever appear beside a claim, by the encoding they charge.
CONTEXT: dict[str, dict[str, Metric]] = {encoding: {} for encoding in ENCODINGS}


def _register(registry: dict[str, dict[str, Metric]], metric: Metric) -> Metric:
    world = registry[metric.encoding]
    if metric.name in world:
        raise ValueError(
            f"metric {metric.name!r} is already registered for the "
            f"{metric.encoding!r} encoding"
        )
    world[metric.name] = metric
    return metric


def register_graded(metric: Metric) -> Metric:
    """Register a metric a claim may be graded on, under the world it charges."""
    return _register(GRADED, metric)


def register_context(metric: Metric) -> Metric:
    """Register a metric that may be reported alongside a claim, never as one."""
    return _register(CONTEXT, metric)


def metrics_for(encoding: str) -> Mapping[str, Metric]:
    """The graded metrics that charge `encoding`, or a refusal.

    The one function the grading path goes through, so "the metric grades the
    world that charges it" is arithmetic rather than a convention: pass an env's
    :attr:`~temper.env.ExecutionEnv.cost_encoding` and what comes back can only
    score that env.
    """
    if encoding not in GRADED:
        raise WorldMismatch(
            f"no graded metrics for the {encoding!r} encoding; known worlds are "
            f"{', '.join(sorted(GRADED))}"
        )
    world = GRADED[encoding]
    if not world:
        raise WorldMismatch(
            f"the {encoding!r} encoding has no graded metrics registered; a claim "
            "cannot be made in a world nothing scores"
        )
    return world


def check_grades_world(encoding: str, metrics: Mapping[str, Metric]) -> None:
    """Refuse `metrics` unless every one of them charges `encoding`.

    Belt and braces over :func:`metrics_for`, and the check the brief names: the
    grader asserts ``metric.encoding == env.cost_encoding`` *before* it computes
    anything, so a caller that assembled its own metric mapping is held to the
    same rule as one that asked the registry.
    """
    mismatched = sorted(
        f"{name} ({metric.encoding})"
        for name, metric in metrics.items()
        if metric.encoding != encoding
    )
    if mismatched:
        raise WorldMismatch(
            f"the env charges the {encoding!r} encoding but would be graded by "
            f"{', '.join(mismatched)}. A metric grades the world that charges it: "
            "an agent scored on a functional its env does not pay out is not a "
            "worse agent, it is not a measurement."
        )


def _registering(register, name: str, encoding: str, summary: str):
    def decorate(fn: Callable[[np.ndarray, Market, float], float]):
        register(Metric(name=name, encoding=encoding, summary=summary, fn=fn))
        return fn

    return decorate


def graded(name: str, encoding: str, summary: str):
    """Decorator form of :func:`register_graded`."""
    return _registering(register_graded, name, encoding, summary)


def context(name: str, encoding: str, summary: str):
    """Decorator form of :func:`register_context`."""
    return _registering(register_context, name, encoding, summary)


#: The three metrics every world must provide, and the names the grading path
#: looks them up by. Registering a world without all three is a
#: :class:`WorldMismatch` waiting to happen at grade time rather than at import
#: time, so ``tests/test_objective_registry.py`` checks the sets match.
CORE_METRICS: tuple[str, ...] = ("objective", "expected_cost", "shortfall_variance")


# ---------------------------------------------------------------------------
# Graded — the frozen objective and its two halves, in the linearised world
# ---------------------------------------------------------------------------


@graded("objective", LINEAR, "E[cost] + lambda * V[cost], the frozen objective")
def objective(trajectory, market: Market, lambda_risk: float) -> float:
    """What a Phase-1 env's summed reward is the negative of."""
    return schedule_moments(trajectory, market).objective(lambda_risk)


@graded("expected_cost", LINEAR, "E[cost] in bps of notional")
def expected_cost(trajectory, market: Market, lambda_risk: float) -> float:
    """Mean implementation shortfall of the schedule, at the tangent eta_tilde."""
    return schedule_moments(trajectory, market).expected


@graded("shortfall_variance", LINEAR, "V[cost] in bps^2")
def shortfall_variance(trajectory, market: Market, lambda_risk: float) -> float:
    """Variance of implementation shortfall — identical under both encodings."""
    return schedule_moments(trajectory, market).variance


# ---------------------------------------------------------------------------
# Graded — the same three, in the power-law world (M4a)
# ---------------------------------------------------------------------------


@graded("objective", POWER_LAW, "E[cost] + lambda * V[cost] under the power law")
def power_law_world_objective(trajectory, market: Market, lambda_risk: float) -> float:
    """What an M4a env's summed reward is the negative of.

    The same functional as its linear sibling with one term replaced, which is
    exactly the mis-specification M4a measures: the closed form solves the
    tangent to this and lands 16 878 shares away at the reference case.
    """
    return cost_moments(trajectory, market).objective(lambda_risk)


@graded("expected_cost", POWER_LAW, "E[cost] in bps under the power law")
def power_law_world_expected_cost(
    trajectory, market: Market, lambda_risk: float
) -> float:
    """Mean implementation shortfall under FrontierView's 0.6-power charge."""
    return cost_moments(trajectory, market).expected


@graded("shortfall_variance", POWER_LAW, "V[cost] in bps^2")
def power_law_world_variance(trajectory, market: Market, lambda_risk: float) -> float:
    """Variance of implementation shortfall.

    Byte-for-byte the linear world's, because the shock model is untouched by
    M4a: ``shortfall_variance_bps2`` does not depend on the impact model at all.
    That is worth registering separately rather than sharing, so the equality is
    something ``tests/test_objective_registry.py`` measures instead of a fact
    about which function object two dictionary entries point at — and it is the
    reason M4a's differential needs a new reference for only one of the two
    moments.
    """
    return cost_moments(trajectory, market).variance


# ---------------------------------------------------------------------------
# Context — the vendored power-law charge as a Phase-1 result reports it
# ---------------------------------------------------------------------------


@context(
    "power_law_expected_cost",
    POWER_LAW,
    "E[cost] under FrontierView's 0.6-power temporary impact — reporting only",
)
def power_law_expected_cost(trajectory, market: Market, lambda_risk: float) -> float:
    """What the vendored model charges a schedule a *linear* env produced.

    Still context, still never a grade: these names exist so a Phase-1 result can
    quote the vendored number beside its own without the grading path ever being
    able to reach it. M4a did not lift that — it gave the power law its own
    world, where the identical arithmetic is registered under
    ``GRADED[POWER_LAW]`` and grades an env that actually charges it.
    """
    return cost_moments(trajectory, market).expected


@context(
    "power_law_objective",
    POWER_LAW,
    "E + lambda * V under the power law — reporting only",
)
def power_law_objective(trajectory, market: Market, lambda_risk: float) -> float:
    """The power-law world's objective, quoted beside a Phase-1 claim."""
    return cost_moments(trajectory, market).objective(lambda_risk)
