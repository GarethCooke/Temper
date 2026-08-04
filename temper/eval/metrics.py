"""Two registries: what a claim may be graded on, and what is context only.

Constitution invariant 7 says the training reward, the evaluation metric and the
oracle optimise one functional. M1 discovered that keeping that true costs
something concrete: FrontierView charges a 0.6-power temporary impact, the closed
form solves its tangent, and on the Phase-1 golden parameter sets the two paths
differ by 12 % to 54 % of expected cost — nowhere near reducing to each other
(``ARCHITECTURE.md`` §9, 2026-08-04). So Phase 1 is the linearised world
end-to-end: env dynamics, training reward, evaluation metric and oracle all speak
:func:`~temper.oracle.cost.schedule_moments`.

That leaves :func:`~temper.oracle.cost.cost_moments` — the vendored power-law
charge, and a genuinely interesting number — with nowhere safe to sit. It sits
here, in ``CONTEXT``, where it can be reported next to a result and never be the
result. The quarantine is mechanical: :func:`register_graded` refuses a
power-law-encoded metric, and ``tests/test_objective_registry.py`` checks the
labels are not merely honest but *true*, by evaluating every graded metric
against both encodings on the golden cases.

M4 promotes the power law to being the world itself; at that point it becomes a
gradeable encoding under a new env, by a §9 amendment, not by editing a string
here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from temper.oracle import Market, cost_moments, schedule_moments

#: The frozen Phase-1 objective: linear temporary impact at the tangent eta_tilde.
LINEAR = "linear"

#: FrontierView's power-law charge. Reporting context in Phase 1; never a grade.
POWER_LAW = "power_law"

ENCODINGS: tuple[str, ...] = (LINEAR, POWER_LAW)

#: The encodings a graded metric may use. Widening this is a §9 amendment.
GRADEABLE_ENCODINGS: frozenset[str] = frozenset({LINEAR})


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


#: Metrics a reported claim may be made on.
GRADED: dict[str, Metric] = {}

#: Metrics that may only ever appear beside a claim.
CONTEXT: dict[str, Metric] = {}


def _register(registry: dict[str, Metric], metric: Metric) -> Metric:
    if metric.name in registry:
        raise ValueError(f"metric {metric.name!r} is already registered")
    registry[metric.name] = metric
    return metric


def register_graded(metric: Metric) -> Metric:
    """Register a metric a claim may be graded on. Linear encoding only."""
    if metric.encoding not in GRADEABLE_ENCODINGS:
        raise ValueError(
            f"metric {metric.name!r} uses the {metric.encoding!r} encoding, which "
            "cannot be graded in Phase 1: the env, the reward and the oracle all "
            "charge linear temporary impact, and grading against a different "
            "functional would break invariant 7. Register it under CONTEXT."
        )
    return _register(GRADED, metric)


def register_context(metric: Metric) -> Metric:
    """Register a metric that may be reported alongside a claim, never as one."""
    return _register(CONTEXT, metric)


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


# ---------------------------------------------------------------------------
# Graded — the frozen objective and its two halves
# ---------------------------------------------------------------------------


@graded("objective", LINEAR, "E[cost] + lambda * V[cost], the frozen objective")
def objective(trajectory, market: Market, lambda_risk: float) -> float:
    """What the env's summed reward is the negative of."""
    return schedule_moments(trajectory, market).objective(lambda_risk)


@graded("expected_cost", LINEAR, "E[cost] in bps of notional")
def expected_cost(trajectory, market: Market, lambda_risk: float) -> float:
    """Mean implementation shortfall of the schedule."""
    return schedule_moments(trajectory, market).expected


@graded("shortfall_variance", LINEAR, "V[cost] in bps^2")
def shortfall_variance(trajectory, market: Market, lambda_risk: float) -> float:
    """Variance of implementation shortfall — identical under both encodings."""
    return schedule_moments(trajectory, market).variance


# ---------------------------------------------------------------------------
# Context — the vendored power-law charge, quarantined
# ---------------------------------------------------------------------------


@context(
    "power_law_expected_cost",
    POWER_LAW,
    "E[cost] under FrontierView's 0.6-power temporary impact — reporting only",
)
def power_law_expected_cost(trajectory, market: Market, lambda_risk: float) -> float:
    """What the vendored model would charge this schedule. Never a grade in Phase 1."""
    return cost_moments(trajectory, market).expected


@context(
    "power_law_objective",
    POWER_LAW,
    "E + lambda * V under the power law — reporting only",
)
def power_law_objective(trajectory, market: Market, lambda_risk: float) -> float:
    """The power-law world's objective. M4 makes this a world; M1 makes it a note."""
    return cost_moments(trajectory, market).objective(lambda_risk)
