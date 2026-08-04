"""Monte-Carlo grading: run policies, score them, keep the scoring honest.

:mod:`~temper.eval.rollout` is the one loop every graded thing goes through;
:mod:`~temper.eval.metrics` is the pair of registries that decides what "graded"
is allowed to mean (invariant 7 — read its docstring before adding a metric).
Figures and report generation arrive with the milestones that need them.
"""

from .metrics import (
    CONTEXT,
    ENCODINGS,
    GRADED,
    LINEAR,
    POWER_LAW,
    Metric,
    register_context,
    register_graded,
)
from .rollout import (
    EpisodeResult,
    SampleResult,
    run_episode,
    sample_costs,
    standardise,
)

__all__ = [
    "CONTEXT",
    "ENCODINGS",
    "GRADED",
    "LINEAR",
    "POWER_LAW",
    "EpisodeResult",
    "Metric",
    "SampleResult",
    "register_context",
    "register_graded",
    "run_episode",
    "sample_costs",
    "standardise",
]
