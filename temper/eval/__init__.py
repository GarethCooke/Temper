"""Grading: run policies, score them, keep the scoring honest.

:mod:`~temper.eval.rollout` is the one loop every graded thing goes through;
:mod:`~temper.eval.metrics` is the pair of registries that decides what "graded"
is allowed to mean (invariant 7 — read its docstring before adding a metric).

From M2 there are two ways to reach a number and they are not interchangeable.
:func:`~temper.eval.rollout.sample_costs` estimates by Monte Carlo, which is what
M1's differential needs because it is testing the *simulator*.
:mod:`~temper.eval.grading` computes exactly, through the oracle, because a
deterministic policy on a price-free observation induces an open-loop schedule
and an open-loop schedule's moments are a closed form — read its docstring for
why the sampled route is not merely slower but unusable at M2's tolerance.

:mod:`~temper.eval.reference` is the oracle-only surface a milestone's thresholds
are derived from before any agent exists; :mod:`~temper.eval.variate` is the
sanctioned estimator change, quarantined and documented;
:mod:`~temper.eval.figures` is the only module in the package that may import
matplotlib.
"""

from .grading import (
    DEFAULT_EVAL_STREAMS,
    RED_FLAG_RTOL,
    Grade,
    ScheduleNotDeterministic,
    SeedSummary,
    deterministic_schedule,
    grade_policy,
    grade_trajectory,
    summarise,
)
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
from .provenance import Provenance, config_digest, git_revision, stamp
from .reference import (
    REFERENCE_SCHEDULES,
    LambdaRule,
    NoAdmissibleLambda,
    ReferenceRow,
    ScheduleReference,
    TrajectoryBand,
    reference_row,
    reference_table,
    select_lambda,
    trajectory_band,
    trajectory_deviation,
    variance_floor_bps2,
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
    "DEFAULT_EVAL_STREAMS",
    "ENCODINGS",
    "GRADED",
    "LINEAR",
    "POWER_LAW",
    "REFERENCE_SCHEDULES",
    "RED_FLAG_RTOL",
    "EpisodeResult",
    "Grade",
    "LambdaRule",
    "Metric",
    "NoAdmissibleLambda",
    "Provenance",
    "ReferenceRow",
    "SampleResult",
    "ScheduleNotDeterministic",
    "ScheduleReference",
    "SeedSummary",
    "TrajectoryBand",
    "config_digest",
    "deterministic_schedule",
    "git_revision",
    "grade_policy",
    "grade_trajectory",
    "reference_row",
    "reference_table",
    "register_context",
    "register_graded",
    "run_episode",
    "sample_costs",
    "select_lambda",
    "stamp",
    "standardise",
    "summarise",
    "trajectory_band",
    "trajectory_deviation",
    "variance_floor_bps2",
]
