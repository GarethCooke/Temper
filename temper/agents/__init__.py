"""Policies: the baselines now, the PPO agent from M2.

Everything in here exposes ``act(observation)`` and is graded by
:mod:`temper.eval` through :class:`~temper.env.ExecutionEnv` — see
:mod:`temper.agents.baselines` for why that uniformity is load-bearing.
"""

from .baselines import (
    BASELINES,
    Policy,
    SchedulePolicy,
    ac_policy,
    baseline,
    optimal_policy,
    twap_policy,
)

__all__ = [
    "BASELINES",
    "Policy",
    "SchedulePolicy",
    "ac_policy",
    "baseline",
    "optimal_policy",
    "twap_policy",
]
