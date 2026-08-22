"""Policies: the baselines, and from M2 the PPO agent.

Everything in here exposes ``act(observation)`` and is graded by
:mod:`temper.eval` through :class:`~temper.env.ExecutionEnv` — see
:mod:`temper.agents.baselines` for why that uniformity is load-bearing.

:mod:`temper.agents.ppo` is the algorithm (CleanRL-derived, single file);
:mod:`temper.agents.execution` is the boundary between it and Temper's env — the
fraction-of-remaining action, the fixed reward scale, and the wrapper that turns
a trained network back into something the eval harness cannot tell apart from
TWAP. Neither may see the price path: ``tests/test_repo_invariants.py`` rejects
the env's shock key, by name or by literal, anywhere under this package.

``torch`` is imported by :mod:`temper.agents.ppo` and by nothing below it. The
oracle and the env stay pure numpy.

:mod:`temper.agents.checkpoint` is how a trained policy leaves this package as a
file: a plain ``.npz`` of named arrays plus a JSON metadata document, readable
with numpy alone. That is what lets ``client/`` run the policy on a venue without
importing the training stack (constitution §3, and the §9 entry *A trained policy
is a committed artefact*).
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
from .checkpoint import (
    POLICY_FORMAT,
    PolicyCheckpoint,
    agent_from_checkpoint,
    load_checkpoint,
    network_description,
    policy_from_checkpoint,
    save_policy,
)
from .execution import (
    FRACTION_SPACE,
    FractionAction,
    FractionPolicy,
    PPOPolicy,
    RewardScale,
    as_fraction,
    execution_env_factory,
    fraction_to_shares,
    twap_fractions,
)
from .ppo import Agent, PPOConfig, TrainResult, evaluate, train

__all__ = [
    "BASELINES",
    "FRACTION_SPACE",
    "POLICY_FORMAT",
    "Agent",
    "FractionAction",
    "FractionPolicy",
    "PPOConfig",
    "PPOPolicy",
    "Policy",
    "PolicyCheckpoint",
    "RewardScale",
    "SchedulePolicy",
    "TrainResult",
    "ac_policy",
    "agent_from_checkpoint",
    "as_fraction",
    "baseline",
    "evaluate",
    "execution_env_factory",
    "fraction_to_shares",
    "load_checkpoint",
    "network_description",
    "optimal_policy",
    "policy_from_checkpoint",
    "save_policy",
    "train",
    "twap_fractions",
    "twap_policy",
]
