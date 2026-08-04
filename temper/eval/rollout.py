"""The policy-agnostic Monte-Carlo harness: run episodes, collect what they cost.

Every graded thing — TWAP, both AC schedules, the M2 agent — reaches a number
through this module and no other, so a comparison between them cannot be a
comparison between two rollout loops (constitution §5).

There is one loop. :func:`run_episode` keeps the per-step record the identity
tests need; :func:`sample_costs` runs the same loop with the record switched off
because M1's deep tier is 2.7 million episodes and a per-step list per episode is
not free. Keeping them as one function rather than two is deliberate: a lean
duplicate of the loop is exactly where a differential harness quietly stops
testing the thing it thinks it is testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temper.env import EPISODE_KEY, ExecutionEnv


@dataclass(frozen=True)
class EpisodeResult:
    """One episode, in full.

    ``cost_bps`` comes from the realised proceeds and ``shortfall_bps`` from
    summing the per-step charges: two routes to the same number, which is what
    the first identity test compares.
    """

    trajectory: np.ndarray
    cost_bps: float
    shortfall_bps: float
    penalty_bps: float
    reward: float
    shortfalls: np.ndarray | None = None
    penalties: np.ndarray | None = None
    rewards: np.ndarray | None = None
    shares: np.ndarray | None = None
    walks: np.ndarray | None = None


@dataclass(frozen=True)
class SampleResult:
    """``n_episodes`` costs, plus the schedule that produced them."""

    costs: np.ndarray
    trajectory: np.ndarray

    @property
    def n_episodes(self) -> int:
        return int(self.costs.size)


def _episode(env: ExecutionEnv, policy, record: bool):
    """Step one episode to termination. Returns the env's episode summary."""
    observation, _ = env.reset()
    policy.reset()

    shortfalls: list[float] = []
    penalties: list[float] = []
    rewards: list[float] = []
    shares: list[float] = []
    walks: list[float] = []

    terminated = False
    info: dict = {}
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(
            policy.act(observation)
        )
        if truncated:  # the env has no time limit; a truncation would be a bug
            raise RuntimeError("ExecutionEnv truncated an episode")
        if record:
            shortfalls.append(info["shortfall_bps"])
            penalties.append(info["penalty_bps"])
            rewards.append(reward)
            shares.append(info["shares"])
            walks.append(info["walk_bps"])

    summary = info[EPISODE_KEY]
    if not record:
        return summary, None
    steps = (
        np.array(shortfalls),
        np.array(penalties),
        np.array(rewards),
        np.array(shares),
        np.array(walks),
    )
    return summary, steps


def run_episode(env: ExecutionEnv, policy, *, seed: int | None = None) -> EpisodeResult:
    """Run one episode and keep every per-step quantity."""
    if seed is not None:
        env.reset(seed=seed)
    summary, steps = _episode(env, policy, record=True)
    shortfalls, penalties, rewards, shares, walks = steps
    return EpisodeResult(
        trajectory=summary["trajectory"],
        cost_bps=summary["cost_bps"],
        shortfall_bps=summary["shortfall_bps"],
        penalty_bps=summary["penalty_bps"],
        reward=summary["reward"],
        shortfalls=shortfalls,
        penalties=penalties,
        rewards=rewards,
        shares=shares,
        walks=walks,
    )


def sample_costs(
    env: ExecutionEnv,
    policy,
    n_episodes: int,
    *,
    seed: int | None = None,
    require_fixed_schedule: bool = False,
) -> SampleResult:
    """Realised cost of `n_episodes` episodes, in bps of notional.

    `require_fixed_schedule` asserts that every episode realised the *same*
    inventory trajectory. The differential standardises costs against
    :func:`~temper.oracle.cost.schedule_moments` of one realised schedule, which
    is only the right reference if there was exactly one — including after the
    env's terminal force-liquidation has had its say.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    if seed is not None:
        env.reset(seed=seed)

    costs = np.empty(n_episodes, dtype=np.float64)
    reference: np.ndarray | None = None
    for index in range(n_episodes):
        summary, _ = _episode(env, policy, record=False)
        costs[index] = summary["cost_bps"]
        trajectory = summary["trajectory"]
        if reference is None:
            reference = trajectory
        elif require_fixed_schedule and not np.array_equal(trajectory, reference):
            raise AssertionError(
                f"policy {getattr(policy, 'name', policy)!r} realised a different "
                f"schedule in episode {index}; it is not a fixed schedule"
            )

    assert reference is not None  # n_episodes > 0
    return SampleResult(costs=costs, trajectory=reference)


def standardise(costs: np.ndarray, expected: float, variance: float) -> np.ndarray:
    """``z = (C - E) / sqrt(V)`` against the analytic moments.

    Under Phase-1 dynamics a deterministic schedule's shortfall is a fixed linear
    combination of independent Gaussian shocks, so it is *exactly* Gaussian and
    ``z`` is exactly standard normal under the null. That is what lets M1's bands
    be stated as ``4/sqrt(N)`` and ``4*sqrt(2/N)`` without a chi-squared quantile,
    and therefore without scipy.
    """
    if variance <= 0.0:
        raise ValueError(f"variance must be positive to standardise, got {variance}")
    return (costs - expected) / np.sqrt(variance)
