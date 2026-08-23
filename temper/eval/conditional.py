"""Grading a policy that reacts: conditional expectation, not sampled cost.

M2 left the project a rule that held for three milestones — *a deterministic
policy on a price-free observation is graded analytically, not by Monte Carlo* —
and M4b is the first milestone that cannot obey it as written. The observation now
carries ``log L_k``, so the schedule is closed-loop: there is no single trajectory
to hand to a closed form, and the open-loop shortcut retires.

**What does not retire is the assertion that licensed it.** The price still enters
realised cost only through M1a's affine term, and the policy still never sees a
price — so conditioning on the liquidity path removes *all* of the price
randomness analytically, and ``E[cost | L]`` is a closed form
(:func:`~temper.oracle.cost.cost_moments` at the realised participations). The
grade is that closed form averaged over sampled liquidity paths, and the only
Monte-Carlo error anywhere is liquidity dispersion. There is **no price sampling**.

The successor to ``deterministic_schedule`` is one axis wider and just as
mechanical: hold the liquidity stream fixed, roll out on two unrelated *price*
streams, require the trajectories bitwise equal. It lives in
:mod:`temper.eval.grading` under its old name because it is the same claim —
"the price never entered the decision" — and it still fails loudly the moment
price reaches the observation.

Why the level is never estimated directly
-----------------------------------------
The per-path standard deviation of ``E[cost | L]`` is ~0.18 bps, and the whole
effect M4b measures is 0.062. An unpaired estimate of a policy's *level* at any
affordable path count is worthless. But the static optimum's expectation is a
closed form, so every policy is scored as a **difference** against the static
schedule on the *same* paths:

.. code::

    J_policy = J_static* + mean_p[ C_policy(L_p) - C_static(L_p) ]

Unbiased, because ``E[C_static(L)] = J_static*`` exactly, and ~9x smaller in
variance because the two share their liquidity. That is not a trick reserved for
the agent: the reference's own bounds are computed the same way
(:class:`~temper.eval.reference.PathBound`), so every rung on M4b's chart is a
difference from one closed form and the comparisons between them are paired.

The red flag became rigorous, and per-path
------------------------------------------
M4a's red flag rested on an algebraic certificate. M4b's rests on a relaxation,
and the relaxation is sharper than the brief predicted: perfect information beats
any policy **on every path**, not merely on average, because the clairvoyant solve
is the minimum over all schedules *given that path* and the agent's realised
schedule is one of them. So the hard failure is checked per path with no
confidence interval involved at all —
:attr:`LiquidityGrade.paths_below_clairvoyant` is a count, and any count above the
solver's own tolerance is a defect with a proof rather than a discovery.

The control that makes the headline mean something
--------------------------------------------------
:class:`ShuffledLiquidity` re-grades a trained policy with the observed ``L``
drawn independently of the ``L`` that is charged. If the advantage survives that,
the agent is not using the signal and the headline is measuring something else.
It costs a re-grade rather than a re-train, which is why M5's overfit-check
pattern arrives here a milestone early.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from gymnasium import Wrapper

from temper.env import ExecutionEnv, LiquidityStream, TemporaryImpact
from temper.eval.reference import LiquidityReferenceRow
from temper.eval.rollout import run_episode
from temper.oracle import Market, cost_moments, trades
from temper.seeding import pool_rng

#: Paired liquidity paths per graded policy. The brief pre-states M = 20 000 and
#: predicts a 95 % half-width of 1.36 % of the effect; the achieved half-width is
#: reported on every grade rather than assumed.
DEFAULT_EVAL_PATHS = 20_000


class ShuffledLiquidity(Wrapper):
    """Show the policy one liquidity path and charge it another — the control.

    The agent's observation's third coordinate is replaced by ``log L'`` drawn
    from an **independent** stream, while the env goes on charging its own ``L``.
    Everything else is identical: same policy, same paths, same grader, same bars.

    If the measured advantage survives this, the agent is not using the liquidity
    signal and whatever the headline is measuring, it is not adaptivity. The gap
    between the real and shuffled capture fractions is the actual claim.

    A wrapper rather than an env argument on purpose. This is an *evaluation*
    construct — it makes a world nobody trades in, where the observation lies —
    and putting it inside :class:`~temper.env.ExecutionEnv` would put a
    deliberately inconsistent market one default away from a training run.
    """

    def __init__(
        self,
        env: ExecutionEnv,
        *,
        root_seed: int,
        pool: str,
        stream_index: int,
    ) -> None:
        if not isinstance(env, ExecutionEnv):
            raise TypeError(
                f"ShuffledLiquidity wraps a raw ExecutionEnv, got {type(env)!r}"
            )
        if not env.liquidity.stochastic:
            raise ValueError(
                "shuffling a deterministic liquidity path is a no-op; the control "
                "is only meaningful where there is a signal to destroy"
            )
        super().__init__(env)
        self._rng = pool_rng(root_seed, pool, stream_index)
        self._law = env.liquidity.law
        self._n_bins = env.market.n_bins
        self._shown = np.zeros(self._n_bins + 1)
        self._index = 0

    def _draw(self) -> None:
        self._shown[: self._n_bins] = np.log(
            self._law.draw(self._rng, self._n_bins)
        )
        self._shown[self._n_bins] = 0.0

    def _mask(self, observation):
        seen = np.array(observation, dtype=np.float64, copy=True)
        seen[2] = self._shown[self._index]
        return seen

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._draw()
        self._index = 0
        return self._mask(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._index += 1
        return self._mask(observation), reward, terminated, truncated, info


def conditional_rollouts(
    policy,
    market: Market,
    order_size: float,
    lambda_risk: float,
    *,
    temporary_impact: TemporaryImpact,
    liquidity: LiquidityStream,
    root_seed: int,
    pool: str,
    stream_index: int = 0,
    paths: int = DEFAULT_EVAL_PATHS,
    shuffle: tuple[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll `policy` out on `paths` liquidity paths; return schedules and paths.

    One env, reset once per path, so the liquidity blocks come off a single
    stream in a fixed order. **That is what makes the random numbers common**:
    a second policy built at the same address sees exactly the same paths in
    exactly the same order, without either of them having to be handed a list.

    Every episode goes through the *real* env and the shared
    :func:`~temper.eval.rollout.run_episode` loop — the same one TWAP and both AC
    schedules use. Rolling forward in numpy instead would be a second ``step``
    loop, and constitution §4 has exactly one.

    `shuffle` is ``(pool, stream_index)`` for the liquidity-shuffled control: the
    observation's multiplier comes from that independent address while the env
    charges its own.

    Returns ``(trajectories (paths, n_bins + 1), multipliers (paths, n_bins))``.
    """
    if paths < 2:
        raise ValueError(f"a paired interval needs at least two paths, got {paths}")
    env = ExecutionEnv(
        market,
        order_size,
        lambda_risk,
        temporary_impact=temporary_impact,
        liquidity=liquidity,
        root_seed=root_seed,
        pool=pool,
        stream_index=stream_index,
    )
    driven = (
        env
        if shuffle is None
        else ShuffledLiquidity(
            env, root_seed=root_seed, pool=shuffle[0], stream_index=shuffle[1]
        )
    )

    trajectories = np.empty((paths, market.n_bins + 1))
    multipliers = np.empty((paths, market.n_bins))
    for index in range(paths):
        trajectories[index] = run_episode(driven, policy).trajectory
        # Read off the *env*, never the wrapper: the control lies to the policy
        # about the market and must not be able to lie to the grader about it.
        multipliers[index] = env.multipliers
    return trajectories, multipliers


def conditional_costs(
    trajectories: np.ndarray,
    multipliers: np.ndarray,
    market: Market,
    lambda_risk: float,
) -> np.ndarray:
    """``E[cost | L] + lambda V`` per path, through the grader's own route.

    Deliberately :func:`~temper.oracle.cost.cost_moments` per schedule rather than
    the vectorised :func:`~temper.oracle.adaptive.path_objective_bps` the bounds
    use: the two are pinned against each other in
    ``tests/test_m4b_adaptive_oracle.py``, and keeping the *graded* path on the
    same function every earlier milestone graded through is what stops a fast
    twin quietly becoming the definition of the objective.
    """
    return np.array(
        [
            cost_moments(trajectory, market, liquidity=path).objective(lambda_risk)
            for trajectory, path in zip(trajectories, multipliers)
        ]
    )


@dataclass(frozen=True)
class LiquidityGrade:
    """A liquidity-observing policy's objective, and how well it is known.

    Every fraction carries its absolute bps beside it, per §9's denominator
    entry, and the *level shift* is reported separately by the caller so nobody
    credits the agent with a constant a static solver gets for free.
    """

    name: str
    encoding: str
    #: ``J_static* + mean(C_policy - C_static)`` — the control-variated level.
    objective: float
    #: 95 % half-width of that estimate, in bps. Paired, so ~3x smaller than the
    #: level's own standard error would be.
    half_width: float
    paired_sd: float
    unpaired_sd: float
    paths: int
    #: ``J_policy - J_DP``, bps. The number the tolerance is a fraction of.
    excess: float
    advantage_fraction: float
    #: Mean and inter-quartile trajectories over the paths, for the figure.
    mean_trajectory: np.ndarray
    #: Paths on which the policy's conditional cost was *below* the clairvoyant
    #: relaxation's. Rigorously zero: perfect information is the per-path minimum
    #: over all schedules and the policy's realised schedule is one of them. Any
    #: count here is a defect with a proof behind it, and needs no interval.
    paths_below_clairvoyant: int
    #: The mean-level form of the same test, kept because it is what the brief
    #: pre-states and because it is the one a reader can check against the table.
    red_flag: bool
    #: ``J_policy < J_DP`` by more than the reference's own numerical
    #: uncertainty. Reported and investigated, never auto-failed: ``J_DP`` is
    #: converged rather than certified, so a policy marginally under it is
    #: possible without being a defect.
    soft_flag: bool

    @property
    def capture_fraction(self) -> float:
        """``(J_static* - J_policy) / (J_static* - J_DP)`` — M4b's headline.

        One is the adaptive optimum, zero is the best fixed schedule that knows
        the liquidity law, and negative is a policy that did worse than not
        reacting at all. **Not** measured from M4a's schedule: 3.8 % of that gap
        is a level shift any static solver picks up by re-solving.
        """
        return 1.0 - self.advantage_fraction

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "encoding": self.encoding,
            "objective_bps": self.objective,
            "half_width_bps": self.half_width,
            "paired_sd_bps": self.paired_sd,
            "unpaired_sd_bps": self.unpaired_sd,
            "paths": self.paths,
            "excess_bps": self.excess,
            "advantage_fraction": self.advantage_fraction,
            "capture_fraction": self.capture_fraction,
            "paths_below_clairvoyant": self.paths_below_clairvoyant,
            "red_flag": self.red_flag,
            "soft_flag": self.soft_flag,
            "mean_trajectory": [float(x) for x in self.mean_trajectory],
        }


def grade_conditional(
    trajectories: np.ndarray,
    multipliers: np.ndarray,
    market: Market,
    order_size: float,
    reference: LiquidityReferenceRow,
    *,
    name: str = "agent",
    clairvoyant_costs: Sequence[float] | None = None,
    soft_slack: float = 0.0,
) -> LiquidityGrade:
    """Score realised schedules against `reference`'s adaptive optimum.

    `reference` carries the lambda, the world **and** the liquidity law, so there
    is no way to grade at one lambda against an optimum computed at another, in
    one world against an optimum solved in the other, or at one ``sigma_log``
    against a reference built at a different one. The four travel together
    (invariant 7).
    """
    lambda_risk = reference.lambda_risk
    static_trajectory = reference.static.trajectory
    policy_costs = conditional_costs(trajectories, multipliers, market, lambda_risk)
    static_costs = conditional_costs(
        np.tile(static_trajectory, (multipliers.shape[0], 1)),
        multipliers,
        market,
        lambda_risk,
    )
    difference = policy_costs - static_costs
    objective = reference.static.objective + float(difference.mean())
    half_width = float(1.96 * difference.std(ddof=1) / math.sqrt(difference.size))

    excess = objective - reference.adaptive_bps
    if clairvoyant_costs is None:
        below = 0
    else:
        bound = np.asarray(clairvoyant_costs, dtype=float)
        # Path-aligned or nothing. The per-path red flag is only rigorous because
        # each comparison is "this policy against perfect information *on this
        # path*"; two arrays of the same length drawn from different streams would
        # broadcast happily and compare unrelated markets, which is a check that
        # looks like a proof and is not one.
        if bound.shape != policy_costs.shape:
            raise ValueError(
                f"the clairvoyant bound covers {bound.shape[0]} paths and the "
                f"policy was rolled out on {policy_costs.shape[0]}; the per-path "
                "red flag compares a policy with perfect information on the *same* "
                "path and there is no meaningful comparison between two different "
                "sets of them"
            )
        below = int(np.sum(policy_costs < bound - 1e-9))
    return LiquidityGrade(
        name=name,
        encoding=reference.encoding,
        objective=objective,
        half_width=half_width,
        paired_sd=float(difference.std(ddof=1)),
        unpaired_sd=float(policy_costs.std(ddof=1)),
        paths=int(difference.size),
        excess=excess,
        advantage_fraction=excess / reference.adaptive_advantage,
        mean_trajectory=trajectories.mean(axis=0),
        paths_below_clairvoyant=below,
        red_flag=bool(
            objective
            < reference.clairvoyant.value_bps - reference.clairvoyant.half_width_bps
        ),
        soft_flag=bool(objective < reference.adaptive_bps - soft_slack),
    )


def fixed_schedule_grade(
    trajectory,
    multipliers: np.ndarray,
    market: Market,
    order_size: float,
    reference: LiquidityReferenceRow,
    *,
    name: str,
) -> LiquidityGrade:
    """The same grade for a *fixed* schedule, on the same paths.

    Every baseline on M4b's chart goes through the identical arithmetic the agent
    does — constitution §5, baselines are policies and not special cases — so a
    comparison between them cannot be a comparison between two grading routes. A
    fixed schedule has a closed-form level, which is what makes this a useful
    check as well as a fair one: the sampled estimate must land on it.
    """
    return grade_conditional(
        np.tile(np.asarray(trajectory, dtype=float), (multipliers.shape[0], 1)),
        multipliers,
        market,
        order_size,
        reference,
        name=name,
    )


def trajectory_quantiles(
    trajectories: np.ndarray, quantiles: Sequence[float] = (0.25, 0.5, 0.75)
) -> dict[str, list[float]]:
    """Per-bin quantiles of a policy's realised schedules, for the figure.

    A liquidity-observing policy has a *distribution* of schedules rather than
    one, so the figure draws its spread. The house note *Below n ~ 10, draw every
    trace* is about seeds and still applies to those; this is the within-seed
    spread across paths, where there are twenty thousand and a band is the honest
    summary.
    """
    return {
        f"q{int(100 * q):02d}": [
            float(v) for v in np.quantile(trajectories, q, axis=0)
        ]
        for q in quantiles
    }
