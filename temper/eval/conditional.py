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

from temper.env import ExecutionEnv, LiquidityStream, SignalStream, TemporaryImpact
from temper.eval.reference import AlphaReferenceRow, LiquidityReferenceRow
from temper.eval.rollout import run_episode
from temper.oracle import AlphaSignal, Market, cost_moments, trades
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


# ---------------------------------------------------------------------------
# M5 — the same pattern, conditioned on the signal path
# ---------------------------------------------------------------------------

#: The seams a conditional grade may condition on, named so the two sides of the
#: check below speak one vocabulary rather than two.
LIQUIDITY_SEAM = "liquidity"
SIGNAL_SEAM = "signal"


class ConditioningMismatch(ValueError):
    """The grade's conditioning set is not the policy's observation set.

    Raised rather than warned, and *before* a number is produced, for the reason
    :class:`~temper.eval.grading.ScheduleNotDeterministic` is: a policy that fails
    this has not been scored badly, it has not been scored at all.
    """


def observed_seams(env: ExecutionEnv) -> frozenset[str]:
    """The stochastic seams this env's observation actually exposes.

    Read off the env's own seam objects rather than the observation's width,
    because width is a consequence and this is about *which* information is in
    there. A seam that is present but uninformative — deterministic liquidity, an
    absent signal, a signal pointed at an already-committed shock — exposes
    nothing and is correctly absent from this set.
    """
    seams = set()
    if env.liquidity.stochastic:
        seams.add(LIQUIDITY_SEAM)
    if env.signal.informative:
        seams.add(SIGNAL_SEAM)
    return frozenset(seams)


def check_conditioning_matches_observation(
    env: ExecutionEnv, conditioned_on: frozenset[str] | set[str]
) -> frozenset[str]:
    """The conditioning set of the grade must be the observation set of the policy.

    This is the property that keeps "the reward is the grade" honest, and until M5
    nothing in the repo asserted it. Every conditional grade since M4b rests on it
    and each was legitimate for a reason argued in prose: M4b conditioned on the
    liquidity path *because* that is what the observation carried, and M5
    conditions on the signal path for the same reason. Prose is not a check, and
    the two failure directions are opposite and both silent.

    **Observation strictly larger than conditioning.** The policy reacts to
    something the grade averages over, so the grade is a conditional expectation
    with respect to the wrong sigma-algebra: it is **biased**, and biased in the
    direction of the policy's own cleverness. Stack M4b's liquidity under M5's
    signal — a real backlog item, explicitly out of scope — and grade with
    ``signal_costs`` alone, and every identity in the differential still passes
    while the headline measures something nobody named.

    **Conditioning strictly larger than observation.** The grade knows something
    the policy never did, so it removes noise the policy actually faced: a policy
    is scored as though it were more deterministic than it is, its interval
    collapses, and the extreme of this is conditioning on the realised price,
    where "expected cost" becomes realised cost and every agent looks perfect.

    Equality, therefore, and it is checked where the env is built rather than
    where the number is computed — the env is the only object that knows what the
    policy could see.
    """
    observed = observed_seams(env)
    conditioning = frozenset(conditioned_on)
    if observed == conditioning:
        return observed
    missing = observed - conditioning
    extra = conditioning - observed
    detail = []
    if missing:
        detail.append(
            f"the policy observes {sorted(missing)} that the grade does not "
            "condition on, so the grade is biased by whatever the policy does "
            "with it"
        )
    if extra:
        detail.append(
            f"the grade conditions on {sorted(extra)} that the policy never "
            "observed, so it removes noise the policy actually faced"
        )
    raise ConditioningMismatch(
        f"conditioning set {sorted(conditioning)} against observation set "
        f"{sorted(observed)}: " + "; and ".join(detail)
    )


#: Paired **signal** paths per graded policy. Larger than M4b's 20 000 because
#: task 0 measured both: the 95 % half-width falls from 1.0405 % of the advantage
#: to 0.3219 % for about twenty seconds of extra rollout, which puts the median
#: tolerance bar 31x above the measurement floor against M4b's 7.4x. Recorded as
#: amendment 1 of ``docs/briefs/M5-alpha-aware-execution.md``, where the soft red
#: flag is stated in these units.
DEFAULT_SIGNAL_PATHS = 200_000


class ShuffledSignal(Wrapper):
    """M5's overfit control: the observed signal is drawn independently of the
    one the shock is composed from.

    The env charges its own market — its shocks are still ``rho s + ...`` in the
    signal *it* drew — and the observation is overwritten with a draw from an
    unrelated address. So the policy sees a signal of exactly the right shape,
    the same distribution, the same scale, and no relationship whatsoever to the
    prices it is about to pay.

    **This is the claim, not an extra.** A policy that still captures the
    advantage under it is not using the signal: it has found a better *fixed*
    schedule, and the milestone's headline would be measuring something nobody
    named. M5's prediction is stronger than M4b's was — the control should come
    back **negative**, because a policy that tilts on noise pays the execution
    premium and monetises nothing, so it does worse than not tilting at all.

    The realised signal is read off the **env**, never off this wrapper: the
    control lies to the policy about the market and must not be able to lie to
    the grader about it. Same rule as
    :class:`ShuffledLiquidity`, and it is the rule that keeps a control from
    quietly becoming a second world.
    """

    def __init__(
        self, env: ExecutionEnv, *, root_seed: int, pool: str, stream_index: int
    ) -> None:
        if not isinstance(env, ExecutionEnv):
            raise TypeError(
                f"ShuffledSignal wraps a raw ExecutionEnv, got {type(env)!r}"
            )
        if not env.signal.informative:
            raise ValueError(
                "shuffling an uninformative signal is a control over nothing: the "
                "observation does not carry it, so there is no channel to break"
            )
        super().__init__(env)
        self._rng = pool_rng(root_seed, pool, stream_index)
        self._law = env.signal.signal
        self._shown = np.zeros(env.market.n_bins + 1)
        self._index = 0

    @property
    def shown(self) -> np.ndarray:
        """The signal the policy was actually shown this episode."""
        return self._shown[: self.env.market.n_bins].copy()

    def _replace(self, observation):
        observation = np.asarray(observation, dtype=np.float64).copy()
        observation[-1] = self._shown[self._index]
        return observation

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._shown[: self.env.market.n_bins] = self._law.draw(
            self._rng, self.env.market.n_bins
        )
        self._shown[self.env.market.n_bins] = 0.0
        self._index = 0
        return self._replace(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._index += 1
        return self._replace(observation), reward, terminated, truncated, info


def signal_rollouts(
    policy,
    market: Market,
    order_size: float,
    lambda_risk: float,
    *,
    temporary_impact: TemporaryImpact,
    signal: SignalStream,
    liquidity: LiquidityStream | None = None,
    root_seed: int,
    pool: str,
    stream_index: int = 0,
    paths: int = DEFAULT_SIGNAL_PATHS,
    shuffle: tuple[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll `policy` out on `paths` signal paths; return schedules and paths.

    :func:`conditional_rollouts` one seam along, and identical in shape: one env,
    reset once per path, so the signal blocks come off a single stream in a fixed
    order and a second policy built at the same address sees exactly the same
    paths. **That is what makes the random numbers common**, and it is what lets
    the agent, the baselines and the reference's own feasible bound be compared
    path by path rather than level against level.

    Every episode goes through the *real* env and the shared
    :func:`~temper.eval.rollout.run_episode` loop. There is one ``step`` loop in
    this repo and rolling forward in numpy here would be a second one.

    Returns ``(trajectories (paths, n_bins + 1), signals (paths, n_bins))``.
    """
    if paths < 2:
        raise ValueError(f"a paired interval needs at least two paths, got {paths}")
    env = ExecutionEnv(
        market,
        order_size,
        lambda_risk,
        temporary_impact=temporary_impact,
        liquidity=liquidity,
        signal=signal,
        root_seed=root_seed,
        pool=pool,
        stream_index=stream_index,
    )
    # The licence, checked where the env is built and before a number exists:
    # this grade conditions on the signal path and nothing else, so the policy
    # must observe the signal path and nothing else.
    check_conditioning_matches_observation(env, {SIGNAL_SEAM})

    driven = (
        env
        if shuffle is None
        else ShuffledSignal(
            env, root_seed=root_seed, pool=shuffle[0], stream_index=shuffle[1]
        )
    )

    trajectories = np.empty((paths, market.n_bins + 1))
    signals = np.empty((paths, market.n_bins))
    for index in range(paths):
        trajectories[index] = run_episode(driven, policy).trajectory
        # Read off the *env*, never the wrapper: the control lies to the policy
        # about the market and must not be able to lie to the grader about it.
        signals[index] = env.signals
    return trajectories, signals


@dataclass(frozen=True)
class ConditionalCosts:
    """``E[cost | s]`` per path, already split the way the milestone reports it.

    The parts are first-class rather than derived by the caller because M5's whole
    methodological finding is that one fraction cannot grade this: the optimum
    monetises ~0.148 bps of signal and pays ~0.067 back, so a policy that captures
    0.15 and pays 0.07 and one that captures 0.25 and pays 0.17 score identically
    at the headline and differ entirely in what they did.

    All four are in bps, one entry per path, and they close:
    ``objective == impact + risk + alpha + invariant``.
    """

    impact: np.ndarray
    risk: np.ndarray
    alpha: np.ndarray
    invariant: np.ndarray
    objective: np.ndarray

    @property
    def execution(self) -> np.ndarray:
        """``impact + risk`` — the half of the objective the signal cannot touch.

        What :func:`~temper.oracle.alpha.execution_floor_bps` bounds from below,
        rigorously and by a *certified* number, for any policy at all.
        """
        return self.impact + self.risk


def signal_costs(
    trajectories: np.ndarray,
    signals: np.ndarray,
    market: Market,
    lambda_risk: float,
    signal: AlphaSignal,
) -> ConditionalCosts:
    """``E[cost | s] + lambda V`` per path, through the grader's own route.

    Deliberately :func:`~temper.oracle.cost.cost_moments` per schedule rather than
    the vectorised :func:`~temper.oracle.alpha.signal_path_objective_bps` the
    reference's bounds use — M4b's rule, unchanged: the two are pinned against each
    other in ``tests/test_m5_conditional_grading.py``, and keeping the *graded*
    path on the function every earlier milestone graded through is what stops a
    fast twin quietly becoming the definition of the objective.
    """
    rows = [
        cost_moments(trajectory, market, signal=signal, signals=path)
        for trajectory, path in zip(trajectories, signals)
    ]
    return ConditionalCosts(
        impact=np.array([row.temporary for row in rows]),
        risk=np.array([lambda_risk * row.variance for row in rows]),
        alpha=np.array([row.alpha for row in rows]),
        invariant=np.array([row.permanent + row.spread for row in rows]),
        objective=np.array([row.objective(lambda_risk) for row in rows]),
    )


@dataclass(frozen=True)
class AlphaGrade:
    """A signal-observing policy's objective, in the three parts the brief demands.

    Every fraction carries its absolute bps beside it (§9's denominator entry) and
    the headline never appears alone — :attr:`net_capture` is meaningless without
    :attr:`alpha_capture` and :attr:`premium_ratio` next to it, which is M5's own
    methodological finding rather than a house style.
    """

    name: str
    #: ``J_M4a + mean[C_policy(s) - C_M4a(s)]``, bps. Paired, so unbiased with a
    #: small interval rather than a level estimate with a large one.
    objective: float
    half_width_bps: float
    paired_sd_bps: float
    unpaired_sd_bps: float
    paths: int
    #: ``E[impact + risk]``, and the certified floor it may not go below.
    execution_bps: float
    execution_floor_bps: float
    #: ``-E[alpha]`` — the gross signal the policy actually monetised, bps.
    alpha_bps: float
    #: The reference's own three, for the fractions below.
    reference_objective: float
    reference_alpha_bps: float
    reference_premium_bps: float
    deterministic_objective: float
    #: ``E[impact + risk] < floor - eps``: a defect with a proof (convexity, and
    #: the floor is M4a's certified optimum).
    red_flag: bool
    #: ``J < J_DP`` beyond the DP's residual *and* this grade's own half-width.
    #: Reported and investigated, never auto-failed.
    soft_flag: bool
    mean_trajectory: np.ndarray

    @property
    def excess_bps(self) -> float:
        """How far above the converged optimum this policy sits, in bps."""
        return self.objective - self.reference_objective

    @property
    def net_capture(self) -> float:
        """``(J_M4a - J) / (J_M4a - J_DP)`` — the headline, and never alone."""
        return (self.deterministic_objective - self.objective) / (
            self.deterministic_objective - self.reference_objective
        )

    @property
    def alpha_capture(self) -> float:
        """The gross signal monetised, over the optimum's. The numerator's half."""
        return self.alpha_bps / self.reference_alpha_bps

    @property
    def execution_premium_bps(self) -> float:
        """What this policy paid for its alpha, over the *certified* floor."""
        return self.execution_bps - self.execution_floor_bps

    @property
    def premium_ratio(self) -> float:
        """The premium as a multiple of the optimum's. The denominator's half."""
        return self.execution_premium_bps / self.reference_premium_bps

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "objective_bps": self.objective,
            "half_width_bps": self.half_width_bps,
            "paired_sd_bps": self.paired_sd_bps,
            "unpaired_sd_bps": self.unpaired_sd_bps,
            "paths": self.paths,
            "excess_bps": self.excess_bps,
            "net_capture": self.net_capture,
            "alpha_bps": self.alpha_bps,
            "alpha_capture": self.alpha_capture,
            "execution_bps": self.execution_bps,
            "execution_floor_bps": self.execution_floor_bps,
            "execution_premium_bps": self.execution_premium_bps,
            "premium_ratio": self.premium_ratio,
            "red_flag": self.red_flag,
            "soft_flag": self.soft_flag,
            "mean_trajectory": [float(x) for x in self.mean_trajectory],
        }
def grade_signal(
    trajectories: np.ndarray,
    signals: np.ndarray,
    market: Market,
    order_size: float,
    reference: AlphaReferenceRow,
    signal: AlphaSignal,
    *,
    name: str = "agent",
    soft_slack: float = 0.0,
    red_flag_slack: float = 0.0,
) -> AlphaGrade:
    """Score realised schedules against `reference`'s alpha-aware optimum.

    `reference` carries the lambda, the world and both of the milestone's
    reference kinds; `signal` carries the law those were solved at, and the two
    are checked to agree before anything is computed. There is no way to grade at
    one lambda against an optimum computed at another, in one world against an
    optimum solved in the other, or at one ``rho`` against a reference built at a
    different one — they travel together (invariant 7).

    **The level is never estimated directly**, for M4b's reason in M5's numbers.
    Every policy is scored as a *difference* against M4a's certified optimum on the
    same signal paths:

    .. code::

        J_policy = J_M4a + mean_p[ C_policy(s_p) - C_M4a(s_p) ]

    Unbiased, and unusually cleanly so: M4a's schedule is deterministic, so
    ``E[alpha(M4a, s)] = -A rho sum_k h_k E[s] = 0`` **exactly**, and the anchor
    carries no sampling error at all. Task 0 measured the pairing at 31x in
    variance — 0.0606 bps paired against 0.3396 unpaired.

    Two flags, and they are different kinds of statement. The **hard** one is
    ``E[impact + risk] < J_M4a_varying``: impact and risk are convex in the trade
    weights and contain no signal, so by Jensen no policy of any kind can go below
    that, and the bound is M4a's *certified* optimum rather than a converged one.
    The **soft** one is ``J < J_DP``, reported and investigated and never
    auto-failed, because ``J_DP`` is converged and the grade is a Monte-Carlo
    estimate — `soft_slack` is the sum of the DP's own Richardson residual and this
    grade's half-width, per the brief's amendment 1.
    """
    if reference.signal.get("rho") != signal.correlation():
        raise ValueError(
            f"the reference was solved at rho = {reference.signal.get('rho')!r} and "
            f"the grade would be computed at {signal.correlation()!r}; the optimum "
            "and the law it is the optimum of travel together"
        )
    lambda_risk = reference.lambda_risk
    anchor = reference.optimal.trajectory
    costs = signal_costs(trajectories, signals, market, lambda_risk, signal)
    anchor_costs = signal_costs(
        np.tile(anchor, (signals.shape[0], 1)), signals, market, lambda_risk, signal
    )

    difference = costs.objective - anchor_costs.objective
    objective = reference.optimal.objective + float(difference.mean())
    half_width = float(
        1.96 * difference.std(ddof=1) / math.sqrt(difference.size)
    )
    execution = float(costs.execution.mean())
    return AlphaGrade(
        name=name,
        objective=objective,
        half_width_bps=half_width,
        paired_sd_bps=float(difference.std(ddof=1)),
        unpaired_sd_bps=float(costs.objective.std(ddof=1)),
        paths=int(difference.size),
        execution_bps=execution,
        execution_floor_bps=reference.execution_floor,
        alpha_bps=-float(costs.alpha.mean()),
        reference_objective=reference.adaptive_bps,
        reference_alpha_bps=reference.alpha_available,
        reference_premium_bps=reference.execution_premium,
        deterministic_objective=reference.optimal.objective,
        red_flag=execution < reference.execution_floor - red_flag_slack,
        soft_flag=objective < reference.adaptive_bps - soft_slack,
        mean_trajectory=trajectories.mean(axis=0),
    )
