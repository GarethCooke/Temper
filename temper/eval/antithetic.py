"""Antithetic pairing: run each episode as ``(xi, -xi)`` and average the two rewards.

**This is M3's reward regime, and it is validated before it is used.** M2's
control variate (:mod:`temper.eval.variate`) drove the reward variance to zero by
subtracting M1a's *analytic* noise identity — an exact fix, but one that needs
the closed form of the noise, which Phase 2 will not have. Antithetic pairing
never touches the analytic form. It needs only the ability to replay an episode
with every shock negated, and it works here for a structural reason established
in M1: the observation is ``(time left, inventory left)`` and carries no price,
so a policy shown the primary episode and its mirror sees identical observations
and takes **identical actions**. Same schedule, mirrored shocks — and because
Phase-1 cost is affine in the shocks (M1a: ``C - E[cost] = -sum_k (n_k / X)
walk_k``), the noise cancels *exactly* on the average of the two rewards. In
Phase 2 the cost stops being affine and the cancellation degrades to partial,
rather than disappearing the way the variate does.

Three pieces
------------
:class:`NegatedDraws` is a generator proxy whose standard normals are the exact
elementwise negation of the wrapped generator's. :class:`MirrorEnv` is an
:class:`~temper.env.ExecutionEnv` at the *same* seed address whose draws go
through that proxy — so its shocks are ``-xi`` by construction, not a fresh
sample from a symmetric distribution. :class:`AntitheticPair` holds a primary
env and its mirror, steps both with the same action, and returns the average
reward. Nothing under ``temper/env/`` changes (M3's out-of-scope list); the
mirror is a subclass that swaps one attribute after ``reset``, and it fails
loudly if that attribute is ever renamed.

Two structural checks, on every step, permanently
-------------------------------------------------
The brief asks for both because they are cheap now and expensive later.

* **Action identity across the pair.** The pair asserts, bitwise, that the two
  halves see the same observation and realise the same trade on every step.
  That is the assumption the whole method rests on, and it fails silently and
  instantly the moment an observation carries price — so it is asserted in the
  wrapper, not only in a test. When Phase 2 enriches the observation this raises
  :class:`PairDiverged`, and that is correct: it is the signal that the
  pairing's exactness has lapsed, which is information M4 needs.
* **Shock negation is exact.** The mirror's published cumulative shock is
  required to be the exact negation of the primary's, elementwise, on every
  step. IEEE arithmetic rounds symmetrically, so a sum of negated terms is the
  negated sum bitwise; anything else means the mirror drew fresh numbers.

The reward-variance evidence
----------------------------
The brief wants the realised reward variance *measured*, not inferred from the
outcome. :class:`PairLedger` records both halves' episode returns as they
finish, and the training driver closes it once per update: the variance of the
primary half's returns is exactly what the sampled-reward regime would have
trained on under these same actions, the variance of the averaged returns is
what the agent actually trained on, and the mean square of the cancelled term is
how much noise the pairing removed. Same run, same actions, no cross-run
confounding.

Why it lives here and not beside the training loop
--------------------------------------------------
Because it names the env's shock key to check the negation, and
``tests/test_repo_invariants.py`` rejects that key — by literal or by name —
anywhere under ``temper/agents/``. Like the variate, this is an estimator, not a
policy: the observation stays two-dimensional, the agent never sees a shock, and
the transform is handed to :func:`~temper.agents.ppo.train` as an env factory
parameter by the driver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import Env, Wrapper
from numpy.random import Generator

from temper.env import SHOCK_KEY, ExecutionEnv
from temper.eval.variate import SHARES_KEY


class PairDiverged(AssertionError):
    """The two halves of an antithetic pair stopped being mirror images.

    Raised rather than returned, and it aborts training. A pair whose halves see
    different observations would take different actions under any stochastic
    policy, the schedules would differ, and the average reward would no longer
    be the noise-free reward — the estimator would silently become something
    else. Which is precisely what happens the day price enters the observation.
    """


# ---------------------------------------------------------------------------
# Exact negation of a generator's draws
# ---------------------------------------------------------------------------


class NegatedDraws:
    """A generator proxy: every standard normal is the negation of `base`'s.

    Deliberately *not* a full :class:`numpy.random.Generator` look-alike. Only
    ``standard_normal`` is provided, because that is the one method
    :class:`~temper.env.ExecutionEnv` draws through, and negation is only the
    right mirror for a symmetric distribution centred on zero. If the env ever
    reaches for ``uniform`` or ``normal(loc, ...)`` the mirror raises
    ``AttributeError`` instead of quietly returning something that is not a
    mirror.
    """

    __slots__ = ("_base",)

    def __init__(self, base: Generator) -> None:
        if not isinstance(base, Generator):
            raise TypeError(f"NegatedDraws wraps a numpy Generator, got {type(base)!r}")
        self._base = base

    @property
    def base(self) -> Generator:
        return self._base

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        """``-base.standard_normal(...)`` — negation is exact in IEEE arithmetic."""
        if out is not None:
            raise ValueError("NegatedDraws does not support the `out` argument")
        if size is None:
            return -self._base.standard_normal(dtype=dtype)
        return -self._base.standard_normal(size, dtype=dtype)


class MirrorEnv(ExecutionEnv):
    """An :class:`~temper.env.ExecutionEnv` whose shocks are exactly negated.

    Constructed at the *same* seed address as the primary it mirrors. Both
    generators then start from the same state and are advanced once per step in
    lockstep, so the mirror's ``k``-th draw is ``-xi_k`` where ``xi_k`` is the
    primary's — the negation of the same number, not a fresh number from a
    mirrored distribution. :class:`AntitheticPair` asserts that on every step.

    The one thing this touches is the env's private generator, immediately after
    ``reset`` has installed it. That is a deliberate reach across the seam
    rather than an edit to ``temper/env/`` (which M3 may not change), and it is
    made loud: if the attribute is renamed or its type changes, construction
    fails here rather than the mirror quietly drawing un-negated shocks.
    """

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        result = super().reset(seed=seed, options=options)
        try:
            rng = self._rng
        except AttributeError:  # pragma: no cover - the env was refactored
            raise RuntimeError(
                "MirrorEnv reaches ExecutionEnv's `_rng` after reset and it is "
                "gone; the mirror can no longer negate the draws"
            ) from None
        if not isinstance(rng, NegatedDraws):
            self._rng = NegatedDraws(rng)
        return result

    @property
    def negated(self) -> bool:
        """True once ``reset`` has installed the negating proxy."""
        return isinstance(getattr(self, "_rng", None), NegatedDraws)


def mirror_of(env: ExecutionEnv) -> MirrorEnv:
    """The mirror of a raw env: same market, order, lambda, **world** and address.

    The temporary-impact model is copied across, and that is not housekeeping.
    Until M4a there was only one world and the mirror rebuilt it by default; the
    moment the model became injectable, a mirror that defaulted was a *Phase-1*
    env being averaged against a power-law primary. The pairing's cancellation is
    exact only because both halves pay the same schedule-dependent cost, so the
    two halves disagreeing about what a trade costs breaks the estimator
    silently — the rewards still look like rewards.

    Found by M4a task 4 rather than reasoned about, which is what that task is
    for: the antithetic average stopped matching the control variate by 0.06 bps
    per step, four orders above the 1e-12 band, on the first cell it was run on.
    """
    if not isinstance(env, ExecutionEnv):
        raise TypeError(f"mirror_of takes a raw ExecutionEnv, got {type(env)!r}")
    root_seed, pool, stream = env.seed_address
    return MirrorEnv(
        env.market,
        env.order_size,
        env.lambda_risk,
        temporary_impact=env.temporary_impact,
        root_seed=root_seed,
        pool=pool,
        stream_index=stream,
    )


# ---------------------------------------------------------------------------
# The reward-variance ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairUpdateStats:
    """One update's worth of finished pairs, summarised.

    All variances are of *episode returns*, in the env's own bps (the pair sits
    below the reward scale), with ``ddof = 1``. ``sampled_variance`` is the
    variance of the primary half's returns — what vanilla PPO would have trained
    on under these same actions; ``averaged_variance`` is what the agent did
    train on; ``cancelled_mean_square`` is ``mean(((r+ - r-) / 2)^2)``, the power
    of the term the pairing removed.
    """

    episodes: int
    sampled_variance: float
    mirror_variance: float
    averaged_variance: float
    cancelled_mean_square: float

    @property
    def variance_ratio(self) -> float:
        """``averaged / sampled`` — how much of the reward variance survived."""
        if not (self.sampled_variance > 0.0):
            return float("nan")
        return self.averaged_variance / self.sampled_variance

    def as_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "sampled_variance": self.sampled_variance,
            "mirror_variance": self.mirror_variance,
            "averaged_variance": self.averaged_variance,
            "cancelled_mean_square": self.cancelled_mean_square,
            "variance_ratio": self.variance_ratio,
        }


def _variance(values: np.ndarray) -> float:
    return float(np.var(values, ddof=1)) if values.size >= 2 else float("nan")


class PairLedger:
    """Both halves' episode returns, closed once per update by the driver.

    Every :class:`AntitheticPair` of one training seed shares one ledger; the
    envs record, the driver's progress hook calls :meth:`close_update`, and
    what accumulates in :attr:`updates` is the per-update trace the results file
    reports. Nothing here knows how many envs there are or when an update ends —
    that is the training loop's business, and the loop tells the driver.
    """

    def __init__(self) -> None:
        self._primary: list[float] = []
        self._mirror: list[float] = []
        self.updates: list[PairUpdateStats] = []

    def record(self, primary_return: float, mirror_return: float) -> None:
        self._primary.append(float(primary_return))
        self._mirror.append(float(mirror_return))

    @property
    def pending(self) -> int:
        """Finished pairs not yet closed into an update."""
        return len(self._primary)

    def close_update(self) -> PairUpdateStats:
        primary = np.asarray(self._primary, dtype=np.float64)
        mirror = np.asarray(self._mirror, dtype=np.float64)
        self._primary.clear()
        self._mirror.clear()
        averaged = 0.5 * (primary + mirror)
        cancelled = 0.5 * (primary - mirror)
        stats = PairUpdateStats(
            episodes=int(primary.size),
            sampled_variance=_variance(primary),
            mirror_variance=_variance(mirror),
            averaged_variance=_variance(averaged),
            cancelled_mean_square=(
                float(np.mean(cancelled**2)) if cancelled.size else float("nan")
            ),
        )
        self.updates.append(stats)
        return stats


# ---------------------------------------------------------------------------
# The pair
# ---------------------------------------------------------------------------


def _same(a, b) -> bool:
    """Bitwise equality of two observations, as the arrays the env returned."""
    return np.array_equal(np.asarray(a), np.asarray(b))


class AntitheticPair(Wrapper):
    """One env to the training loop; two mirrored envs underneath.

    Wraps the *raw* :class:`~temper.env.ExecutionEnv` — below the fraction
    action and below the reward scale — so that both halves receive the same
    shares and the average is taken in the env's own bps. ``reset`` resets both;
    ``step`` steps both with the same action and returns the primary's
    observation and ``info`` with the averaged reward. On every step the two
    halves must agree bitwise on the observation and on the realised trade, and
    the mirror's cumulative shock must be the exact negation of the primary's;
    any of those failing raises :class:`PairDiverged`.
    """

    def __init__(self, env: ExecutionEnv, ledger: PairLedger | None = None) -> None:
        if not isinstance(env, ExecutionEnv):
            raise TypeError(
                "AntitheticPair wraps the raw ExecutionEnv (below FractionAction "
                f"and RewardScale), got {type(env)!r}"
            )
        super().__init__(env)
        self.mirror = mirror_of(env)
        self.ledger = ledger
        self._primary_return = 0.0
        self._mirror_return = 0.0

    @property
    def primary(self) -> ExecutionEnv:
        return self.env  # type: ignore[return-value]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        mirrored, _ = self.mirror.reset(seed=seed, options=options)
        if not self.mirror.negated:
            raise PairDiverged("the mirror did not install its negating draws")
        if not _same(observation, mirrored):
            raise PairDiverged(
                f"pair halves reset to different observations: {observation} vs "
                f"{mirrored}"
            )
        self._primary_return = 0.0
        self._mirror_return = 0.0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        mirrored, mirror_reward, m_terminated, m_truncated, m_info = self.mirror.step(
            action
        )

        # -- the structural checks, every step -------------------------------
        if not _same(observation, mirrored):
            raise PairDiverged(
                "pair halves diverged: the observation depends on the shocks "
                f"({observation} vs {mirrored}). Analytic grading and antithetic "
                "cancellation both assume a price-free observation."
            )
        if info[SHARES_KEY] != m_info[SHARES_KEY]:
            raise PairDiverged(
                f"pair halves realised different trades ({info[SHARES_KEY]} vs "
                f"{m_info[SHARES_KEY]}) from the same action"
            )
        if m_info[SHOCK_KEY] != -info[SHOCK_KEY]:
            raise PairDiverged(
                f"the mirror's shock {m_info[SHOCK_KEY]!r} is not the exact "
                f"negation of the primary's {info[SHOCK_KEY]!r}"
            )
        if (terminated, truncated) != (m_terminated, m_truncated):
            raise PairDiverged("pair halves disagree on episode termination")

        reward = float(reward)
        mirror_reward = float(mirror_reward)
        self._primary_return += reward
        self._mirror_return += mirror_reward
        if (terminated or truncated) and self.ledger is not None:
            self.ledger.record(self._primary_return, self._mirror_return)

        return observation, 0.5 * (reward + mirror_reward), terminated, truncated, info

    def close(self) -> None:
        self.mirror.close()
        super().close()


def antithetic_reward(ledger: PairLedger | None = None):
    """`reward_wrapper` form, for :func:`~temper.agents.execution.execution_env_factory`.

    Returns the ``Env -> Env`` callable the factory applies to the raw env; every
    env built through the returned callable records into the same `ledger`.
    """

    def wrap(env: Env) -> Env:
        return AntitheticPair(env, ledger)

    return wrap
