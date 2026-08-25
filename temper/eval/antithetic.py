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

* **Action identity across the pair — it does *not* end in M5, and task 4
  measured why.** The brief predicted this check would go red because the halves
  would see signals they disagree about, and M5 task 3 duly made the mirror negate
  the signal. Task 4 measured what that does to the estimator and reversed it.

  The pairing hands **one action to both halves**, so the mirror executes the
  primary's schedule whatever it sees. Negating the signal therefore negates the
  *whole* shock, ``xi = rho s + sqrt(1 - rho^2) e``, and the average of the two
  realised costs is the shock-free cost: an agent trained on it has no reason to
  tilt at all. Measured over 4 000 episodes at ``rho = 0.4``, the averaged reward
  had a standard deviation of **exactly zero** across signal paths and a
  correlation with the alpha term of **0.0016**. The estimator is blind to the
  thing the milestone is about.

  **Sharing the signal and negating only the price is the arrangement that works,
  and it is better than the brief hoped for.** Then the mirror's shock is
  ``rho s - sqrt(1 - rho^2) e``, the two halves' shocks average to ``rho s`` —
  the conditional mean, exactly — and the averaged reward is
  ``E[cost | s]`` itself: 4.5e-12 bps from the closed form over 2 500 episodes with
  a signal-reacting schedule, correlation 1.000000000. The unpredictable half of
  the price noise is removed entirely and the predictable half is kept whole.

  So both halves see the same observation, take the same action, and the old
  bitwise assertion stands verbatim in every world. What generalises instead is the
  *shock* identity, below.
* **Shock negation is exact.** The mirror's published cumulative shock is
  required to be the exact negation of the primary's, elementwise, on every
  step. IEEE arithmetic rounds symmetrically, so a sum of negated terms is the
  negated sum bitwise; anything else means the mirror drew fresh numbers.
* **Liquidity is shared, exactly** (M4b). The mirror's per-bin multiplier must
  equal the primary's — *not* be its mirror image. This is the third identity and
  it is what makes the first one survive M4b: because both halves see the same
  ``L``, they see the same observation even though the observation is now richer,
  so they take the same action and the price noise still cancels exactly given
  ``(x, L)``. What the pairing no longer removes is the *liquidity* noise, which
  is the reward variance M4b's agent has to train through.
* **The shocks average to the conditional mean** (M5) — which *is* the negation
  identity above, generalised rather than replaced. The signal is **shared**, so
  the mirror negates only the unpredictable component and

  .. code::

      (xi_k + xi'_k) / 2 = rho * s_{k-1} = E[xi_k | s]

  exactly. At ``rho = 0`` that is ``xi' = -xi`` and the assertion is M3's verbatim,
  which is why the earlier worlds keep the check they were built under rather than
  inheriting a looser one. Measured worst 7.1e-14 bps per step at ``rho = 0.4``.

  The halves are also required to be in one world, checked once at construction
  where a mismatch is loud and free: the same impact model and the same three seed
  addresses. That is M4a's lesson made structural — it was found there by a
  cancellation band four orders wide, after a run.

What M4b actually changed, against what §9 predicted
----------------------------------------------------
M4a's amendment named "a second, independent noise source or a price-bearing
observation" as what ends the pairing's exactness. Half of that is right and the
useful half is wrong: a second noise source is harmless as long as the pair
*shares* it, because what the action identity needs is not a poor observation but
an observation the two halves **agree about**. A price-bearing one would end it;
a richer one they both see does not.

And M5 is where the other half comes true, on the schedule §9 predicted. The
observation is price-bearing, the halves are handed signals they disagree about
by construction, and action identity ends. What is worth recording is that it
ends by *design* rather than by discovery: the pair could have shared the signal
the way it shares liquidity, and the reason it does not is that sharing it would
break the shock negation instead — there is no arrangement in which both hold.
Given the choice, the pairing keeps the exact property (negated shocks) and gives
up the derived one (identical actions).

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
policy: the agent never sees a shock, and the transform is handed to
:func:`~temper.agents.ppo.train` as an env factory parameter by the driver. From
M4b the observation is three-dimensional in the liquidity world and still carries
no price, which is the distinction the seam was drawn along in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import Env, Wrapper
from numpy.random import Generator

from temper.env import LIQUIDITY_KEY, SHOCK_KEY, ExecutionEnv, SignalStream
from temper.eval.variate import SHARES_KEY
from temper.oracle import AlphaSignal, alpha_coefficient


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


@dataclass(frozen=True)
class NegatedSignal(AlphaSignal):
    """The mirror's signal: the primary's draws, negated. ``NegatedDraws``' twin.

    **This is the arrangement M5 task 3 chose and task 4 measured and rejected**,
    and it is kept because the rejection is a measurement the suite re-runs rather
    than a paragraph. ``mirror_of`` does *not* use it.

    Negating the signal does make the mirror's shock the exact negation of the
    primary's — ``rho (-s) + sqrt(1 - rho^2) (-e) = -xi`` — which is why it looked
    right. What it also does is cancel the *predictable* half of the shock, and the
    predictable half is the entire content of the milestone: with one action fed to
    both halves the averaged reward becomes the shock-free cost, whose standard
    deviation across signal paths is exactly zero and whose correlation with the
    alpha term is 0.0016. An agent trained on it would learn to ignore the signal.
    See ``tests/test_m5_conditional_grading.py``, which measures both arrangements
    side by side.

    Deliberately *not* a full :class:`~temper.oracle.signal.AlphaSignal`
    look-alike, for the reason :class:`NegatedDraws` is not a full
    :class:`~numpy.random.Generator` one: what is delegated is every pure value and
    what is negated is exactly one thing, :meth:`draw`. ``correlation`` in
    particular is **not** negated — negating it would flip the model rather than
    the draw, and would put the mirror in a different world at the same seed
    address, which is the M4a bug's shape one seam along.
    """

    base: AlphaSignal

    def __post_init__(self) -> None:
        if not isinstance(self.base, AlphaSignal):
            raise TypeError(
                f"NegatedSignal wraps an AlphaSignal, got {type(self.base)!r}"
            )

    @property
    def name(self) -> str:
        return f"negated:{self.base.name}"

    @property
    def informative(self) -> bool:
        return self.base.informative

    @property
    def lag(self) -> int:
        return self.base.lag

    def mean(self) -> float:
        return self.base.mean()

    def variance(self) -> float:
        return self.base.variance()

    def correlation(self) -> float:
        return self.base.correlation()

    def quadrature(self, nodes: int):
        """Delegated. The nodes are symmetric about zero, so negation is a no-op."""
        return self.base.quadrature(nodes)

    def draw(self, rng, size):
        """``-base.draw(...)`` — negation is exact in IEEE arithmetic."""
        return -self.base.draw(rng, size)

    def draw_pair(self, rng, shape):
        raise NotImplementedError(
            "a mirrored signal has no independent joint draw: the shock it belongs "
            "with is the primary's, negated, and composing a fresh pair here would "
            "silently produce a second world rather than a mirror"
        )

    def shocks_from(self, signals, independent):
        raise NotImplementedError(
            "a mirrored signal composes its shocks from the primary's independent "
            "draws, negated by NegatedDraws inside the env, and not from a path "
            "handed in here"
        )

    def as_dict(self) -> dict:
        return self.base.as_dict() | {"mirrored": True}


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
    """The mirror: same market, order, lambda, **both worlds**, and the same address.

    Every per-episode property injected into the env has to be handed over, and
    M4b hands over two. The temporary-impact model was M4a's lesson and it was
    learned the expensive way: until then there was one world and the mirror
    rebuilt it by default, so the moment the model became injectable a mirror that
    defaulted was a *Phase-1* env being averaged against a power-law primary. The
    rewards still looked like rewards and the schedules were still identical; the
    estimator was simply no longer the one the config named. M4a task 4 caught it
    on the first cell it ran, at 0.06 bps per step against a 1e-12 band.

    **The liquidity stream is the second, and it is subtler**, because the failure
    would not merely mis-price — it would break the pairing's one exact property.
    The mirror is constructed at the same seed address and only its *price*
    generator is proxied by :class:`NegatedDraws`, so its liquidity generator runs
    unnegated at the same address and draws the **same path**. That is what M4b
    needs and it is why the action-identity assertion survives a richer
    observation: both halves see the same ``log L_k``, so both take the same
    action, so the price noise still cancels exactly given ``(x, L)``.

    A mirror on a *different* liquidity path would look exactly like the M4a bug
    looked. :class:`AntitheticPair` therefore asserts the shared multiplier on
    every step rather than trusting this constructor.

    Deliberately **not** antithetic in liquidity: mirroring ``u -> 1 - u`` on the
    liquidity uniform would make the two halves disagree about ``L``, hence about
    their actions, and would trade the pairing's one exact property for a partial
    second one.

    **The signal stream is the third, and it is SHARED, like liquidity** — which
    is the opposite of what M5's brief predicted and of what task 3 first built.
    Task 4 measured both arrangements and the difference is not marginal.

    The env composes each shock as ``rho * s + sqrt(1 - rho^2) * e``, and the pair
    hands **one action to both halves**. A mirror that negates the signal realises
    ``-xi`` exactly — which is why it looks right — and the two realised costs then
    average to the *shock-free* cost, because the predictable part of the shock is
    negated along with the unpredictable part. Measured: standard deviation across
    signal paths exactly zero, correlation with the alpha term 0.0016. The
    estimator would be blind to the milestone.

    Sharing the signal gives the mirror ``rho * s - sqrt(1 - rho^2) * e``. The
    unpredictable half is negated, the predictable half is not, and the two shocks
    average to ``rho * s`` — the conditional mean, exactly. So the averaged reward
    is ``E[cost | s]`` itself, to 4.5e-12 bps over 2 500 episodes with a
    signal-reacting schedule. The pairing removes **all** of the price noise and
    keeps the whole of the signal, which is the brief's "it should help more here
    than in M4b" arriving by the opposite mechanism to the one the brief named.

    Both halves therefore see the same observation and act identically, and the
    action-identity assertion does not retire after all. What generalises is the
    shock identity: ``(xi + xi') / 2 = rho * s``, which is ``xi' = -xi`` at
    ``rho = 0`` and so is M3's assertion verbatim in every earlier world.
    """
    if not isinstance(env, ExecutionEnv):
        raise TypeError(f"mirror_of takes a raw ExecutionEnv, got {type(env)!r}")
    root_seed, pool, stream = env.seed_address
    return MirrorEnv(
        env.market,
        env.order_size,
        env.lambda_risk,
        temporary_impact=env.temporary_impact,
        liquidity=env.liquidity,
        signal=env.signal,
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


#: Bar on ``|(walk + walk') / 2 - E[walk | s]|``, in bps. The two sides sum the
#: same terms in different orders, so this cannot be bitwise; 1e-9 is four orders
#: above the 7.1e-14 measured at ``rho = 0.4`` and ten below anything that could
#: move a reported number. In a world with no signal the assertion is the exact
#: negation instead and no tolerance is involved at all.
CONDITIONAL_WALK_TOLERANCE = 1.0e-9


def _same(a, b) -> bool:
    """Bitwise equality of two observations, as the arrays the env returned."""
    return np.array_equal(np.asarray(a), np.asarray(b))


def _observation_disagreement(primary, mirror) -> str | None:
    """What, if anything, is wrong with how the two halves' observations relate.

    Bitwise equality, in every world — M1a's assertion verbatim, and M5 does not
    retire it after all. The pair shares the signal and negates only the
    unpredictable half of the price, so the halves see the same market, the same
    inventory and the same prediction, and take the same action.

    Kept as its own function rather than inlined because M5 task 3 briefly needed a
    second relation here, and the shape of that mistake is worth leaving room for:
    a pairing arrangement is a choice about *which* exactness to keep, and the
    check that states it should be one named thing rather than a condition buried
    in two call sites.
    """
    a, b = np.asarray(primary), np.asarray(mirror)
    if a.shape != b.shape:
        return f"observation shapes differ ({a.shape} vs {b.shape})"
    if np.array_equal(a, b):
        return None
    return (
        f"the observation depends on the shocks ({a} vs {b}). Analytic grading "
        "and antithetic cancellation both assume the halves agree about what they "
        "are looking at"
    )


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
        # Whether the shock identity is the plain negation or its conditional-mean
        # generalisation. Read off the *primary* once: a mirror that had somehow
        # been built signal-free would then fail the per-step check rather than
        # quietly agreeing with a weaker rule.
        self.shares_signal = env.signal.informative
        self._alpha_walk = alpha_coefficient(env.market) * env.signal.signal.correlation()
        self._conditional_walk = np.zeros(env.market.n_bins + 1)
        self._primary_return = 0.0
        self._mirror_return = 0.0

        # Checked once, where a mismatch is loud and costs nothing. M4a's lesson
        # was that an injected per-episode property the mirror did not receive is
        # invisible in the rewards and the schedules; these are the three that
        # exist, plus the world they are charged in.
        if self.mirror.temporary_impact is not env.temporary_impact:
            raise PairDiverged(
                f"the mirror charges {self.mirror.cost_encoding!r} against the "
                f"primary's {env.cost_encoding!r}; the halves must be one world"
            )
        if self.mirror.seed_address != env.seed_address:
            raise PairDiverged(
                f"the mirror is at price address {self.mirror.seed_address} against "
                f"the primary's {env.seed_address}; the negation is only a mirror "
                "at the same address"
            )
        if self.mirror.liquidity_address != env.liquidity_address:
            raise PairDiverged(
                f"the mirror draws liquidity at {self.mirror.liquidity_address} "
                f"against the primary's {env.liquidity_address}"
            )
        if self.mirror.signal_address != env.signal_address:
            raise PairDiverged(
                f"the mirror draws its signal at {self.mirror.signal_address} "
                f"against the primary's {env.signal_address}; a mirrored signal is "
                "the same stream negated, never a second stream"
            )

    @property
    def primary(self) -> ExecutionEnv:
        return self.env  # type: ignore[return-value]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        mirrored, _ = self.mirror.reset(seed=seed, options=options)
        if not self.mirror.negated:
            raise PairDiverged("the mirror did not install its negating draws")
        wrong = _observation_disagreement(observation, mirrored)
        if wrong is not None:
            # The phrase "different observations" is load-bearing: M4b's
            # guarantee suite matches on it to show that a mirror on a different
            # liquidity path is refused before a step, and that behaviour is
            # unchanged here. The wording is part of the contract.
            raise PairDiverged(f"pair halves reset to different observations: {wrong}")
        # E[walk_k | s] for each step boundary: the conditional mean the two
        # halves' walks must average to. Zero in every world without a signal, so
        # the assertion below is M3's exact negation there.
        signals = self.primary.signals
        self._conditional_walk[0] = 0.0
        self._conditional_walk[1] = 0.0
        np.cumsum(self._alpha_walk * signals[:-1], out=self._conditional_walk[2:])
        self._primary_return = 0.0
        self._mirror_return = 0.0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        mirrored, mirror_reward, m_terminated, m_truncated, m_info = self.mirror.step(
            action
        )

        # -- the structural checks, every step -------------------------------
        wrong = _observation_disagreement(observation, mirrored)
        if wrong is not None:
            raise PairDiverged(f"pair halves diverged: {wrong}")
        if info[SHARES_KEY] != m_info[SHARES_KEY]:
            # **Not** retired, in any world. The pair hands one action to both
            # halves, so equal realised trades is a statement about the clip and
            # the inventory path rather than about the policy — it would fire on a
            # mirror built at a different order size, and it costs nothing.
            raise PairDiverged(
                f"pair halves realised different trades ({info[SHARES_KEY]} vs "
                f"{m_info[SHARES_KEY]}) from the same action"
            )
        if not self.shares_signal:
            if m_info[SHOCK_KEY] != -info[SHOCK_KEY]:
                raise PairDiverged(
                    f"the mirror's shock {m_info[SHOCK_KEY]!r} is not the exact "
                    f"negation of the primary's {info[SHOCK_KEY]!r}"
                )
        else:
            # The same identity, generalised: the pair negates the unpredictable
            # half of the shock and shares the predictable half, so the two walks
            # average to E[walk | s] rather than to zero. Not bitwise — the two
            # sides reach the number by different orderings — but ten orders inside
            # anything that could matter, and measured at 7.1e-14 per step.
            expected = self._conditional_walk[info["step"] + 1]
            middle = 0.5 * (info[SHOCK_KEY] + m_info[SHOCK_KEY])
            if abs(middle - expected) > CONDITIONAL_WALK_TOLERANCE:
                raise PairDiverged(
                    f"the pair's shocks average to {middle!r} where the signal "
                    f"path says E[walk | s] = {expected!r}; the mirror is not "
                    "negating exactly the unpredictable half of the price"
                )
        if m_info[LIQUIDITY_KEY] != info[LIQUIDITY_KEY]:
            # M4b's third identity. Liquidity is *common* across the pair, not
            # mirrored: the halves must see the same market and negate only the
            # price. A mirror on its own liquidity path would still produce
            # plausible rewards and identical-looking schedules while quietly
            # averaging two different worlds — the M4a mirror bug's exact shape,
            # one seam along.
            raise PairDiverged(
                f"pair halves saw different liquidity ({info[LIQUIDITY_KEY]!r} vs "
                f"{m_info[LIQUIDITY_KEY]!r}); the pairing holds liquidity common "
                "and negates only the price"
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
