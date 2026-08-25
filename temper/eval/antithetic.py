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

* **Action identity across the pair — half of it retires in M5.** Through M4b
  this was one conjunction: the two halves *see the same observation* and *realise
  the same trade*. M5 separates them, because only the first is a claim about the
  method.

  The **observation** half retires. The mirror's signal is the primary's negated
  (:class:`NegatedSignal`), so the halves disagree about ``s`` by construction —
  which is what §9 predicted would end this check. What replaces it is a sharper
  statement of the same kind: **the halves differ in exactly one coordinate, and it
  is the coordinate the milestone is about.** Every other coordinate must still be
  bitwise equal and the signal coordinate must be the exact *negation*, not merely
  different — a mirror on a fresh signal path would differ too, and would be the
  M4a bug wearing M5's clothes. In a world with no informative signal the old
  bitwise equality is asserted verbatim, so three milestones keep the check they
  were built under.

  The **trade** half does not retire and is not scoped. The pair hands one action
  to both halves, so equal realised trades is a statement about the clip and the
  inventory path rather than about the policy, and it is as true in M5 as in M3.

  What genuinely ends is the *interpretation*: the average is no longer "the same
  policy replayed against mirrored shocks", because a policy shown the mirror's
  observation would not have chosen the primary's action. It is the primary's
  actions evaluated in the mirrored world — which is what M5 task 4 has to measure
  the consequences of, and is why this is recorded as a retirement rather than a
  refinement.
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
* **The signal is negated, exactly** (M5), and the halves are in one world. The
  mirror's signal coordinate must be the primary's negated, every other
  coordinate must be bitwise equal, and — checked once at construction, where a
  mismatch is loud and free — the two halves must carry the same impact model and
  the same three seed addresses. Negating the signal is not a preference: with
  ``xi = rho s + sqrt(1 - rho^2) e``, negating ``e`` alone gives the mirror
  ``rho s - sqrt(1 - rho^2) e``, which is **not** ``-xi``, and the shock-negation
  identity above goes red. Negating both gives ``-xi`` exactly, so the pairing's
  one remaining exact property survives the seam that ended its first one.

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
from temper.oracle import AlphaSignal


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

    Deliberately *not* a full :class:`~temper.oracle.signal.AlphaSignal`
    look-alike, for the reason :class:`NegatedDraws` is not a full
    :class:`~numpy.random.Generator` one. What is delegated is every pure value —
    the correlation, the lag, the moments, the quadrature — and what is negated is
    exactly one thing: :meth:`draw`. The sampling routines that compose a signal
    with an independent path (:meth:`draw_pair`, :meth:`shocks_from`) raise, because
    a caller supplying its own independent draws would get something that is not a
    mirror and would have no way to notice.

    **``correlation`` is not negated, and that is the crux.** The env composes its
    shock as ``rho * s + sqrt(1 - rho^2) * e``. The mirror wants ``-xi``, and it
    gets it because both inputs are negated at the same ``rho``:
    ``rho * (-s) + sqrt(1 - rho^2) * (-e) = -(rho * s + sqrt(1 - rho^2) * e)``.
    Negating ``rho`` instead would flip the model rather than the draw, and would
    produce a mirror in a different world at the same seed address — the M4a bug's
    shape, one seam along.
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

    **The signal stream is the third, and it is the one that is mirrored** — the
    opposite of liquidity, for a reason that is arithmetic rather than taste. The
    env composes each shock as ``rho * s + sqrt(1 - rho^2) * e``. Only the price
    generator is proxied by :class:`NegatedDraws`, so a mirror sharing the
    primary's signal would realise ``rho * s - sqrt(1 - rho^2) * e``, which is not
    ``-xi``, and the pair's shock-negation identity would go red on the first
    step of any signal-bearing world. Negating the signal as well gives ``-xi``
    exactly.

    So the choice is forced, and what it costs is the *action* identity: the two
    halves see signals of opposite sign, act differently, and the assertion that
    they act identically retires here (scoped, not deleted — see the module
    docstring). Given a choice between the pairing's exact property and its
    derived one, this keeps the exact one.

    The wrap is applied **only when the signal is informative**, so a world with no
    signal — M0 through M4b, and M5's own uninformative controls — gets the very
    same :class:`~temper.env.signal.SignalStream` object it always did, and this
    constructor is provably unchanged there rather than merely equivalent.
    """
    if not isinstance(env, ExecutionEnv):
        raise TypeError(f"mirror_of takes a raw ExecutionEnv, got {type(env)!r}")
    root_seed, pool, stream = env.seed_address
    signal = env.signal
    if signal.informative:
        signal = SignalStream(
            signal=NegatedSignal(signal.signal), pool=signal.pool, index=signal.index
        )
    return MirrorEnv(
        env.market,
        env.order_size,
        env.lambda_risk,
        temporary_impact=env.temporary_impact,
        liquidity=env.liquidity,
        signal=signal,
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


def _observation_disagreement(primary, mirror, *, mirrors_signal: bool) -> str | None:
    """What, if anything, is wrong with how the two halves' observations relate.

    ``None`` means they relate the way the pairing requires. Which relation that
    is depends on the world, and both are exact:

    * **No informative signal** — bitwise equal, M1a through M4b's assertion
      verbatim. Retiring it here would have thrown away a live check on three
      worlds to accommodate a fourth.
    * **An informative signal** — every coordinate but the last bitwise equal, and
      the last the exact negation. Stronger than "they differ": a mirror on a
      *fresh* signal path would differ too, and would be the M4a bug wearing M5's
      clothes.
    """
    a, b = np.asarray(primary), np.asarray(mirror)
    if a.shape != b.shape:
        return f"observation shapes differ ({a.shape} vs {b.shape})"
    if not mirrors_signal:
        if np.array_equal(a, b):
            return None
        return (
            f"the observation depends on the shocks ({a} vs {b}). Analytic grading "
            "and antithetic cancellation both assume a price-free observation"
        )
    if not np.array_equal(a[:-1], b[:-1]):
        return (
            f"the halves disagree on a coordinate that is not the signal ({a} vs "
            f"{b}); the signal is the only place a mirrored pair may differ"
        )
    if not np.array_equal(b[-1], -a[-1]):
        return (
            f"the mirror's signal {b[-1]!r} is not the exact negation of the "
            f"primary's {a[-1]!r}; the mirror is on a different signal path rather "
            "than the reflected one"
        )
    return None


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
        # Whether the halves are entitled to disagree, and about exactly what.
        # Read off the *primary* once: a mirror that had somehow been built
        # signal-free would then fail the per-step check rather than quietly
        # agreeing with a weaker rule.
        self.mirrors_signal = env.signal.informative
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
        wrong = _observation_disagreement(
            observation, mirrored, mirrors_signal=self.mirrors_signal
        )
        if wrong is not None:
            # The phrase "different observations" is load-bearing: M4b's
            # guarantee suite matches on it to show that a mirror on a different
            # liquidity path is refused before a step, and that behaviour is
            # unchanged here. The wording is part of the contract.
            raise PairDiverged(f"pair halves reset to different observations: {wrong}")
        self._primary_return = 0.0
        self._mirror_return = 0.0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        mirrored, mirror_reward, m_terminated, m_truncated, m_info = self.mirror.step(
            action
        )

        # -- the structural checks, every step -------------------------------
        wrong = _observation_disagreement(
            observation, mirrored, mirrors_signal=self.mirrors_signal
        )
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
        if m_info[SHOCK_KEY] != -info[SHOCK_KEY]:
            raise PairDiverged(
                f"the mirror's shock {m_info[SHOCK_KEY]!r} is not the exact "
                f"negation of the primary's {info[SHOCK_KEY]!r}"
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
