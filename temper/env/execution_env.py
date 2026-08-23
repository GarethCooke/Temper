"""``ExecutionEnv`` — the execution simulator (constitution §4).

One episode liquidates a parent order of ``X`` shares over ``n_bins`` intervals of
``dt`` hours. Prices follow arithmetic Brownian motion; trading pays linear
permanent impact, the half-spread on every share, and a temporary concession from
an **injected** impact model (:mod:`temper.env.impact`). The default is Phase 1's
linear tangent ``eta_tilde``; M4a's power-law world is the same env with a
different model, not a different env.

The whole point of this module is that it is **not** the oracle. Nothing here
imports a trajectory, a kappa or a closed-form moment: the env accumulates cost
bin by bin from the price path, and M1's differential then checks that the
moments of what it produces match what :mod:`temper.oracle` predicts. Sharing
code between the two would collapse that check into a tautology.

Price path and the cost decomposition
-------------------------------------
Everything is measured against the arrival price, in basis points, with the
arrival price normalised to ``1``. Within bin ``k``:

.. code::

    walk_k     = walk_{k-1} + sigma_bin * BPS * eps_k          eps_k ~ N(0, 1)
    own_k      = gamma * sigma * BPS / v_hourly * n_k          this bin's permanent drift
    temp_k     = temporary_impact(n_k)                         per share; the world's model
    price_k    = walk_k - perm_{<k} - own_k / 2 - temp_k - half_spread
    shortfall_k = -(n_k / X) * price_k                         bps of notional

The bin's own permanent drift is charged at its midpoint — half of it moves the
price the bin itself executes against, all of it moves the price every later bin
sees. The shock ``eps_k`` lands *before* bin ``k`` executes, so it is carried by
every share still held at the start of the bin; summing the shortfall gives a
variance of ``(sigma_bin * BPS)^2 * sum_k (x_k / X)^2`` over inventory *before*
each bin, which is the convention
:func:`~temper.oracle.cost.shortfall_variance_bps2` encodes.

The reward is §4's, frozen: ``r_k = -shortfall_k - lambda * sigma_bin^2 * BPS^2 *
(x_k / X)^2``. Summed over an episode that is exactly ``-(realised shortfall +
lambda * V)`` — the same functional the oracle minimises (invariant 7), which
``tests/test_env_identities.py`` pins mechanically rather than by assertion in a
docstring.

Seeding
-------
Randomness enters only by *pool address*: ``(root_seed, pool, stream_index)``
resolved through :mod:`temper.seeding`. There is deliberately no way to hand the
env a raw integer seed, because a raw seed could collide with a stream a
committed training or evaluation result is addressed by (invariant 5).
``reset(seed=i)`` therefore means "restart at stream ``i`` of this env's pool",
and plain ``reset()`` draws the next episode from the current stream.
"""

from __future__ import annotations

import math

import numpy as np
from gymnasium import Env, spaces
from numpy.random import Generator

from temper.env.impact import TemporaryImpact, linear_temporary
from temper.env.liquidity import DETERMINISTIC_LIQUIDITY, LiquidityStream
from temper.oracle import BPS, Market
from temper.seeding import DIFFERENTIAL_POOL, pool_rng

#: ``info`` key the terminal step publishes its episode summary under. Not
#: ``"episode"``: gymnasium's own ``RecordEpisodeStatistics`` wrapper writes that
#: key, and a wrapper silently overwriting the differential's numbers is exactly
#: the kind of quiet substitution this milestone exists to prevent.
EPISODE_KEY = "episode_summary"

#: ``info`` key the realised price shock is published under — and the *only* route
#: to it. The observation is `(time left, inventory left)` and nothing else
#: (constitution §4: rediscovery must not smuggle in signal), so a policy that
#: could see the shock would be seeing the future of its own price path. The key
#: is a named constant rather than a literal so that
#: ``tests/test_repo_invariants.py`` can statically reject the literal everywhere
#: outside this package: one greppable name, and every read of it auditable.
SHOCK_KEY = "walk_bps"

#: ``info`` key the bin's realised liquidity multiplier is published under.
#:
#: Deliberately **not** quarantined the way :data:`SHOCK_KEY` is, and the contrast
#: is the point. The shock is the future of the episode's own price path and a
#: policy that could see it would be cheating; the multiplier is the *current*
#: state of the market the agent is trading in, it is in the observation on
#: purpose, and M4b's whole claim is that reacting to it is worth something. The
#: key exists so the antithetic pair can assert that both halves saw the same
#: liquidity — the third per-step identity — and so the differential can measure
#: the realised draws against the oracle's closed-form moments.
LIQUIDITY_KEY = "liquidity_multiplier"


def _as_shares(action) -> float:
    """Coerce an action to a share count.

    Accepts a Python float or a one-element array — the action space is
    ``Box(shape=(1,))``, but a policy that returns a plain float is not doing
    anything wrong. Any other shape is a bug in the caller, not a shape to guess
    the intent of.
    """
    if isinstance(action, np.ndarray):
        if action.size != 1:
            raise ValueError(f"action must be one number, got shape {action.shape}")
        return float(action.reshape(-1)[0])
    return float(action)


class ExecutionEnv(Env):
    """Sell ``order_size`` shares over ``market.n_bins`` bins under Phase-1 dynamics.

    Parameters
    ----------
    market:
        Symbol parameters and the execution grid (:class:`~temper.oracle.Market`).
    order_size:
        Parent order ``X``, in shares. Under the default linear model it also
        fixes ``eta_tilde``, the tangent the temporary impact is taken at — a
        property of the *order*, frozen for the episode, not something the
        realised schedule moves.
    lambda_risk:
        Risk aversion in the frozen objective ``E + lambda * V``, with ``E`` in
        bps and ``V`` in bps².
    temporary_impact:
        The world's temporary-impact model
        (:mod:`temper.env.impact`). ``None`` builds
        :func:`~temper.env.impact.linear_temporary` — Phase 1, exactly as this
        env built it before the model became injectable. A Phase-2 world is
        never inherited: it has to be named, here or in a config.
    liquidity:
        M4b's second seam: a :class:`~temper.env.liquidity.LiquidityStream`, which
        is a per-bin multiplier on ``v_hourly`` **bound to the seed pool it draws
        from**. ``None`` is
        :data:`~temper.env.liquidity.DETERMINISTIC_LIQUIDITY` — ``L = 1``, no
        randomness consumed, the market every milestone through M4a ran in. Same
        rule as the impact model and for the same reason; there are now two ways
        to inherit a Phase-2 world by omission and ``tests/test_repo_invariants.py``
        closes both.

        A **stochastic** stream changes two visible things: the observation grows
        a third coordinate, ``log L_k``, and the temporary charge is paid at the
        bin's own liquidity. It changes nothing else — the shock model, permanent
        impact, the half-spread and the frozen objective are all Phase 1's.
    root_seed, pool, stream_index:
        The seed address (see the module docstring). The liquidity stream shares
        `root_seed` and `stream_index` and differs only in its pool, so the two
        noise sources cannot collide and neither can move the other.

    Actions are shares to execute this interval, clipped to ``[0, remaining]``.
    The final bin force-liquidates whatever is left and charges it like any other
    trade, which is the discrete form of the closed form's ``x_N = 0`` terminal
    constraint: the agent cannot dodge the boundary by simply not trading.

    **One env, one ``step``.** The temporary charge is injected rather than
    subclassed. A Phase-2 env with its own loop would duplicate the twenty lines
    the whole differential is a claim about — including
    :attr:`step_count`, whose "``N_sim`` episodes went through *this* loop"
    (invariant 6) stops being falsifiable the moment there are two of them.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        market: Market,
        order_size: float,
        lambda_risk: float,
        *,
        temporary_impact: TemporaryImpact | None = None,
        liquidity: LiquidityStream | None = None,
        root_seed: int,
        pool: str = DIFFERENTIAL_POOL,
        stream_index: int = 0,
    ) -> None:
        if order_size <= 0.0:
            raise ValueError(f"order_size must be positive, got {order_size}")
        if lambda_risk < 0.0:
            raise ValueError(f"lambda_risk must be non-negative, got {lambda_risk}")

        self.market = market
        self.order_size = float(order_size)
        self.lambda_risk = float(lambda_risk)
        self.temporary_impact = (
            linear_temporary(market, self.order_size)
            if temporary_impact is None
            else temporary_impact
        )
        self.liquidity = (
            DETERMINISTIC_LIQUIDITY if liquidity is None else liquidity
        )
        if not isinstance(self.liquidity, LiquidityStream):
            raise TypeError(
                "liquidity is a LiquidityStream — a law bound to the pool it draws "
                f"from — got {type(self.liquidity)!r}"
            )

        self._root_seed = int(root_seed)
        self._pool = pool
        self._stream_index = int(stream_index)
        self._rng: Generator | None = None
        # A *second* generator, at a second pool. Never the same object as
        # `_rng`, never derived from it: a liquidity draw taken out of the price
        # generator would shift every downstream shock and silently un-reproduce
        # every committed Phase-1 and M4a number.
        self._liquidity_rng: Generator | None = None
        self._multipliers = np.ones(market.n_bins, dtype=np.float64)
        self._log_multipliers = np.zeros(market.n_bins + 1, dtype=np.float64)

        n_bins = market.n_bins
        self._n_bins = n_bins
        self._last_bin = n_bins - 1

        # Per-share coefficients, all in bps, all precomputed: `step` runs tens of
        # millions of times in the deep differential tier. The temporary charge is
        # bound to the grid once here for the same reason.
        params = market.params
        self._temporary_bps = self.temporary_impact.bind(market)
        self._permanent_per_share = params.gamma * params.sigma * BPS / market.v_hourly
        self._shock_bps = market.sigma_bin * BPS
        self._variance_bps2 = self._shock_bps**2
        self._half_spread = params.half_spread

        # Time remaining fraction at each bin boundary, including the terminal 0.
        self._time_fractions = 1.0 - np.arange(n_bins + 1, dtype=np.float64) / n_bins

        self.action_space = spaces.Box(
            low=0.0, high=self.order_size, shape=(1,), dtype=np.float64
        )
        # Two coordinates in the deterministic world — bitwise the Phase-1 and M4a
        # observation, which is what lets their committed seeds retrain through
        # this seam unchanged — and three when liquidity is a noise source, the
        # third being `log L_k` for the bin about to execute. The observation grows
        # once per milestone and only where the milestone needs it (§7).
        if self.liquidity.stochastic:
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, -np.inf]),
                high=np.array([1.0, 1.0, np.inf]),
                shape=(3,),
                dtype=np.float64,
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(2,), dtype=np.float64
            )

        self._trajectory = np.empty(n_bins + 1, dtype=np.float64)
        # Monotone over the env's whole life — deliberately *not* cleared by
        # `reset`. See the `step_count` property.
        self._step_count = 0
        self._step_index: int | None = None
        self._inventory = 0.0
        self._walk = 0.0
        self._permanent = 0.0
        self._proceeds = 0.0
        self._shortfall_total = 0.0
        self._penalty_total = 0.0
        self._reward_total = 0.0

    # -- gymnasium API ------------------------------------------------------

    @property
    def seed_address(self) -> tuple[int, str, int]:
        """``(root_seed, pool, stream_index)`` — what a result must record."""
        return (self._root_seed, self._pool, self._stream_index)

    @property
    def liquidity_address(self) -> tuple[int, str, int]:
        """``(root_seed, liquidity pool, stream_index)`` — the *second* address.

        Shares the root seed and the index with :attr:`seed_address` and differs
        only in the pool, which is the whole disjointness argument in one line.
        """
        return (
            self._root_seed,
            self.liquidity.pool,
            self.liquidity.stream_index(self._stream_index),
        )

    @property
    def multipliers(self) -> np.ndarray:
        """This episode's realised per-bin liquidity, drawn at ``reset``.

        Published so the estimator can assert that a mirror saw the *same*
        liquidity, and so the grader can evaluate ``E[cost | L]`` at the path the
        policy actually faced. A copy, because a caller holding the env's own
        buffer would see it change under them at the next reset.
        """
        return self._multipliers.copy()

    @property
    def cost_encoding(self) -> str:
        """Which cost functional this env charges — the world, named.

        M4a's rule is that a metric grades the world that charges it, and this is
        the half of the pairing the env states. :mod:`temper.eval.grading` refuses
        to compute anything until this and the metric's ``encoding`` agree, which
        is strictly stronger than the flat refusal it replaced: the old rule
        banned the power-law encoding outright and so could not have caught the
        failure that is now live — a *linear* metric grading a power-law env.
        """
        return self.temporary_impact.encoding

    @property
    def step_count(self) -> int:
        """Calls to :meth:`step` over this env's whole life. Never reset.

        The differential's claim is that ``N_sim`` episodes went through *this*
        loop, one bin at a time — "no vectorised side-channel", because the loop
        is the thing under test. That is otherwise an unfalsifiable statement
        about the harness: a future session could add a batched path that
        computes the same costs a hundred times faster and every band would stay
        green. Here the claim is arithmetic instead. Each tier asserts the
        counter advanced by exactly ``N_sim * n_bins``
        (``tests/test_differential.py``), so a shortcut round the loop shows up
        as a missing count rather than as a pleasant surprise on the wall clock.
        """
        return self._step_count

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode; `seed` selects a stream index within the pool.

        Passing `seed` rewinds to the start of that stream, so the same
        ``(config, seed)`` replays the same episodes. Omitting it draws the next
        episode from the stream already in flight.
        """
        if seed is not None:
            if seed < 0:
                raise ValueError(f"seed is a stream index and must be >= 0, got {seed}")
            self._stream_index = int(seed)
            self._rng = None
            self._liquidity_rng = None
        if self._rng is None:
            self._rng = pool_rng(self._root_seed, self._pool, self._stream_index)
            self.np_random = self._rng
        if self._liquidity_rng is None:
            self._liquidity_rng = self.liquidity.generator(
                self._root_seed, self._stream_index
            )

        # The whole episode's liquidity, drawn in one block from the *liquidity*
        # generator. The publication ordering is the thing that matters and it is
        # M1a's ordering one seam along: `L_k` is revealed **before** bin `k`
        # executes, so `reset` publishes `L_0` and the observation after step `k`
        # carries `L_{k+1}`. That ordering is invisible in the code and load-bearing
        # for the dynamic program's state definition — the DP's `(k, x_k, L_k)` is
        # sufficient only if the agent sees `L_k` in time to act on it.
        #
        # Blockwise rather than one draw per step because the two are identical for
        # a numpy Generator (`standard_normal(n)` fills the same sequence as `n`
        # scalar calls) and the block makes "the same address gives the same
        # liquidity path" a property a test can read off in one line.
        self._multipliers = self.liquidity.law.draw(
            self._liquidity_rng, self._n_bins
        )
        self._log_multipliers[: self._n_bins] = np.log(self._multipliers)
        self._log_multipliers[self._n_bins] = 0.0

        self._step_index = 0
        self._inventory = self.order_size
        self._walk = 0.0
        self._permanent = 0.0
        self._proceeds = 0.0
        self._shortfall_total = 0.0
        self._penalty_total = 0.0
        self._reward_total = 0.0
        self._trajectory[0] = self.order_size

        return self._observation(), {"step": 0, "inventory": self.order_size}

    def step(self, action):
        """Execute `action` shares this interval and advance one bin."""
        step_index = self._step_index
        if step_index is None:
            raise RuntimeError("step() before reset()")
        if step_index >= self._n_bins:
            raise RuntimeError("step() after the episode terminated; call reset()")

        inventory = self._inventory
        shares = _as_shares(action)
        if math.isnan(shares):
            raise ValueError("action is NaN")
        self._step_count += 1

        if step_index == self._last_bin:
            # The terminal constraint x_N = 0: whatever is left goes now, charged
            # exactly like any other trade.
            shares = inventory
        else:
            shares = min(max(shares, 0.0), inventory)

        self._walk += self._shock_bps * self._rng.standard_normal()
        own_drift = self._permanent_per_share * shares
        liquidity = self._multipliers[step_index]
        price_bps = (
            self._walk
            - self._permanent
            - 0.5 * own_drift
            - self._temporary_bps(shares, liquidity)
            - self._half_spread
        )
        self._permanent += own_drift

        weight = shares / self.order_size
        shortfall = -weight * price_bps
        penalty = (
            self.lambda_risk
            * self._variance_bps2
            * (inventory / self.order_size) ** 2
        )
        reward = -(shortfall + penalty)

        self._proceeds += shares * (1.0 + price_bps / BPS)
        self._inventory = inventory - shares
        self._shortfall_total += shortfall
        self._penalty_total += penalty
        self._reward_total += reward

        step_index += 1
        self._step_index = step_index
        self._trajectory[step_index] = self._inventory
        terminated = step_index == self._n_bins

        info = {
            "step": step_index - 1,
            "shares": shares,
            "inventory": self._inventory,
            "shortfall_bps": shortfall,
            "penalty_bps": penalty,
            "execution_price_bps": price_bps,
            # The cumulative price shock this bin executed against. Published so a
            # test can subtract the noise off a single episode and compare what
            # remains against the oracle's E[cost] *exactly*, instead of only
            # statistically — see tests/test_noise_identity.py. `info` is the only
            # route to it: it is deliberately absent from the observation, and
            # `SHOCK_KEY`'s docstring says why.
            SHOCK_KEY: self._walk,
            # The bin's realised liquidity. In `info` as well as in the
            # observation because the estimator has to be able to *assert* that a
            # mirror saw the same market, and reading that off a log inside an
            # observation vector would be a check on the encoding rather than on
            # the draw.
            LIQUIDITY_KEY: liquidity,
        }
        if terminated:
            info[EPISODE_KEY] = self._episode_summary()

        return self._observation(), reward, terminated, False, info

    # -- internals ----------------------------------------------------------

    def _observation(self) -> np.ndarray:
        """``(time left, inventory left)``, plus ``log L_k`` when liquidity is a world.

        The third coordinate is the multiplier for the bin **about to execute**,
        so a policy sees the market it is about to trade in rather than the one it
        has just traded through. At the terminal boundary there is no next bin and
        the entry is ``0.0``; that observation is returned only after the episode
        has terminated and no policy acts on it.

        ``log L`` rather than ``L`` because the law is lognormal, so the log is the
        coordinate the distribution is symmetric and unit-scaled in — at
        ``sigma_log = 0.5`` it sits in roughly ``[-1.6, 1.4]``, which is the same
        order as the other two coordinates. No running normaliser is involved and
        none is permitted (``tests/test_repo_invariants.py``): the scaling is a
        property of the committed law, not of the data seen so far.
        """
        if not self.liquidity.stochastic:
            return np.array(
                (
                    self._time_fractions[self._step_index],
                    self._inventory / self.order_size,
                ),
                dtype=np.float64,
            )
        return np.array(
            (
                self._time_fractions[self._step_index],
                self._inventory / self.order_size,
                self._log_multipliers[self._step_index],
            ),
            dtype=np.float64,
        )

    def _episode_summary(self) -> dict:
        """What the episode actually did, for the eval harness and the tests.

        ``cost_bps`` is rebuilt from the realised proceeds — the price path — and
        so is an independent route to the same number as ``shortfall_bps``, which
        sums the per-step charges. M1's first identity test is exactly that the
        two agree.
        """
        return {
            "trajectory": self._trajectory.copy(),
            "liquidity": self._multipliers.copy(),
            "cost_bps": BPS * (1.0 - self._proceeds / self.order_size),
            "shortfall_bps": self._shortfall_total,
            "penalty_bps": self._penalty_total,
            "reward": self._reward_total,
        }
