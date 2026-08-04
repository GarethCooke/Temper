"""``ExecutionEnv`` — the Phase-1 execution simulator (constitution §4).

One episode liquidates a parent order of ``X`` shares over ``n_bins`` intervals of
``dt`` hours. Prices follow arithmetic Brownian motion; trading pays linear
permanent impact, linear temporary impact at the tangent slope ``eta_tilde``, and
the half-spread on every share.

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
    temp_k     = eta_tilde * BPS / dt * n_k                    temporary, per share
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

from temper.oracle import BPS, Market, linearised_eta
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
        Parent order ``X``, in shares. Also fixes ``eta_tilde``, the tangent the
        linear temporary impact is taken at — a property of the *order*, frozen
        for the episode, not something the realised schedule moves.
    lambda_risk:
        Risk aversion in the frozen objective ``E + lambda * V``, with ``E`` in
        bps and ``V`` in bps².
    root_seed, pool, stream_index:
        The seed address (see the module docstring).

    Actions are shares to execute this interval, clipped to ``[0, remaining]``.
    The final bin force-liquidates whatever is left and charges it like any other
    trade, which is the discrete form of the closed form's ``x_N = 0`` terminal
    constraint: the agent cannot dodge the boundary by simply not trading.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        market: Market,
        order_size: float,
        lambda_risk: float,
        *,
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
        self.eta_tilde = linearised_eta(market, self.order_size)

        self._root_seed = int(root_seed)
        self._pool = pool
        self._stream_index = int(stream_index)
        self._rng: Generator | None = None

        n_bins = market.n_bins
        self._n_bins = n_bins
        self._last_bin = n_bins - 1

        # Per-share coefficients, all in bps, all precomputed: `step` runs tens of
        # millions of times in the deep differential tier.
        params = market.params
        self._temporary_per_share = self.eta_tilde * BPS / market.dt
        self._permanent_per_share = params.gamma * params.sigma * BPS / market.v_hourly
        self._shock_bps = market.sigma_bin * BPS
        self._variance_bps2 = self._shock_bps**2
        self._half_spread = params.half_spread

        # Time remaining fraction at each bin boundary, including the terminal 0.
        self._time_fractions = 1.0 - np.arange(n_bins + 1, dtype=np.float64) / n_bins

        self.action_space = spaces.Box(
            low=0.0, high=self.order_size, shape=(1,), dtype=np.float64
        )
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
        if self._rng is None:
            self._rng = pool_rng(self._root_seed, self._pool, self._stream_index)
            self.np_random = self._rng

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
        price_bps = (
            self._walk
            - self._permanent
            - 0.5 * own_drift
            - self._temporary_per_share * shares
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
        }
        if terminated:
            info[EPISODE_KEY] = self._episode_summary()

        return self._observation(), reward, terminated, False, info

    # -- internals ----------------------------------------------------------

    def _observation(self) -> np.ndarray:
        """``(time remaining fraction, inventory remaining fraction)``."""
        return np.array(
            (
                self._time_fractions[self._step_index],
                self._inventory / self.order_size,
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
            "cost_bps": BPS * (1.0 - self._proceeds / self.order_size),
            "shortfall_bps": self._shortfall_total,
            "penalty_bps": self._penalty_total,
            "reward": self._reward_total,
        }
