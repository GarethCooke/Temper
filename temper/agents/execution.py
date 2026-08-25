"""The agent's side of the execution boundary: action parameterisation and scaling.

Three small pieces, and the reason each exists is the whole of M2 tasks 4 and 5.

**The action is a fraction of remaining inventory** (:class:`FractionAction`),
squashed to ``[0, 1]``, converted to shares against the inventory the
*observation* reports. Not shares directly: a share-denominated action makes the
reachable set depend on how much the agent has already sold, so the same network
output means different things at different points of the same episode. Under the
fraction parameterisation TWAP is the fixed sequence ``1/13, 1/12, ..., 1`` —
a constant-free, state-independent target — so the baseline is exactly
representable and "the agent converged to something better than TWAP" cannot be
an artefact of TWAP being hard to express. ``tests/test_m2_action_space.py``
pins that sequence against :func:`~temper.oracle.schedules.twap_trajectory`.

The env's own contract is untouched: it still receives shares, still clips to
``[0, remaining]``, and still force-liquidates the final bin (constitution §4).
The wrapper converts; it does not negotiate.

**Reward scaling is one committed affine constant** (:class:`RewardScale`) and
never a running normaliser. Running statistics make the reward non-stationary
and seed-dependent, which is objective drift by the back door (invariant 7); and
because the same constant is applied at train and at eval — and because the
graded metric is analytic on the *unscaled* objective through
:func:`~temper.oracle.cost.schedule_moments` regardless — the scaling provably
cannot move what is being rediscovered.

**A trained network becomes a policy** (:class:`PPOPolicy`) exposing
``act(observation)`` in shares, so it runs through
:func:`~temper.eval.rollout.run_episode` on exactly the code path TWAP and the
two AC schedules run through (constitution §5).

Nothing here can see the price path. The observation is ``(time left, inventory
left)`` and this module never names the env's shock key, which
``tests/test_repo_invariants.py`` enforces statically over the whole of
``temper/agents/``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from gymnasium import Env, Wrapper, spaces

from temper.agents.ppo import AGENT_DTYPE, Agent
from temper.env import ExecutionEnv, LiquidityStream, SignalStream, TemporaryImpact
from temper.oracle import Market
from temper.seeding import DIFFERENTIAL_POOL

#: The agent's action space: the fraction of *remaining* inventory to execute
#: this bin. One number, on the unit interval, for every bin of every case.
FRACTION_SPACE = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)


def as_fraction(action) -> float:
    """Squash a raw action to ``[0, 1]``.

    A clip, not a sigmoid. The policy's density lives on the unsquashed action
    (see :class:`~temper.agents.ppo.GaussianHead`), so the squash is a property
    of the environment boundary rather than of the distribution — and a clip
    reaches the endpoints, which a sigmoid does not. The final bin's TWAP
    fraction is exactly ``1``.
    """
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    if value.size != 1:
        raise ValueError(f"the fraction action is one number, got shape {value.shape}")
    return float(np.clip(value[0], 0.0, 1.0))


def fraction_to_shares(action, observation, order_size: float) -> float:
    """Shares to execute, from a fraction and the inventory the observation reports.

    The single conversion, shared by the training wrapper and the evaluation
    policy. Two copies of this line would be two action spaces that agree until
    they do not — and the failure would look like a train/eval generalisation
    gap rather than like the arithmetic bug it is.

    Inventory comes from ``observation[1]``, the fraction of the parent order
    still to be worked, because that is what the *policy* was shown. Reading the
    env's internal inventory instead would let the boundary act on information
    the agent does not have.
    """
    remaining = float(np.asarray(observation, dtype=np.float64).reshape(-1)[1])
    return as_fraction(action) * remaining * order_size


def twap_fractions(n_bins: int) -> np.ndarray:
    """``1/N, 1/(N-1), ..., 1`` — TWAP under this parameterisation.

    Flat participation is *not* a constant fraction: selling a thirteenth of
    what is left every bin leaves inventory decaying geometrically, not
    linearly. The fraction that keeps the run-down linear is one over the number
    of bins remaining, which reaches ``1`` in the final bin — the same terminal
    condition the env enforces anyway.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    return 1.0 / np.arange(n_bins, 0, -1, dtype=np.float64)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class FractionAction(Wrapper):
    """Re-express :class:`~temper.env.ExecutionEnv`'s action as a fraction.

    A full :class:`~gymnasium.Wrapper` rather than an ``ActionWrapper`` because
    the conversion needs the current observation, which ``ActionWrapper.action``
    is not given. The observation, reward, termination and ``info`` all pass
    through untouched.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        # `unwrapped`, not `env`: an estimator wrapper (the control variate, the
        # antithetic pair) may sit between this and the ExecutionEnv, and the
        # parent order is the raw env's to state.
        self.order_size = float(env.unwrapped.order_size)
        self.action_space = FRACTION_SPACE
        self._observation = np.zeros(2, dtype=np.float64)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._observation = np.asarray(observation, dtype=np.float64)
        return observation, info

    def step(self, action):
        shares = fraction_to_shares(action, self._observation, self.order_size)
        observation, reward, terminated, truncated, info = self.env.step(shares)
        self._observation = np.asarray(observation, dtype=np.float64)
        return observation, reward, terminated, truncated, info


class RewardScale(Wrapper):
    """Multiply the reward by a fixed constant. That is the entire class.

    It exists to be a *named, greppable, committed* thing rather than a
    multiplication buried in a training loop — and so that
    ``tests/test_m2_action_space.py`` can pin the two properties invariant 7
    needs: the constant is affine and it is the same at train and eval. There is
    no ``update``, no running mean, and no state, which is the point.
    """

    def __init__(self, env: Env, scale: float) -> None:
        super().__init__(env)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"reward scale must be finite and positive, got {scale}")
        self.scale = float(scale)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.scale * float(reward), terminated, truncated, info


# ---------------------------------------------------------------------------
# The env factory the training driver hands to `ppo.train`
# ---------------------------------------------------------------------------


def execution_env_factory(
    market: Market,
    order_size: float,
    lambda_risk: float,
    *,
    root_seed: int,
    pool: str = DIFFERENTIAL_POOL,
    stream_index: int = 0,
    reward_scale: float = 1.0,
    reward_wrapper: Callable[[Env], Env] | None = None,
    temporary_impact: TemporaryImpact | None = None,
    liquidity: LiquidityStream | None = None,
    signal: SignalStream | None = None,
) -> Callable[[], Env]:
    """A thunk building one training env at one seed address.

    The address is fixed *here*, by the caller, and never by the training loop:
    :func:`~temper.agents.ppo.train` only ever calls the argument-free
    ``reset()``, so each parallel env spends exactly the stream its factory
    named (constitution invariant 5).

    `reward_wrapper` is the seam an estimator enters through — M2's sanctioned
    control variate, M3's antithetic pair — an ``Env -> Env`` callable applied to
    the **raw** env, below the fraction action and below the scaling. Below the
    scaling so the estimator works in the env's own bps and the scale is applied
    once to whatever is left; below the fraction action so an estimator that
    steps a second env (the antithetic mirror) hands it the same *shares* the
    primary received, with the fraction converted exactly once. It is a
    parameter rather than an import because every estimator so far reads the
    env's published price shock, which no module under ``temper/agents/`` may
    name; the experiment driver passes it in.

    `temporary_impact` is the world the agent trains in, and it arrives the same
    way and for a related reason: ``None`` is Phase 1, and a Phase-2 world is
    named by the config the driver read rather than defaulted to by the training
    path (constitution §4). `liquidity` is M4b's second seam, arriving the same
    way again — a law bound to the pool it draws from, ``None`` being ``L = 1``.
    Two seams now default to Phase 1 and **every env this factory builds gets
    both**, which is M4a's §9 lesson stated as a signature rather than as a hope.
    """

    def make() -> Env:
        env: Env = ExecutionEnv(
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
        if reward_wrapper is not None:
            env = reward_wrapper(env)
        env = FractionAction(env)
        if reward_scale != 1.0:
            env = RewardScale(env, reward_scale)
        return env

    return make


# ---------------------------------------------------------------------------
# A trained network, as a gradable policy
# ---------------------------------------------------------------------------


class PPOPolicy:
    """A trained :class:`~temper.agents.ppo.Agent` behind the baselines' interface.

    ``act`` returns *shares*, so this object is interchangeable with
    :class:`~temper.agents.baselines.SchedulePolicy` everywhere: the same
    :func:`~temper.eval.rollout.run_episode`, the same env, the same metric
    registry. The action is the policy distribution's mean, squashed and
    converted exactly as :class:`FractionAction` does during training — through
    :func:`fraction_to_shares`, so train and eval cannot drift apart.

    Determinism is load-bearing rather than incidental. The observation carries
    no price, inventory evolves only from actions, and the mean action is a
    function of the observation alone; so the schedule this policy induces is
    the same schedule under every shock stream, and can be graded analytically
    by :func:`~temper.oracle.cost.schedule_moments` with no Monte-Carlo error at
    all. ``tests/test_m2_grading.py`` asserts the bitwise version of that claim.
    """

    def __init__(self, agent: Agent, order_size: float, name: str = "ppo") -> None:
        self.agent = agent
        self.order_size = float(order_size)
        self.name = name
        self.agent.eval()

    def reset(self) -> None:
        """Stateless between episodes: the policy is a function of the observation."""

    def fraction(self, observation) -> float:
        """The squashed mean action for this observation — the policy itself."""
        features = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), dtype=AGENT_DTYPE
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.agent.deterministic_action(features)
        return as_fraction(action.numpy()[0])

    def act(self, observation) -> float:
        return fraction_to_shares(
            self.fraction(observation), observation, self.order_size
        )

    def __repr__(self) -> str:
        return f"PPOPolicy({self.name!r}, X={self.order_size:g})"


class FractionPolicy:
    """A fixed fraction-of-remaining sequence, replayed as a policy.

    The counterpart of :class:`~temper.agents.baselines.SchedulePolicy` in the
    agent's coordinates, and the thing that makes "TWAP is representable" a
    statement one can execute rather than assert: build it from
    :func:`twap_fractions` and the realised trajectory is TWAP's.
    """

    def __init__(self, fractions, order_size: float, name: str) -> None:
        values = np.asarray(fractions, dtype=np.float64)
        if values.ndim != 1 or values.size < 1:
            raise ValueError(f"fractions must be 1-D and non-empty, got {values.shape}")
        self.fractions = values
        self.order_size = float(order_size)
        self.name = name
        self._n_bins = int(values.size)

    def reset(self) -> None:
        """Nothing to forget: the bin index comes out of the observation's clock."""

    def act(self, observation) -> float:
        clock = float(np.asarray(observation, dtype=np.float64).reshape(-1)[0])
        bin_index = self._n_bins - int(round(clock * self._n_bins))
        if not 0 <= bin_index < self._n_bins:
            raise ValueError(
                f"observation clock {clock!r} is outside the {self._n_bins} bins"
            )
        return fraction_to_shares(
            self.fractions[bin_index], observation, self.order_size
        )

    def __repr__(self) -> str:
        return f"FractionPolicy({self.name!r}, {self._n_bins} bins)"
