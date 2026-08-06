"""Single-file PPO — adapted from CleanRL, readable line by line.

Attribution
-----------
Adapted from CleanRL's ``ppo_continuous_action.py`` and ``ppo.py`` (Huang et al.,
*CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning
Algorithms*, JMLR 23(274), 2022 — https://github.com/vwxyzjn/cleanrl, MIT
licence). The layer initialisation, the clipped surrogate, the clipped value
loss, the GAE recursion and the minibatch schedule are CleanRL's; they are
reproduced rather than imported because constitution §5 asks for an agent that
can be read line by line, and a framework call is not that.

What is deliberately *not* CleanRL
----------------------------------
* **No ``NormalizeReward``, no ``NormalizeObservation``.** Running statistics make
  the reward non-stationary and seed-dependent, which is objective drift by the
  back door (constitution invariant 7). Any scaling here is a single affine
  constant out of the committed config, applied identically at train and eval —
  see :class:`RewardScale` in :mod:`temper.agents.execution`.
* **One file, two action spaces.** CleanRL keeps a file per algorithm-variant;
  the discrete and continuous heads share this loop instead, because M2's smoke
  test is only evidence about the Temper agent if the Temper agent runs the same
  code. :class:`GaussianHead` and :class:`CategoricalHead` differ in eleven lines
  and nothing else does.
* **An explicit synchronous vector loop.** Rather than
  ``gymnasium.vector.SyncVectorEnv``, whose autoreset semantics changed in
  gymnasium 1.0 and whose bootstrapping-on-truncation behaviour is version
  dependent. The loop here resets on the same step it terminates and bootstraps
  truncated episodes explicitly, which is thirty lines and no version risk.

Determinism
-----------
:func:`train` seeds torch and numpy from integers the caller resolves through
:mod:`temper.seeding`, and never calls a global RNG that a caller did not seed.
Given the same ``(config, seed, env factory)`` it produces the same weights.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env, spaces
from torch.distributions import Categorical, Normal

#: The agent boundary is float32: small MLPs on CPU, and the env's float64 core
#: is untouched on the other side of the cast (``ARCHITECTURE.md`` §5).
AGENT_DTYPE = torch.float32


@dataclass(frozen=True)
class PPOConfig:
    """Every hyperparameter, in one committed object.

    Defaults are CleanRL's where CleanRL has one. They are *defaults*, not the
    milestone's values: the values a reported result was produced with live in
    ``configs/*.yaml`` and reach here through :meth:`from_mapping`, so a number
    in a results file can always be traced to a committed file rather than to a
    dataclass default that has since moved.
    """

    total_timesteps: int = 300_000
    num_envs: int = 8
    num_steps: int = 128
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 10
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    norm_adv: bool = True
    hidden_sizes: tuple[int, ...] = (64, 64)
    log_std_init: float = 0.0
    #: Cap on wall-clock seconds. A stop here is reported, never silently
    #: rounded up to "converged": :class:`TrainResult` carries `timed_out`.
    max_seconds: float | None = None
    #: Intra-op thread count for torch. Part of the *experiment*, not of the
    #: host, because it is the one input to a trained artefact that the seed
    #: address does not reach: torch's CPU reductions sum in an order that
    #: depends on how many threads share the work, PPO compounds that over
    #: hundreds of updates, and this objective is flat enough near its minimum
    #: for the difference to move the trajectory visibly. Left unpinned, "same
    #: config, same seed" reproduces only on a host with the same core count —
    #: which is invariant 1 holding by luck. ``None`` keeps torch's default.
    torch_threads: int | None = None

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)

    def __post_init__(self) -> None:
        if self.batch_size % self.num_minibatches:
            raise ValueError(
                f"batch size {self.batch_size} is not divisible by "
                f"{self.num_minibatches} minibatches"
            )
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must name at least one layer")

    @classmethod
    def from_mapping(cls, mapping: dict) -> "PPOConfig":
        """Build from a config block, rejecting keys that are not hyperparameters.

        Rejecting rather than ignoring: a typo'd key in a committed config would
        otherwise train at a default while the file claims otherwise, and the
        results JSON would record the file.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"unknown PPO hyperparameter(s) {', '.join(unknown)}; "
                f"known keys are {', '.join(sorted(known))}"
            )
        values = dict(mapping)
        if "hidden_sizes" in values:
            values["hidden_sizes"] = tuple(int(n) for n in values["hidden_sizes"])
        return cls(**values)

    def as_dict(self) -> dict:
        """A JSON-safe view for the results file."""
        return {
            f.name: (
                list(getattr(self, f.name))
                if isinstance(getattr(self, f.name), tuple)
                else getattr(self, f.name)
            )
            for f in fields(type(self))
        }


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    """CleanRL's orthogonal initialisation, verbatim."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(in_features: int, hidden: Sequence[int], out_features: int, out_std: float):
    """``tanh`` MLP with CleanRL's per-layer gains; the head gets `out_std`."""
    layers: list[nn.Module] = []
    size = in_features
    for width in hidden:
        layers += [layer_init(nn.Linear(size, width)), nn.Tanh()]
        size = width
    layers.append(layer_init(nn.Linear(size, out_features), std=out_std))
    return nn.Sequential(*layers)


class GaussianHead(nn.Module):
    """Continuous actions: a state-dependent mean and a state-independent log-std.

    The distribution is over the *unsquashed* action, and the squash to the
    environment's box is a clip applied outside the policy (CleanRL's
    ``ClipAction``). Keeping the density on the pre-clip variable is what makes
    the importance ratio exact — a tanh squash would need a Jacobian correction,
    and a truncated Gaussian would need a normalising constant that moves with
    the mean. It also keeps every point of the box reachable, including the
    corners: under Temper's parameterisation the last bin's TWAP fraction is
    exactly ``1``, and a policy that could only approach it asymptotically would
    make the baseline unrepresentable (M2 task 5).
    """

    def __init__(self, in_features: int, hidden, action_dim: int, log_std_init: float):
        super().__init__()
        self.mean = _mlp(in_features, hidden, action_dim, out_std=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def distribution(self, features: torch.Tensor) -> Normal:
        mean = self.mean(features)
        return Normal(mean, self.log_std.expand_as(mean).exp())

    def deterministic(self, features: torch.Tensor) -> torch.Tensor:
        return self.mean(features)

    @staticmethod
    def log_prob(distribution: Normal, action: torch.Tensor) -> torch.Tensor:
        return distribution.log_prob(action).sum(-1)

    @staticmethod
    def entropy(distribution: Normal) -> torch.Tensor:
        return distribution.entropy().sum(-1)


class CategoricalHead(nn.Module):
    """Discrete actions — the cheap second check (CartPole), same loop."""

    def __init__(self, in_features: int, hidden, n_actions: int, log_std_init: float):
        super().__init__()
        del log_std_init  # a categorical policy has no scale parameter
        self.logits = _mlp(in_features, hidden, n_actions, out_std=0.01)

    def distribution(self, features: torch.Tensor) -> Categorical:
        return Categorical(logits=self.logits(features))

    def deterministic(self, features: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.logits(features), dim=-1)

    @staticmethod
    def log_prob(distribution: Categorical, action: torch.Tensor) -> torch.Tensor:
        return distribution.log_prob(action)

    @staticmethod
    def entropy(distribution: Categorical) -> torch.Tensor:
        return distribution.entropy()


class Agent(nn.Module):
    """Actor and critic as two separate MLPs over the raw observation.

    Separate rather than a shared trunk: the observation is two numbers, the
    networks are tiny, and a shared trunk couples the value loss scale to the
    policy gradient for no capacity saving worth having.
    """

    def __init__(self, observation_space, action_space, config: PPOConfig):
        super().__init__()
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(f"observations must be a Box, got {observation_space}")
        obs_dim = int(np.prod(observation_space.shape))

        self.critic = _mlp(obs_dim, config.hidden_sizes, 1, out_std=1.0)
        if isinstance(action_space, spaces.Box):
            self.continuous = True
            self.head = GaussianHead(
                obs_dim,
                config.hidden_sizes,
                int(np.prod(action_space.shape)),
                config.log_std_init,
            )
        elif isinstance(action_space, spaces.Discrete):
            self.continuous = False
            self.head = CategoricalHead(
                obs_dim, config.hidden_sizes, int(action_space.n), config.log_std_init
            )
        else:
            raise TypeError(f"unsupported action space {action_space}")

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """``(action, log_prob, entropy, value)`` — CleanRL's one-call interface."""
        distribution = self.head.distribution(obs)
        if action is None:
            action = distribution.sample()
        return (
            action,
            self.head.log_prob(distribution, action),
            self.head.entropy(distribution),
            self.value(obs),
        )

    @torch.no_grad()
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """The evaluation action: the distribution's mean (or argmax).

        Evaluation is deterministic because M2 grades the *schedule* a policy
        induces, analytically, through the oracle — and a schedule is only
        well-defined if the policy is.
        """
        return self.head.deterministic(obs)


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------


def _to_env_action(action: np.ndarray, action_space) -> np.ndarray | int:
    """Clip a continuous action into the box; pass a discrete one straight through."""
    if isinstance(action_space, spaces.Box):
        return np.clip(action, action_space.low, action_space.high)
    return int(action)


def _torch_seed(seed: int) -> int:
    """Fold an arbitrary-width pool seed into torch's 64-bit seed space."""
    value = int(seed)
    folded = 0
    while value:
        folded ^= value & 0x7FFF_FFFF_FFFF_FFFF
        value >>= 63
    return folded


@dataclass
class _EpisodeTracker:
    """Undiscounted return and length of the episodes that finished this rollout."""

    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    _running: np.ndarray | None = None
    _steps: np.ndarray | None = None

    def start(self, num_envs: int) -> None:
        self._running = np.zeros(num_envs, dtype=np.float64)
        self._steps = np.zeros(num_envs, dtype=np.int64)

    def record(self, index: int, reward: float, done: bool) -> None:
        self._running[index] += reward
        self._steps[index] += 1
        if done:
            self.returns.append(float(self._running[index]))
            self.lengths.append(int(self._steps[index]))
            self._running[index] = 0.0
            self._steps[index] = 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainResult:
    """What a training run produced, and what it cost.

    ``returns`` is the *training* return — the sampled reward the agent actually
    optimised, scaling included. It is a progress trace and never a graded
    number: M2's metric is analytic, through
    :func:`~temper.oracle.cost.schedule_moments` on the eval schedule.
    """

    agent: Agent
    config: PPOConfig
    seed: int
    global_step: int
    updates: int
    seconds: float
    timed_out: bool
    returns: list[float]          # mean training return per update
    episode_counts: list[int]     # episodes finished per update
    approx_kls: list[float]
    entropies: list[float]
    value_losses: list[float]

    def as_dict(self) -> dict:
        """A JSON-safe summary. The weights are not in it; the trace is."""
        return {
            "seed": self.seed,
            "global_step": self.global_step,
            "updates": self.updates,
            "seconds": self.seconds,
            "timed_out": self.timed_out,
            "final_train_return": self.returns[-1] if self.returns else None,
            "train_returns": self.returns,
            "approx_kl": self.approx_kls,
            "entropy": self.entropies,
            "value_loss": self.value_losses,
        }


def train(
    env_fns: Sequence[Callable[[], Env]],
    config: PPOConfig,
    *,
    seed: int,
    progress: Callable[[int, dict], None] | None = None,
) -> TrainResult:
    """Run PPO on `env_fns` and return the trained agent.

    Parameters
    ----------
    env_fns:
        One factory per parallel environment. They are *called here*, and each
        factory is responsible for putting its env in a seeded state before
        returning it — this loop only ever calls the argument-free
        ``env.reset()``, which continues the stream the factory opened. That is
        the one arrangement that is correct for both sides: gymnasium's classic
        control envs are seeded by ``reset(seed=...)``, while
        :class:`~temper.env.ExecutionEnv` reads ``seed`` as a *stream index*
        within its pool, so a loop that helpfully passed ``reset(seed=i)`` would
        silently redirect every training env to streams ``0 .. num_envs - 1``
        and undo the caller's seed addressing (invariant 5).
    config:
        Hyperparameters; ``config.num_envs`` must match ``len(env_fns)``.
    seed:
        Seeds torch and this function's numpy generator. Resolve it through
        :func:`temper.seeding.pool_seeds` so a committed result is addressed
        rather than drawn (invariant 5).
    progress:
        Optional ``(update, metrics) -> None`` callback, for a live trace.

    There is deliberately no reward hook here. Whatever an experiment does to the
    reward — the fixed affine scale, or M2 task 3's sanctioned control variate —
    it does with an environment wrapper the factory composes, so the reward this
    loop optimises is exactly the reward the env emitted and there is one place
    to look to find out what that was.
    """
    if len(env_fns) != config.num_envs:
        raise ValueError(
            f"config.num_envs is {config.num_envs} but {len(env_fns)} env "
            "factories were given"
        )

    # Pinned before the first tensor is allocated: the thread count decides the
    # reduction order, and the reduction order decides the weights. This is the
    # half of reproducibility the seed cannot address (see `torch_threads`).
    if config.torch_threads is not None:
        torch.set_num_threads(int(config.torch_threads))

    # `pool_seeds` hands back 128-bit integers; torch wants 64. Folding rather
    # than truncating so two pool seeds that share their low bits cannot land on
    # the same network initialisation.
    torch.manual_seed(_torch_seed(seed))
    rng = np.random.default_rng(seed)

    envs = [make() for make in env_fns]
    observation_space = envs[0].observation_space
    action_space = envs[0].action_space
    agent = Agent(observation_space, action_space, config)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    obs_shape = observation_space.shape
    action_shape = action_space.shape if isinstance(action_space, spaces.Box) else ()

    obs_buf = torch.zeros((config.num_steps, config.num_envs, *obs_shape), dtype=AGENT_DTYPE)
    act_buf = torch.zeros((config.num_steps, config.num_envs, *action_shape), dtype=AGENT_DTYPE)
    logp_buf = torch.zeros((config.num_steps, config.num_envs), dtype=AGENT_DTYPE)
    rew_buf = torch.zeros((config.num_steps, config.num_envs), dtype=AGENT_DTYPE)
    done_buf = torch.zeros((config.num_steps, config.num_envs), dtype=AGENT_DTYPE)
    val_buf = torch.zeros((config.num_steps, config.num_envs), dtype=AGENT_DTYPE)

    next_obs = np.stack([env.reset()[0] for env in envs]).astype(np.float32)
    next_done = np.zeros(config.num_envs, dtype=np.float32)

    tracker = _EpisodeTracker()
    tracker.start(config.num_envs)

    returns_trace: list[float] = []
    episode_counts: list[int] = []
    kl_trace: list[float] = []
    entropy_trace: list[float] = []
    value_loss_trace: list[float] = []

    global_step = 0
    started = time.perf_counter()
    updates_done = 0

    for update in range(1, config.num_updates + 1):
        if config.anneal_lr:
            fraction = 1.0 - (update - 1.0) / config.num_updates
            optimizer.param_groups[0]["lr"] = fraction * config.learning_rate

        before = len(tracker.returns)
        for step in range(config.num_steps):
            obs_tensor = torch.as_tensor(next_obs, dtype=AGENT_DTYPE)
            obs_buf[step] = obs_tensor
            done_buf[step] = torch.as_tensor(next_done, dtype=AGENT_DTYPE)

            with torch.no_grad():
                action, log_prob, _, value = agent.action_and_value(obs_tensor)
            val_buf[step] = value
            act_buf[step] = action.to(AGENT_DTYPE)
            logp_buf[step] = log_prob

            actions = action.numpy()
            step_obs = np.empty_like(next_obs)
            for index, env in enumerate(envs):
                observation, reward, terminated, truncated, _ = env.step(
                    _to_env_action(actions[index], action_space)
                )
                global_step += 1

                done = bool(terminated or truncated)
                # The trace records what the episode actually paid out, before
                # the bootstrap below adds a value estimate that no environment
                # ever emitted.
                tracker.record(index, float(reward), done)
                if truncated and not terminated:
                    # Time-limit bootstrapping: the episode did not end, the
                    # clock did, so the value of where it stopped is part of the
                    # return. Without this the agent is trained to believe the
                    # world ends at the horizon, which on Pendulum is the
                    # difference between -200 and never getting there.
                    with torch.no_grad():
                        tail = agent.value(
                            torch.as_tensor(observation, dtype=AGENT_DTYPE).unsqueeze(0)
                        )
                    reward = float(reward) + config.gamma * float(tail.item())

                rew_buf[step, index] = float(reward)
                if done:
                    observation, _ = env.reset()
                step_obs[index] = np.asarray(observation, dtype=np.float32)
                next_done[index] = float(done)
            next_obs = step_obs

        # -- advantages ----------------------------------------------------
        with torch.no_grad():
            next_value = agent.value(torch.as_tensor(next_obs, dtype=AGENT_DTYPE))
            advantages = torch.zeros_like(rew_buf)
            last_gae = torch.zeros(config.num_envs, dtype=AGENT_DTYPE)
            for step in reversed(range(config.num_steps)):
                if step == config.num_steps - 1:
                    next_non_terminal = 1.0 - torch.as_tensor(next_done, dtype=AGENT_DTYPE)
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - done_buf[step + 1]
                    next_values = val_buf[step + 1]
                delta = (
                    rew_buf[step]
                    + config.gamma * next_values * next_non_terminal
                    - val_buf[step]
                )
                last_gae = (
                    delta
                    + config.gamma * config.gae_lambda * next_non_terminal * last_gae
                )
                advantages[step] = last_gae
            returns = advantages + val_buf

        b_obs = obs_buf.reshape((-1, *obs_shape))
        b_actions = act_buf.reshape((-1, *action_shape))
        if not agent.continuous:
            b_actions = b_actions.long()
        b_logprobs = logp_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = val_buf.reshape(-1)

        # -- the update ----------------------------------------------------
        indices = np.arange(config.batch_size)
        approx_kl = torch.zeros(())
        entropy_loss = torch.zeros(())
        value_loss = torch.zeros(())
        for _ in range(config.update_epochs):
            rng.shuffle(indices)
            for start in range(0, config.batch_size, config.minibatch_size):
                batch = indices[start : start + config.minibatch_size]
                _, new_logprob, entropy, new_value = agent.action_and_value(
                    b_obs[batch], b_actions[batch]
                )
                log_ratio = new_logprob - b_logprobs[batch]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()

                minibatch_advantages = b_advantages[batch]
                if config.norm_adv:
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std() + 1e-8)

                policy_loss = torch.max(
                    -minibatch_advantages * ratio,
                    -minibatch_advantages
                    * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef),
                ).mean()

                new_value = new_value.view(-1)
                if config.clip_vloss:
                    unclipped = (new_value - b_returns[batch]) ** 2
                    clipped_value = b_values[batch] + torch.clamp(
                        new_value - b_values[batch], -config.clip_coef, config.clip_coef
                    )
                    clipped = (clipped_value - b_returns[batch]) ** 2
                    value_loss = 0.5 * torch.max(unclipped, clipped).mean()
                else:
                    value_loss = 0.5 * ((new_value - b_returns[batch]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    - config.ent_coef * entropy_loss
                    + config.vf_coef * value_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None and approx_kl.item() > config.target_kl:
                break

        updates_done = update
        finished = tracker.returns[before:]
        returns_trace.append(float(np.mean(finished)) if finished else float("nan"))
        episode_counts.append(len(finished))
        kl_trace.append(float(approx_kl.item()))
        entropy_trace.append(float(entropy_loss.item()))
        value_loss_trace.append(float(value_loss.item()))

        if progress is not None:
            progress(
                update,
                {
                    "global_step": global_step,
                    "train_return": returns_trace[-1],
                    "approx_kl": kl_trace[-1],
                    "entropy": entropy_trace[-1],
                    "seconds": time.perf_counter() - started,
                },
            )

        if config.max_seconds is not None:
            if time.perf_counter() - started > config.max_seconds:
                break

    for env in envs:
        env.close()

    # "Timed out" means *the run was cut short*, not "the last update happened to
    # finish after the cap". The check above fires at update boundaries, so a run
    # that completes its final update a second past the budget would otherwise be
    # labelled truncated while having done every step the config asked for — and
    # a results file that says `timed_out` about a complete run is worse than one
    # that says nothing, because it invites a re-run that changes nothing.
    timed_out = updates_done < config.num_updates

    return TrainResult(
        agent=agent,
        config=config,
        seed=seed,
        global_step=global_step,
        updates=updates_done,
        seconds=time.perf_counter() - started,
        timed_out=timed_out,
        returns=returns_trace,
        episode_counts=episode_counts,
        approx_kls=kl_trace,
        entropies=entropy_trace,
        value_losses=value_loss_trace,
    )


# ---------------------------------------------------------------------------
# Evaluation of a trained agent on a plain gymnasium env (the smoke test's bar)
# ---------------------------------------------------------------------------


def evaluate(
    make_env: Callable[[], Env],
    agent: Agent,
    episodes: int,
    *,
    seed: int = 0,
    deterministic: bool = True,
) -> list[float]:
    """Undiscounted return of `episodes` episodes under the greedy policy.

    Used by the control-task smoke test. Temper's own evaluation does *not* come
    through here — it is analytic, through the oracle, because a sampled estimate
    of a 1 bps objective under a 95 bps per-episode standard deviation would need
    on the order of 10^7 episodes to resolve the tolerance
    (``docs/briefs/M2-ppo-rediscovery.md``).
    """
    env = make_env()
    scores: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total, done = 0.0, False
        while not done:
            obs_tensor = torch.as_tensor(
                np.asarray(observation, dtype=np.float32), dtype=AGENT_DTYPE
            ).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    action = agent.deterministic_action(obs_tensor)
                else:
                    action, _, _, _ = agent.action_and_value(obs_tensor)
            observation, reward, terminated, truncated, _ = env.step(
                _to_env_action(action.numpy()[0], env.action_space)
            )
            total += float(reward)
            done = bool(terminated or truncated)
        scores.append(total)
    env.close()
    return scores


def with_overrides(config: PPOConfig, **overrides) -> PPOConfig:
    """`dataclasses.replace`, named for what a caller is doing when they use it."""
    return replace(config, **overrides)
