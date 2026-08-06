"""M2 task 2 — the PPO adaptation solves a standard control task.

Green **before** the agent is ever pointed at Temper, and permanently in the
suite afterwards (constitution §5). What it buys is a question that is otherwise
unanswerable from inside the milestone: when the agent plateaus above epsilon on
``ExecutionEnv``, is PPO broken or is the environment hard? Pendulum answers it
in three minutes, and the answer does not depend on anyone's judgement about the
shape of a training curve.

Two tasks, because the two heads must both stay alive. ``Pendulum-v1`` is
continuous, which is the path Temper uses — a Gaussian head, an action clipped
into a box, a reward traded off over a horizon. ``CartPole-v1`` is the cheap
discrete second check: it shares the loop, and this is what notices if it stops
doing so.

Behind the ``training`` marker: `make test` is the per-commit gate and stays
evening-sized. `make smoke` runs this, and it is part of milestone acceptance
like `make differential`.
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import pytest
import yaml

from temper.agents.execution import RewardScale
from temper.agents.ppo import PPOConfig, evaluate, train
from temper.seeding import POOLS, pool_seeds

from .conftest import REPO_ROOT

SMOKE_CONFIG_PATH = REPO_ROOT / "configs" / "ppo_smoke.yaml"
SMOKE_CONFIG = yaml.safe_load(SMOKE_CONFIG_PATH.read_text(encoding="utf-8"))
TASKS = {task["id"]: task for task in SMOKE_CONFIG["tasks"]}

#: What each task actually scored, printed at the end. A green dot does not say
#: whether Pendulum cleared -200 by one point or by four hundred, and the margin
#: is the only part of a smoke test that ages informatively.
_OBSERVED: dict[str, list[tuple[float, float]]] = {}


@pytest.fixture(scope="module", autouse=True)
def report_smoke(request):
    yield
    if not _OBSERVED:
        return
    writer = request.config.get_terminal_writer()
    writer.line("")
    writer.line("PPO smoke test:")
    for task_id, scores in _OBSERVED.items():
        task = TASKS[task_id]
        worst = min(mean for mean, _ in scores)
        seconds = sum(elapsed for _, elapsed in scores)
        writer.line(
            f"  {task['env_id']:14s} {len(scores)} seeds, worst mean return "
            f"{worst:+9.2f} against a threshold of {task['threshold']:+.1f}"
            f"  ({seconds:.0f}s)"
        )


def _env_factory(env_id: str, seed: int, reward_scale: float = 1.0):
    """A factory that returns an env already seeded.

    :func:`~temper.agents.ppo.train` only ever calls the argument-free
    ``reset()``, so seeding is the factory's job — the same contract Temper's own
    factory meets by carrying a pool address.

    `reward_scale` goes through the same :class:`~temper.agents.RewardScale`
    wrapper Temper's factory uses, and for the same reason: it is a committed
    constant, not a running normaliser. Evaluation passes 1.0, so the threshold
    is always measured on the environment's own reward.
    """

    def make():
        env = gym.make(env_id)
        if reward_scale != 1.0:
            env = RewardScale(env, reward_scale)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return make


def _run_task(task: dict, ordinal: int) -> tuple[float, float]:
    config = PPOConfig.from_mapping(task["ppo"])
    seeds = pool_seeds(int(task["root_seed"]), task["pool"], int(task["seeds"]))
    seed = seeds[ordinal]
    # gymnasium seeds are 32-bit; the pool hands back 128 bits of entropy, so the
    # env seed is folded rather than truncated. The *training* seed keeps its full
    # width and is folded inside `train` for torch.
    env_seed = seed % (2**31 - 1)

    scale = float(task.get("reward_scale", 1.0))
    started = time.perf_counter()
    result = train(
        [
            _env_factory(task["env_id"], env_seed + index, scale)
            for index in range(config.num_envs)
        ],
        config,
        seed=seed,
    )
    # Evaluated unscaled: the threshold is a statement about the environment's
    # own reward, so the training constant cannot flatter it.
    scores = evaluate(
        _env_factory(task["env_id"], env_seed + 10_000),
        result.agent,
        int(task["eval_episodes"]),
        seed=int(env_seed % 100_000),
    )
    elapsed = time.perf_counter() - started

    assert result.global_step <= config.total_timesteps + config.batch_size, (
        "the run used more steps than the config's budget"
    )
    assert not result.timed_out, (
        f"{task['env_id']} seed {ordinal} hit its {config.max_seconds}s wall-clock "
        f"cap after {result.global_step:,} of {config.total_timesteps:,} steps"
    )
    return float(np.mean(scores)), elapsed


def _check(task_id: str, ordinal: int) -> None:
    task = TASKS[task_id]
    mean_return, elapsed = _run_task(task, ordinal)
    _OBSERVED.setdefault(task_id, []).append((mean_return, elapsed))
    assert mean_return >= float(task["threshold"]), (
        f"{task['env_id']} seed {ordinal} scored {mean_return:.2f} over "
        f"{task['eval_episodes']} greedy episodes, below the pre-stated "
        f"{task['threshold']}. PPO itself is not converging — no result it "
        "produces on ExecutionEnv means anything until this is green."
    )


# ---------------------------------------------------------------------------
# The config is the pre-statement (invariant 3)
# ---------------------------------------------------------------------------


def test_the_smoke_thresholds_are_the_ones_the_brief_states():
    """Fast, always runs: the thresholds cannot drift while the run is marked.

    The runs below are behind a marker, so a session that never invokes them
    could still edit the bar they are measured against. This test is not behind
    the marker.
    """
    assert SMOKE_CONFIG["brief"] == "docs/briefs/M2-ppo-rediscovery.md"
    pendulum, cartpole = TASKS["pendulum"], TASKS["cartpole"]

    assert pendulum["env_id"] == "Pendulum-v1"
    assert pendulum["threshold"] == -200.0
    assert pendulum["eval_episodes"] == 100
    assert pendulum["seeds"] >= 3
    assert pendulum["ppo"]["total_timesteps"] <= 300_000

    assert cartpole["env_id"] == "CartPole-v1"
    assert cartpole["threshold"] >= 475.0
    assert cartpole["seeds"] >= 3

    for task in TASKS.values():
        assert task["pool"] in POOLS
        assert task["pool"] not in {"train", "eval"}, (
            "a control task may not spend a stream a committed result is "
            "addressed by"
        )


def test_both_action_space_paths_are_exercised():
    """One file, two heads — and both are actually run.

    Cheap to let the discrete path rot; this notices, without training anything.
    """
    assert gym.make(TASKS["pendulum"]["env_id"]).action_space.shape == (1,)
    assert gym.make(TASKS["cartpole"]["env_id"]).action_space.n == 2


# ---------------------------------------------------------------------------
# The runs
# ---------------------------------------------------------------------------


@pytest.mark.training
@pytest.mark.parametrize("ordinal", range(TASKS["pendulum"]["seeds"]))
def test_ppo_solves_pendulum(ordinal):
    """The continuous path, on the path Temper uses."""
    _check("pendulum", ordinal)


@pytest.mark.training
@pytest.mark.parametrize("ordinal", range(TASKS["cartpole"]["seeds"]))
def test_ppo_solves_cartpole(ordinal):
    """The discrete second check."""
    _check("cartpole", ordinal)
