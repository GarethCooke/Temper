"""Task 5 — the exact, per-episode identities the env must satisfy.

The Monte-Carlo differential in ``test_differential.py`` is statistical: it can
only ever say "consistent within 4 sigma". These are not. Each one holds episode
by episode, to float round-off, and between them they pin the structure the
differential then measures:

* per-step shortfalls sum to the shortfall implied by the realised price path;
* the summed inventory penalty *is* ``lambda * V[cost]`` — the mechanical form of
  invariant 7, and the reason "the env pays out the functional the oracle
  computes" is a checked statement rather than a design intention;
* summed reward is ``-(realised IS + lambda V)``;
* subtracting the realised price path from the realised cost leaves the oracle's
  ``E[cost]`` exactly — including for a policy whose tail the env had to
  force-liquidate;
* the same ``(config, seed address)`` replays bitwise.

Thresholds come from ``configs/m1_differential.yaml``.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from numpy.random import default_rng

from temper.agents import baseline
from temper.env import SHOCK_KEY, ExecutionEnv
from temper.eval import run_episode
from temper.oracle import schedule_moments, twap_trajectory

from .conftest import M1_CONFIG, build_env, case_by_id

IDENTITIES = M1_CONFIG["identities"]
TELESCOPE_RTOL = float(IDENTITIES["telescope_rtol"])
PENALTY_RTOL = float(IDENTITIES["penalty_rtol"])
EPISODES = int(IDENTITIES["episodes"])
UNDERTRADE_FRACTION = float(IDENTITIES["undertrade_fraction"])
IDENTITY_STREAM = int(M1_CONFIG["seeding"]["identity_stream"])

CASES = [case_by_id(case_id) for case_id in M1_CONFIG["tiers"]["fast"]["cases"]]
SCHEDULES = list(M1_CONFIG["schedules"])


def assert_identity(actual: float, expected: float, scale: float, rtol: float, what: str):
    """Assert an identity to `rtol` of the *magnitude of the terms it sums*.

    Two of these identities equate sums that cancel: an episode whose price path
    happened to offset its impact charges has a total reward of ~1e-2 bps built
    out of per-bin terms of ~1e2 bps. Round-off in such a sum is ~1e-16 of the
    terms — ~1e-14 absolute — however small the total it lands on, so measuring
    it against the total would make the verdict depend on which Gaussian draws
    came up rather than on whether the env is right. The comparison is therefore
    against the scale of the quantities being summed. The bar the brief
    pre-states, 1e-10, is unchanged; what it is relative to is stated here rather
    than left to the luck of the seed.
    """
    error = abs(actual - expected)
    assert error <= rtol * scale, (
        f"{what}: {actual!r} != {expected!r}; error {error:.3e} exceeds "
        f"{rtol:g} x {scale:.3e}"
    )


class UnderTradePolicy:
    """Executes a fixed fraction of each bin's TWAP trade, and so runs out of road.

    Task 5(d)'s instrument: it leaves roughly ``(1 - fraction) * X`` unexecuted
    going into the final bin, which the env must force-liquidate and charge like
    any other trade. A policy that could dodge the terminal constraint by simply
    trading less would make every cost comparison in this project meaningless.
    """

    def __init__(self, market, order_size: float, fraction: float) -> None:
        self.name = f"undertrade-{fraction:g}"
        self.market = market
        self.trades = fraction * -np.diff(twap_trajectory(market, order_size))
        self._n_bins = market.n_bins

    def reset(self) -> None:
        pass

    def act(self, observation) -> float:
        return float(self.trades[self._n_bins - int(round(observation[0] * self._n_bins))])


class RandomFractionPolicy:
    """Sells a seeded random fraction of what is left. Only used to test replay.

    Its generator persists across episodes on purpose: that makes the *trajectory*
    differ from episode to episode, so "the same seed address replays bitwise" is
    a statement about the whole (env, policy) pair rather than about a schedule
    that was never going to vary anyway.
    """

    name = "random-fraction"

    def __init__(self, order_size: float, seed: int) -> None:
        self.order_size = order_size
        self._rng = default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, observation) -> float:
        return float(self._rng.uniform(0.0, 1.0) * observation[1] * self.order_size)


def _policies(case):
    """The named baselines plus the under-trader, for one case."""
    named = [
        baseline(name, case.market, case.order_size, case.lambda_risk)
        for name in SCHEDULES
    ]
    return named + [UnderTradePolicy(case.market, case.order_size, UNDERTRADE_FRACTION)]


PAIRS = [(case, policy) for case in CASES for policy in _policies(case)]
PAIR_IDS = [f"{case.case_id}:{policy.name}" for case, policy in PAIRS]


@pytest.fixture(params=PAIRS, ids=PAIR_IDS)
def episodes(request):
    """`EPISODES` episodes of one (case, policy) pair, with the per-step record."""
    case, policy = request.param
    stream = IDENTITY_STREAM + PAIRS.index(request.param)
    env = build_env(case, stream)
    env.reset(seed=stream)
    return case, policy, [run_episode(env, policy) for _ in range(EPISODES)]


# ---------------------------------------------------------------------------
# (a) (b) (c) — the three sums
# ---------------------------------------------------------------------------


def test_per_step_shortfalls_telescope_to_the_realised_cost(episodes):
    """(a) The charges billed bin by bin add up to what the order actually cost.

    The two sides are computed by different routes: the left sums the per-step
    shortfall the env reports, the right rebuilds the shortfall from the realised
    proceeds — the price path, notional in, notional out. A double-count or a
    dropped component in one bin breaks this even though both sides come from the
    same episode.
    """
    _, _, results = episodes
    for result in results:
        assert_identity(
            float(np.sum(result.shortfalls)),
            result.cost_bps,
            scale=float(np.sum(np.abs(result.shortfalls))),
            rtol=TELESCOPE_RTOL,
            what="summed per-step shortfalls vs the cost implied by the price path",
        )


def test_the_inventory_penalty_is_lambda_times_the_oracles_variance(episodes):
    """(b) Invariant 7, pinned mechanically.

    The per-step penalty ``lambda * sigma_bin^2 * BPS^2 * (x_k / X)^2`` summed over
    an episode is ``lambda * V[cost]`` for the realised schedule, where ``V`` is
    the oracle's. Not approximately, not in expectation: the same number, because
    it is the same formula evaluated on the same inventory path. If the env's
    penalty and the oracle's variance ever stop being one functional, this fails
    before any training curve gets the chance to hide it.
    """
    case, _, results = episodes
    for result in results:
        variance = schedule_moments(result.trajectory, case.market).variance
        assert result.penalty_bps == pytest.approx(
            case.lambda_risk * variance, rel=PENALTY_RTOL
        )


def test_summed_reward_is_minus_the_frozen_objective(episodes):
    """(c) ``sum_k r_k == -(realised IS + lambda V)`` — §4's reward, end to end."""
    case, _, results = episodes
    for result in results:
        variance = schedule_moments(result.trajectory, case.market).variance
        target = -(result.cost_bps + case.lambda_risk * variance)
        scale = abs(result.cost_bps) + case.lambda_risk * variance
        for label, total in (
            ("summed step rewards", float(np.sum(result.rewards))),
            ("the env's own reward total", result.reward),
        ):
            assert_identity(
                total,
                target,
                scale=scale,
                rtol=TELESCOPE_RTOL,
                what=f"{label} vs -(realised IS + lambda V)",
            )


# ---------------------------------------------------------------------------
# The exact expectation identity, and (d) forced terminal liquidation
# ---------------------------------------------------------------------------


def test_removing_the_price_path_leaves_the_oracles_expected_cost(episodes):
    """Cost minus noise equals ``E[cost]``, exactly, in every single episode.

    The realised cost is ``sum_k w_k * (impact_k - walk_k)``: a deterministic part
    and a mean-zero part driven by the price path the env drew. Add the price-path
    term back and what remains must be the oracle's ``E[cost]`` for the realised
    schedule — no averaging, no confidence interval. This is the strongest form of
    the differential available, and it is why the Monte-Carlo tiers are checking
    the *distribution* rather than hunting for a bias they could not resolve.
    """
    case, _, results = episodes
    for result in results:
        weights = result.shares / case.order_size
        noise = float(np.sum(weights * result.walks))
        expected = schedule_moments(result.trajectory, case.market).expected
        assert_identity(
            result.cost_bps + noise,
            expected,
            scale=abs(result.cost_bps) + abs(noise) + expected,
            rtol=TELESCOPE_RTOL,
            what="realised cost with the price path removed vs the oracle's E[cost]",
        )


def test_an_undertrading_policy_is_force_liquidated_at_the_last_bin():
    """(d) The terminal constraint bites, and the remainder is charged normally."""
    case = CASES[0]
    policy = UnderTradePolicy(case.market, case.order_size, UNDERTRADE_FRACTION)
    env = build_env(case, IDENTITY_STREAM + 500)
    result = run_episode(env, policy, seed=IDENTITY_STREAM + 500)

    n_bins = case.market.n_bins
    planned = policy.trades
    assert np.allclose(result.shares[:-1], planned[:-1], rtol=1e-15), (
        "the env changed a trade it was not asked to change"
    )
    remainder = result.trajectory[n_bins - 1]
    assert remainder > 0.4 * case.order_size, "the policy did not under-trade"
    assert result.shares[-1] == pytest.approx(remainder, rel=1e-15), (
        "the final bin did not liquidate the remainder"
    )
    assert result.trajectory[-1] == 0.0, "the order did not finish flat"
    assert result.shares[-1] > planned[-1], "the forced trade was not larger than planned"

    # ...and the forced schedule is costed like any other: the same noise-removal
    # identity holds against schedule_moments of the *realised* trajectory.
    weights = result.shares / case.order_size
    noise = float(np.sum(weights * result.walks))
    moments = schedule_moments(result.trajectory, case.market)
    assert_identity(
        result.cost_bps + noise,
        moments.expected,
        scale=abs(result.cost_bps) + abs(noise) + moments.expected,
        rtol=TELESCOPE_RTOL,
        what="the force-liquidated schedule's cost vs its oracle E[cost]",
    )
    assert result.penalty_bps == pytest.approx(
        case.lambda_risk * moments.variance, rel=PENALTY_RTOL
    )


# ---------------------------------------------------------------------------
# (e) determinism
# ---------------------------------------------------------------------------


def _replay(case, stream_index: int, policy_seed: int, n_episodes: int = 8):
    env = build_env(case, stream_index)
    policy = RandomFractionPolicy(case.order_size, policy_seed)
    env.reset(seed=stream_index)
    return [run_episode(env, policy) for _ in range(n_episodes)]


def test_identical_seed_addresses_replay_identical_episodes():
    """(e) Same ``(config, seed)``, bitwise-identical arrays. Invariant 1."""
    case = CASES[0]
    first = _replay(case, IDENTITY_STREAM + 600, policy_seed=7)
    second = _replay(case, IDENTITY_STREAM + 600, policy_seed=7)

    for left, right in zip(first, second):
        assert np.array_equal(left.trajectory, right.trajectory)
        assert np.array_equal(left.rewards, right.rewards)
        assert np.array_equal(left.shortfalls, right.shortfalls)
        assert left.cost_bps == right.cost_bps

    trajectories = {tuple(result.trajectory) for result in first}
    assert len(trajectories) == len(first), (
        "the replay policy produced the same trajectory twice; the test would pass "
        "on a constant"
    )


def test_a_different_stream_gives_a_different_episode():
    """...and the stream index is doing something, so (e) is not vacuous."""
    case = CASES[0]
    first = _replay(case, IDENTITY_STREAM + 600, policy_seed=7, n_episodes=1)[0]
    other = _replay(case, IDENTITY_STREAM + 601, policy_seed=7, n_episodes=1)[0]
    assert not np.array_equal(first.shortfalls, other.shortfalls)


def test_the_diagnostic_pool_is_not_the_train_or_eval_pool():
    """Invariant 5: M1's tens of millions of draws must not touch M2's streams."""
    assert M1_CONFIG["seeding"]["pool"] == "m1/differential"
    env = build_env(CASES[0], IDENTITY_STREAM)
    _, pool, _ = env.seed_address
    assert pool not in ("train", "eval")


# ---------------------------------------------------------------------------
# The §4 contract: action clipping, observation, episode boundaries
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> ExecutionEnv:
    return build_env(CASES[0], IDENTITY_STREAM + 700)


def test_the_observation_is_the_two_fractions_and_nothing_else(env):
    """§4: `(time remaining fraction, inventory remaining fraction)`, minimal on purpose."""
    observation, _ = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    assert observation.dtype == np.float64
    assert observation == pytest.approx(np.array([1.0, 1.0]))

    n_bins = env.market.n_bins
    per_bin = env.order_size / n_bins
    for step in range(n_bins):
        observation, _, terminated, _, _ = env.step(per_bin)
        assert env.observation_space.contains(observation)
        assert observation[0] == pytest.approx((n_bins - step - 1) / n_bins)
        assert observation[1] == pytest.approx(max(0.0, 1.0 - (step + 1) / n_bins), abs=1e-12)
    assert terminated
    assert observation == pytest.approx(np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# Observation minimality (M1a task 3) — closing the M2 leak before M2 exists
# ---------------------------------------------------------------------------
#
# §4: "Rediscovery must not smuggle in signal." Under Phase-1 dynamics the shock
# is the *only* thing an agent could learn from that TWAP and the sinh schedules
# cannot see, so an observation that leaked it — by width, by a stray field, or
# by a public attribute an M2 wrapper could reach for — would turn "the agent
# rediscovered Almgren-Chriss" into "the agent traded on tomorrow's prices". The
# tests below are cheap now and unwriteable once a training loop depends on the
# answer.


def test_the_observation_space_is_exactly_two_dimensional(env):
    """Two numbers. Not two-or-more, and not a dict that could grow a third."""
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (2,)
    assert env.observation_space.dtype == np.float64

    observation, _ = env.reset(seed=11)
    assert observation.shape == (2,)
    for _ in range(env.market.n_bins):
        observation, _, _, _, _ = env.step(env.order_size / env.market.n_bins)
        assert observation.shape == (2,)


def test_the_observation_carries_no_information_about_the_shock():
    """Two streams, different shocks, byte-identical observation sequences.

    The structural version of "the shock is not in the observation": run the same
    deterministic schedule against two different shock streams and require the
    observations to be bitwise equal while the shocks are not. A leak of any
    size — the walk itself, a noisy price, a realised-cost field — breaks this,
    including one that a shape assertion would miss because it replaced a field
    rather than adding one.
    """
    case = CASES[0]
    policy_trades = twap_trajectory(case.market, case.order_size)
    per_bin = -np.diff(policy_trades)

    def observations(stream_index):
        env = build_env(case, stream_index)
        observation, _ = env.reset(seed=stream_index)
        seen, shocks = [observation], []
        for shares in per_bin:
            observation, _, _, _, info = env.step(float(shares))
            seen.append(observation)
            shocks.append(info[SHOCK_KEY])
        return np.array(seen), np.array(shocks)

    first, first_shocks = observations(IDENTITY_STREAM + 800)
    second, second_shocks = observations(IDENTITY_STREAM + 801)

    assert not np.array_equal(first_shocks, second_shocks), (
        "the two streams drew the same shocks; the comparison below is vacuous"
    )
    assert np.array_equal(first, second), (
        "the observation changed with the shock stream: something about the price "
        "path is reaching the agent, and rediscovery is no longer a claim about "
        "learning the schedule (constitution §4)"
    )


def test_the_shock_is_published_through_info_and_nowhere_else(env):
    """`info[SHOCK_KEY]` is the whole public route to the price path.

    The env's public surface is pinned exactly rather than spot-checked: a new
    public attribute is how a leak would actually arrive — nobody would put the
    walk in the observation on purpose, but `env.walk_bps` for "debugging" is one
    line and no test would have noticed.

    Not covered, deliberately: gymnasium's own ``Env.np_random``. It is the
    framework's contract rather than ours, and it exposes the *generator*, not
    the realised path — anything drawing from it advances the stream and dies to
    the determinism test instead.
    """
    env.reset(seed=12)
    _, _, _, _, info = env.step(env.order_size / env.market.n_bins)
    assert SHOCK_KEY in info
    assert info[SHOCK_KEY] != 0.0

    instance_surface = {name for name in vars(env) if not name.startswith("_")}
    assert instance_surface == {
        "market",
        "order_size",
        "lambda_risk",
        "eta_tilde",
        "action_space",
        "observation_space",
    }, "the env grew a public attribute; if it exposes the price path, §4 is broken"

    class_surface = {name for name in vars(ExecutionEnv) if not name.startswith("_")}
    assert class_surface == {"metadata", "seed_address", "step_count", "reset", "step"}

    for name in instance_surface | class_surface:
        value = getattr(env, name)
        if isinstance(value, float):
            assert value != info[SHOCK_KEY], f"{name} exposes the realised shock"


def test_the_step_counter_is_monotone_and_survives_reset(env):
    """M1a task 2's instrument, checked before the tiers lean on it."""
    assert env.step_count == 0
    env.reset(seed=13)
    assert env.step_count == 0, "reset() must not clear the counter's history"

    n_bins = env.market.n_bins
    for expected in range(1, n_bins + 1):
        env.step(env.order_size / n_bins)
        assert env.step_count == expected

    env.reset(seed=14)
    assert env.step_count == n_bins, "the counter restarted; it is not monotone"
    env.step(0.0)
    assert env.step_count == n_bins + 1

    with pytest.raises(ValueError, match="NaN"):
        env.step(np.nan)
    assert env.step_count == n_bins + 1, "a rejected action counted as a bin"


def test_actions_are_clipped_to_what_is_left(env):
    """§4: shares this interval, clipped to `[0, remaining]` — both ends."""
    env.reset(seed=2)
    _, _, _, _, info = env.step(-1e9)
    assert info["shares"] == 0.0
    assert info["inventory"] == env.order_size

    _, _, _, _, info = env.step(10.0 * env.order_size)
    assert info["shares"] == pytest.approx(env.order_size, rel=1e-15)
    assert info["inventory"] == 0.0

    # Flat with bins to spare: the rest of the episode still has to be stepped,
    # and must cost nothing more.
    for _ in range(env.market.n_bins - 2):
        _, reward, terminated, _, info = env.step(0.0)
        assert info["shares"] == 0.0
        assert reward == 0.0
    assert terminated


def test_an_action_may_be_a_scalar_or_a_one_element_array(env):
    """Policies return floats; gymnasium spaces produce arrays. Both are fine."""
    env.reset(seed=3)
    _, _, _, _, scalar = env.step(1000.0)
    _, _, _, _, array = env.step(np.array([1000.0]))
    assert scalar["shares"] == array["shares"] == 1000.0
    assert env.action_space.contains(np.array([1000.0]))
    with pytest.raises(ValueError, match="must be one number"):
        env.step(np.array([1.0, 2.0]))


@pytest.mark.parametrize("bad", [np.nan, float("nan")])
def test_a_nan_action_is_an_error_not_a_silent_no_op(env, bad):
    """A broken agent must fail loudly rather than poison an episode with NaNs."""
    env.reset(seed=4)
    with pytest.raises(ValueError, match="NaN"):
        env.step(bad)


def test_stepping_outside_an_episode_is_an_error():
    """No implicit reset: an unstarted or finished episode is a bug in the caller."""
    case = CASES[0]
    env = build_env(case, IDENTITY_STREAM + 701)
    with pytest.raises(RuntimeError, match="before reset"):
        env.step(0.0)

    env.reset(seed=5)
    for _ in range(case.market.n_bins):
        env.step(0.0)
    with pytest.raises(RuntimeError, match="after the episode terminated"):
        env.step(0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"order_size": 0.0}, "order_size must be positive"),
        ({"lambda_risk": -1.0}, "lambda_risk must be non-negative"),
    ],
)
def test_the_env_rejects_impossible_configurations(kwargs, match):
    case = CASES[0]
    settings = {
        "order_size": case.order_size,
        "lambda_risk": case.lambda_risk,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        ExecutionEnv(case.market, root_seed=1, **settings)
