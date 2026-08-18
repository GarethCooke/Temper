"""M3 task 1 — the antithetic pair does exactly what the brief says it rests on.

Two structural checks the brief makes permanent, because they are cheap now and
expensive later:

* **Action identity across the pair.** A policy shown the primary half's
  observations and, separately, the mirror half's takes bitwise-identical
  action sequences. This is the assumption the whole method rests on, and it
  fails silently and instantly the moment an observation carries price — so it
  is asserted here for the baselines and for a PPO network, and the pair
  wrapper's own per-step check is shown to be live rather than decorative.
* **Shock negation is exact.** The mirror's draws are the elementwise negation
  of the primary's — the same numbers negated, not a fresh sample from a
  mirrored distribution — at the generator, and the published cumulative shock
  is the bitwise negation on every step.

And the consequence they are there to secure: on the average of the pair the
noise cancels *exactly*, so the averaged reward equals the control variate's
deterministic reward to floating-point dust and sums to ``-(E + lambda V)`` —
invariant 7 restated for this estimator, without the analytic noise form ever
being subtracted.

Nothing in this module reports a number, so it draws from ``m3/diagnostic``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.random import Generator

from temper.agents import (
    Agent,
    FractionAction,
    FractionPolicy,
    PPOConfig,
    PPOPolicy,
    RewardScale,
    baseline,
    execution_env_factory,
    twap_fractions,
)
from temper.env import EPISODE_KEY, SHOCK_KEY, ExecutionEnv
from temper.eval import run_episode
from temper.eval.antithetic import (
    AntitheticPair,
    MirrorEnv,
    NegatedDraws,
    PairDiverged,
    PairLedger,
    antithetic_reward,
    mirror_of,
)
from temper.eval.experiment import ANTITHETIC, REGIMES, Estimator, load_experiment
from temper.eval.variate import DeterministicReward, deterministic_reward
from temper.oracle import optimal_trajectory, schedule_moments
from temper.seeding import M3_DIAGNOSTIC_POOL, POOLS, pool_rng

from .conftest import REPO_ROOT, RESOLVED_SEED_ADDRESSES, m2_experiment

M2 = m2_experiment()
M3 = load_experiment(REPO_ROOT / "configs" / "m3_antithetic_validation.yaml")
MARKET = M3.case.market
ORDER_SIZE = M3.case.order_size
LAMBDA = M3.lambda_risk
N_BINS = MARKET.n_bins
ROOT_SEED = M3.seeds.root_seed

#: Relative band on the exact identities — M1a's noise-identity order (1e-12);
#: the sums cancel ~1e2 bps of per-bin quantities down to ~1 bps, so the band is
#: relative to the summed absolute terms rather than to the surviving total.
IDENTITY_RTOL = 1e-11


def _env(stream: int) -> ExecutionEnv:
    return ExecutionEnv(
        MARKET, ORDER_SIZE, LAMBDA, root_seed=ROOT_SEED, pool=M3_DIAGNOSTIC_POOL,
        stream_index=stream,
    )


def _run(env, fractions):
    """Step a fraction-wrapped env through one episode; rewards and schedule."""
    env.reset()
    rewards: list[float] = []
    info: dict = {}
    for fraction in fractions:
        _, reward, _, _, info = env.step(np.array([fraction]))
        rewards.append(float(reward))
    return np.array(rewards), info[EPISODE_KEY]["trajectory"]


OPTIMAL_FRACTIONS = (
    lambda x: (x[:-1] - x[1:]) / x[:-1]
)(optimal_trajectory(MARKET, ORDER_SIZE, LAMBDA))
FRACTION_SEQUENCES = {"twap": twap_fractions(N_BINS), "optimal": OPTIMAL_FRACTIONS}


# ---------------------------------------------------------------------------
# Shock negation is exact
# ---------------------------------------------------------------------------


def test_negated_draws_are_the_exact_negation_at_the_generator():
    """Elementwise, bitwise, scalar and array — the same numbers, negated."""
    base = pool_rng(ROOT_SEED, M3_DIAGNOSTIC_POOL, 900)
    mirror = NegatedDraws(pool_rng(ROOT_SEED, M3_DIAGNOSTIC_POOL, 900))
    for _ in range(2_000):
        a, b = base.standard_normal(), mirror.standard_normal()
        assert b == -a and (b != a or a == 0.0)
    assert np.array_equal(mirror.standard_normal(64), -base.standard_normal(64))
    assert mirror.base is not base


def test_negated_draws_only_mirror_what_can_be_mirrored():
    """Only ``standard_normal``: negation is not a mirror for anything else."""
    with pytest.raises(TypeError):
        NegatedDraws(np.random.RandomState(0))  # legacy API is not a Generator
    proxy = NegatedDraws(pool_rng(ROOT_SEED, M3_DIAGNOSTIC_POOL, 901))
    with pytest.raises(AttributeError):
        proxy.uniform()
    with pytest.raises(AttributeError):
        proxy.normal(0.0, 1.0)
    with pytest.raises(ValueError):
        proxy.standard_normal(4, out=np.empty(4))


def test_the_mirror_env_negates_the_published_shock_bitwise_on_every_step():
    """Same address, lockstep — the cumulative walk is ``-walk`` to the bit."""
    primary, mirror = _env(902), mirror_of(_env(902))
    assert mirror.seed_address == primary.seed_address
    assert not mirror.negated
    rng = np.random.default_rng(1)
    for _ in range(5):
        primary.reset()
        mirror.reset()
        assert mirror.negated
        for _ in range(N_BINS):
            shares = float(rng.uniform(0.0, 0.3)) * ORDER_SIZE
            _, _, _, _, info = primary.step(shares)
            _, _, _, _, m_info = mirror.step(shares)
            assert m_info[SHOCK_KEY] == -info[SHOCK_KEY]
            assert m_info["shares"] == info["shares"]
        assert np.array_equal(
            info[EPISODE_KEY]["trajectory"], m_info[EPISODE_KEY]["trajectory"]
        )


def test_the_mirror_re_negates_after_a_rewind_to_a_stream():
    """`reset(seed=k)` rebuilds the env's generator; the mirror must re-wrap it."""
    mirror = mirror_of(_env(903))
    mirror.reset()
    first = mirror._rng
    mirror.reset(seed=904)
    assert isinstance(mirror._rng, NegatedDraws)
    assert mirror._rng is not first
    assert mirror.seed_address == (ROOT_SEED, M3_DIAGNOSTIC_POOL, 904)
    plain = _env(904)
    plain.reset()
    for _ in range(N_BINS):
        _, _, _, _, a = plain.step(1000.0)
        _, _, _, _, b = mirror.step(1000.0)
        assert b[SHOCK_KEY] == -a[SHOCK_KEY]


def test_mirror_of_refuses_anything_but_the_raw_env():
    with pytest.raises(TypeError):
        mirror_of(FractionAction(_env(905)))
    with pytest.raises(TypeError):
        AntitheticPair(FractionAction(_env(905)))


# ---------------------------------------------------------------------------
# Action identity across the pair
# ---------------------------------------------------------------------------


def _fresh_agent(seed: int) -> PPOPolicy:
    torch.manual_seed(seed)
    config = PPOConfig(num_envs=1, num_steps=13, num_minibatches=1)
    env = FractionAction(_env(906))
    agent = Agent(env.observation_space, env.action_space, config)
    return PPOPolicy(agent, ORDER_SIZE, name=f"fresh{seed}")


@pytest.mark.parametrize(
    "policy_name", ["twap", "ac", "optimal", "fractions", "ppo_a", "ppo_b"]
)
def test_a_policy_shown_each_half_separately_takes_identical_actions(policy_name):
    """The load-bearing assumption, made executable for every kind of policy.

    **When this test goes red in Phase 2, that is the designed signal and not a
    regression to be silenced.** The pairing cancels noise exactly only because
    the observation carries no price, so both halves take the same actions; an
    enriched observation breaks that, and this test is how the repo finds out.
    See ``ARCHITECTURE.md`` §9, *Antithetic pairing is the Phase-1
    variance-reduction regime, and at this reward magnitude it is bitwise the
    control variate* — the estimator degrades to partial cancellation, which is
    a different claim, not a broken one.

    Two rollouts through the shared :func:`~temper.eval.run_episode`, one on the
    primary env and one on its mirror, each with the policy acting on *that*
    half's observations. The action sequences must be bitwise identical, and so
    must the schedules — the observation carries no price, so nothing the policy
    sees can differ between the halves.
    """
    if policy_name in ("twap", "ac", "optimal"):
        policy = baseline(policy_name, MARKET, ORDER_SIZE, LAMBDA)
    elif policy_name == "fractions":
        policy = FractionPolicy(twap_fractions(N_BINS), ORDER_SIZE, "twap_fractions")
    else:
        policy = _fresh_agent(7 if policy_name == "ppo_a" else 11)

    primary, mirror = _env(907), mirror_of(_env(907))
    on_primary = run_episode(primary, policy)
    on_mirror = run_episode(mirror, policy)

    assert np.array_equal(on_primary.shares, on_mirror.shares), (
        f"{policy_name} took different actions on the two halves of the pair"
    )
    assert np.array_equal(on_primary.trajectory, on_mirror.trajectory)
    # ...while the halves really are mirrored, so the identity is not vacuous.
    assert np.array_equal(on_mirror.walks, -on_primary.walks)
    assert np.max(np.abs(on_primary.walks)) > 1.0


def test_the_pair_wrapper_checks_observation_identity_and_it_is_live():
    """The per-step check fires the moment a half's observation carries price.

    Non-vacuity for the assertion training relies on: perturb the mirror's
    observation by a shock-dependent amount and the pair must refuse to continue.
    """
    pair = AntitheticPair(_env(908))
    pair.reset()
    for fraction in twap_fractions(N_BINS)[:3]:
        pair.step(fraction * pair.primary.order_size)  # healthy: raw env, shares

    leaky = AntitheticPair(_env(908))
    leaky.reset()
    real = type(leaky.mirror)._observation

    def leaking(self):
        observation = real(self)
        observation[1] += 1e-12 * self._walk  # a price-dependent whisper
        return observation

    type(leaky.mirror)._observation = leaking
    try:
        with pytest.raises(PairDiverged):
            for fraction in twap_fractions(N_BINS):
                leaky.step(fraction * ORDER_SIZE)
    finally:
        type(leaky.mirror)._observation = real


def test_the_pair_wrapper_checks_the_negation_and_it_is_live():
    """If the mirror ever draws fresh numbers, the pair refuses to continue."""
    pair = AntitheticPair(_env(909))
    pair.reset()
    # Swap the mirror's negating proxy for a plain generator at a different
    # stream: same distribution, not the same numbers negated.
    pair.mirror._rng = pool_rng(ROOT_SEED, M3_DIAGNOSTIC_POOL, 910)
    with pytest.raises(PairDiverged):
        for fraction in twap_fractions(N_BINS):
            pair.step(fraction * ORDER_SIZE)


# ---------------------------------------------------------------------------
# On the average, the noise cancels exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FRACTION_SEQUENCES))
def test_the_averaged_reward_is_the_control_variates_reward_to_the_ulp(name):
    """Two estimators, one number — without the analytic noise form.

    The control variate subtracts M1a's identity; the pair averages a mirrored
    realisation. Same actions, same stream, and the two per-step rewards agree
    to a couple of ulps of the noise removed. That is the mechanism confirmed
    directly, before any training run infers it from an outcome.
    """
    fractions = FRACTION_SEQUENCES[name]
    averaged, schedule = _run(FractionAction(AntitheticPair(_env(911))), fractions)
    variate, schedule_v = _run(FractionAction(DeterministicReward(_env(911))), fractions)
    noisy, _ = _run(FractionAction(_env(911)), fractions)

    removed = float(np.max(np.abs(noisy - variate)))
    assert removed > 1.0, "this stream carries no noise worth cancelling"
    assert float(np.max(np.abs(averaged - variate))) <= 1e-12 * removed
    assert np.array_equal(schedule, schedule_v)


@pytest.mark.parametrize("name", sorted(FRACTION_SEQUENCES))
def test_the_averaged_reward_is_stream_independent(name):
    """Two unrelated streams, the same actions, the same averaged rewards."""
    fractions = FRACTION_SEQUENCES[name]
    a, _ = _run(FractionAction(AntitheticPair(_env(912))), fractions)
    b, _ = _run(FractionAction(AntitheticPair(_env(913))), fractions)
    noisy_a, _ = _run(FractionAction(_env(912)), fractions)
    noisy_b, _ = _run(FractionAction(_env(913)), fractions)
    noisy_spread = float(np.max(np.abs(noisy_a - noisy_b)))
    assert noisy_spread > 1.0
    assert float(np.max(np.abs(a - b))) <= 1e-12 * noisy_spread


@pytest.mark.parametrize("name", sorted(FRACTION_SEQUENCES))
def test_the_averaged_reward_sums_to_minus_the_analytic_objective(name):
    """Invariant 7 for this estimator: what is left is exactly ``-(E + lambda V)``."""
    rewards, trajectory = _run(FractionAction(AntitheticPair(_env(914))), FRACTION_SEQUENCES[name])
    objective = schedule_moments(trajectory, MARKET, order_size=ORDER_SIZE).objective(LAMBDA)
    scale = float(np.sum(np.abs(rewards)))
    assert float(np.sum(rewards)) == pytest.approx(-objective, rel=0.0, abs=IDENTITY_RTOL * scale)


def test_the_pair_leaves_the_realised_schedule_untouched():
    """It changes the estimator, not the world."""
    fractions = twap_fractions(N_BINS)
    _, plain = _run(FractionAction(_env(915)), fractions)
    _, paired = _run(FractionAction(AntitheticPair(_env(915))), fractions)
    assert np.array_equal(plain, paired)


# ---------------------------------------------------------------------------
# The reward-variance evidence
# ---------------------------------------------------------------------------


def test_the_ledger_measures_the_cancellation_rather_than_inferring_it():
    """Both halves' returns are recorded; the averaged variance is dust."""
    ledger = PairLedger()
    fractions = twap_fractions(N_BINS)
    for stream in (916, 917, 918, 919):
        _run(FractionAction(AntitheticPair(_env(stream), ledger)), fractions)
    assert ledger.pending == 4
    stats = ledger.close_update()
    assert ledger.pending == 0 and ledger.updates == [stats]
    assert stats.episodes == 4
    assert stats.sampled_variance > 100.0, "no noise to speak of on the primary half"
    assert stats.mirror_variance == pytest.approx(stats.sampled_variance, rel=1e-9)
    assert stats.averaged_variance <= 1e-20 * stats.sampled_variance
    assert stats.cancelled_mean_square > 100.0
    assert stats.variance_ratio <= 1e-20
    as_dict = stats.as_dict()
    assert set(as_dict) >= {"episodes", "sampled_variance", "averaged_variance", "variance_ratio"}


def test_the_ledger_is_honest_about_too_few_episodes():
    ledger = PairLedger()
    stats = ledger.close_update()
    assert stats.episodes == 0
    assert np.isnan(stats.sampled_variance) and np.isnan(stats.variance_ratio)
    ledger.record(1.0, -1.0)
    one = ledger.close_update()
    assert one.episodes == 1 and np.isnan(one.averaged_variance)
    assert one.cancelled_mean_square == 1.0


# ---------------------------------------------------------------------------
# Where the pair sits in the factory, and what it spends
# ---------------------------------------------------------------------------


def _chain(env) -> list[type]:
    classes = []
    while hasattr(env, "env"):
        classes.append(type(env))
        env = env.env
    classes.append(type(env))
    return classes


def test_the_factory_puts_the_estimator_below_the_fraction_and_the_scale():
    """``RewardScale(FractionAction(estimator(ExecutionEnv)))`` — both estimators.

    Below the scale so the estimator works in bps; below the fraction so the
    mirror is handed the same *shares* the primary received, converted once.
    """
    ledger = PairLedger()
    kwargs = dict(
        root_seed=ROOT_SEED, pool=M3_DIAGNOSTIC_POOL, stream_index=920,
        reward_scale=M3.reward_scale,
    )
    paired = execution_env_factory(
        MARKET, ORDER_SIZE, LAMBDA, reward_wrapper=antithetic_reward(ledger), **kwargs
    )()
    variate = execution_env_factory(
        MARKET, ORDER_SIZE, LAMBDA, reward_wrapper=deterministic_reward, **kwargs
    )()
    assert _chain(paired) == [RewardScale, FractionAction, AntitheticPair, ExecutionEnv]
    assert _chain(variate) == [RewardScale, FractionAction, DeterministicReward, ExecutionEnv]
    assert paired.unwrapped.seed_address == (ROOT_SEED, M3_DIAGNOSTIC_POOL, 920)
    assert isinstance(paired.env.env.mirror, MirrorEnv)
    assert paired.env.env.ledger is ledger

    # And the composed reward is the scaled average — the ledger sees bps.
    rewards, _ = _run(paired, twap_fractions(N_BINS))
    bare, _ = _run(FractionAction(AntitheticPair(_env(920))), twap_fractions(N_BINS))
    assert rewards == pytest.approx(M3.reward_scale * bare, rel=1e-15)
    stats = ledger.close_update()
    assert stats.episodes == 1
    assert stats.averaged_variance != stats.averaged_variance  # nan: one pair


def test_the_pair_spends_exactly_its_addressed_stream_twice_and_nothing_else():
    """Invariant 5: the mirror is the *same* address, so the pair costs one stream.

    Recorded through the conftest ledger of every address the env resolves: a
    reset of the pair must show the addressed stream exactly twice (primary and
    mirror) and no other stream at all — a mirror that quietly opened stream+1
    would spend a stream a committed result may be addressed by.
    """
    before = len(RESOLVED_SEED_ADDRESSES)
    pair = AntitheticPair(_env(921))
    pair.reset()
    resolved = RESOLVED_SEED_ADDRESSES[before:]
    assert resolved == [(ROOT_SEED, M3_DIAGNOSTIC_POOL, 921)] * 2
    assert pair.mirror.seed_address == pair.primary.seed_address


def test_the_diagnostic_pool_was_appended_not_inserted():
    """Pool order fixes spawn keys; M3's pool must not have moved M2's."""
    assert POOLS[:4] == ("train", "eval", "m1/differential", "m2/diagnostic")
    assert POOLS[4] == M3_DIAGNOSTIC_POOL


# ---------------------------------------------------------------------------
# The committed config: everything else identical to M2
# ---------------------------------------------------------------------------

#: The fields the brief pre-states as different between the validation config
#: and M2's control-variate config. Anything else differing is a red test.
PRESTATED_DIFFERENCES = {"path", "milestone", "estimator", "runtime", "gate", "seeding"}


def test_the_validation_config_differs_from_m2_only_where_the_brief_says():
    m2 = M2.as_dict()
    m3 = M3.as_dict()
    differing = sorted(key for key in set(m2) | set(m3) if m2.get(key) != m3.get(key))
    assert set(differing) <= PRESTATED_DIFFERENCES, (
        f"the validation config differs from m2_ppo.yaml in {differing}; only "
        f"{sorted(PRESTATED_DIFFERENCES)} may differ"
    )
    # Seeding differs in the seed count and nothing else.
    m2_seeding, m3_seeding = dict(m2["seeding"]), dict(m3["seeding"])
    assert m2_seeding.pop("n_seeds") == 5 and m3_seeding.pop("n_seeds") == 10
    assert m2_seeding == m3_seeding
    assert M3.milestone == "M3"
    assert M3.estimator.regime == ANTITHETIC and M3.estimator.antithetic
    assert "antithetic" in M3.estimator.claim.lower()
    assert M3.gate is not None
    assert M3.gate.median_gap_fraction == 0.002
    assert M3.gate.reference == M2.results_metrics
    assert M3.ppo == M2.ppo
    assert M3.tolerances == M2.tolerances
    assert M3.lambda_risk == M2.lambda_risk
    M3.verify_lambda_rule()


def test_the_estimator_block_accepts_either_spelling_and_rejects_contradiction():
    assert Estimator("control_variate", "deterministic").control_variate
    assert Estimator("sampled", "sampled").sampled
    assert not Estimator("antithetic", "antithetic").control_variate
    assert set(REGIMES) == {"sampled", "control_variate", "antithetic"}
    with pytest.raises(ValueError):
        Estimator("bootstrap", "no")
    from temper.eval.experiment import _estimator

    assert _estimator({"control_variate": True, "claim": "x"}).regime == "control_variate"
    assert _estimator({"control_variate": False, "claim": "x"}).regime == "sampled"
    assert _estimator({"regime": "antithetic", "claim": "x"}).antithetic
    with pytest.raises(ValueError):
        _estimator({"regime": "sampled", "control_variate": True, "claim": "x"})
    with pytest.raises(ValueError):
        _estimator({"claim": "x"})
    assert M2.estimator.as_dict()["regime"] == "control_variate"
    assert M2.estimator.as_dict()["control_variate"] is True
