"""M5 task 2 — the third seam's acceptance, across **three** worlds.

M4a's regression re-ran one M3 seed and required its grade bitwise. M4b did it
twice. M5 does it three times, because it adds a third injected seam to an env
that already had two — and because the third one is the only one that is
*correlated with the price shocks by design*.

What the bitwise check is actually testing, since it is not obvious
-------------------------------------------------------------------
Not "is the code still deterministic". The mechanism is this:

The env composes each bin's shock as ``rho * s_{k-1} + sqrt(1 - rho^2) * e_k``,
where ``e_k`` is one ``standard_normal()`` off the **price** generator — same
order, same count, same generator as every milestone before — and ``s`` comes from
``m5/signal-train`` or ``m5/signal-eval``, which are *new pools*. Because the
signal has its own spawn key, drawing it cannot advance the price generator; and
because ``rho = 0`` makes the composition ``0.0 * s + 1.0 * e``, which is ``e`` in
IEEE, a signal-free world's shocks are not merely statistically unchanged but
**identical to the bit**.

So the three retrains below are a measurement of exactly one proposition: *the
signal seam does not reach into the price generator*. If a seed moves, the
correlation the milestone is about to measure would be partly manufactured by two
noise sources sharing a stream rather than by the model — the seam would look
right and mean nothing, an agent would be reading the shock rather than a
prediction of it, and every M5 number would be a claim about a world nobody
described. That is the defect, and a moved digit is the only cheap way to see it.

The same class of check earned its place in M4a: the antithetic mirror was
charging the wrong world, and a per-step identity caught it at 0.06 bps against a
1e-12 band before a single seed was spent. This is that check, one seam along.

Three ways it could have failed, one fast test each
---------------------------------------------------
* **The signal variate out of the price generator.** Prevented by construction —
  the signal has its own pools — and checked here by seed address, and then by the
  trained digits.
* **The composition not being the identity at ``rho = 0``.** ``0.0 * s`` is a
  signed zero and ``1.0 * e`` is ``e``, so the arithmetic is exact — but "so it is
  fine" is an argument and the shocks are a measurement.
* **An observation that grew where it should not have.** A signal-free world that
  acquired a constant third coordinate would be quiet and would still move every
  trained digit. The width is asserted per world.

And one thing that must **fail**
---------------------------------
``ExecutionEnv``'s observation-minimality guard has refused a price-bearing
observation since M1a. M5's observation *is* price-bearing — that is the milestone
— and task 2 does not amend the guard. So the guard is run verbatim against an M5
env here and required to **refuse**, with a measurement of what it is refusing.
If it passed, the guard would not be testing what it claims and that would be the
finding rather than a convenience. Task 3 narrows it; seeing it refuse first is
the evidence that the narrowing is real.

The retrains are marked ``training``: three 5 M-step seeds, about an hour on the
reference box. Milestone acceptance, not the per-commit gate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from temper.agents.execution import PPOPolicy
from temper.env import (
    NO_SIGNAL_STREAM,
    SHOCK_KEY,
    ExecutionEnv,
    SignalStream,
    signal_stream,
)
from temper.eval import antithetic
from temper.eval.antithetic import (
    AntitheticPair,
    MirrorEnv,
    NegatedSignal,
    PairDiverged,
    mirror_of,
)
from temper.eval.experiment import load_experiment
from temper.eval.grading import grade_policy
from temper.eval.sweep import (
    evaluation_signal,
    grade_liquidity,
    liquidity_reference,
    train_seed,
    training_liquidity,
    training_signal,
)
from temper.env import impact_for
from temper.oracle import (
    LINEAR_ENCODING,
    POWER_LAW_ENCODING,
    Market,
    NoSignal,
    OneStepSignal,
    SymbolParams,
    twap_trajectory,
)
from temper.seeding import (
    M5_DIFFERENTIAL_POOL,
    SIGNAL_EVAL_POOL,
    SIGNAL_TRAIN_POOL,
)

from .conftest import REPO_ROOT

#: The world a signal-bearing env is named by, for the rows below that are about
#: the seam rather than about a committed result.
SIGNAL_WORLD = "power_law+signal"

#: The three committed points, one per world. All at 10^-3.5, all antithetic, all
#: ten-seed sweeps whose seed 0 is retrained here.
POINTS = {
    LINEAR_ENCODING: (
        REPO_ROOT / "configs" / "m3_frontier" / "lambda_1e-3.5.yaml",
        REPO_ROOT / "results" / "m3_frontier" / "lambda_1e-3.5.json",
    ),
    POWER_LAW_ENCODING: (
        REPO_ROOT / "configs" / "m4a_power_law.yaml",
        REPO_ROOT / "results" / "m4a_power_law.json",
    ),
    SIGNAL_WORLD: (
        REPO_ROOT / "configs" / "m4b_liquidity.yaml",
        REPO_ROOT / "results" / "m4b_liquidity.json",
    ),
}


def _load(world: str):
    config, result = POINTS[world]
    return load_experiment(config), json.loads(result.read_text(encoding="utf-8"))


def _case_market() -> Market:
    return Market(
        params=SymbolParams(
            adv=6e7, sigma=0.0155, half_spread=0.3, eta=0.142, gamma=0.314
        ),
        horizon_hours=6.5,
        n_bins=13,
    )


def _episode(env, schedule) -> tuple[np.ndarray, np.ndarray]:
    """Run one full episode; return the observations and the standardised shocks."""
    observation, _ = env.reset(seed=env.seed_address[2])
    seen, walk = [observation], []
    for shares in schedule:
        observation, _, _, _, info = env.step(float(shares))
        seen.append(observation)
        walk.append(info[SHOCK_KEY])
    increments = np.diff(np.concatenate(([0.0], walk)))
    return np.array(seen), increments / (env.market.sigma_bin * 1e4)


# ---------------------------------------------------------------------------
# The seam is inert where there is no signal — the mechanism, measured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stream",
    [
        None,
        NO_SIGNAL_STREAM,
        signal_stream(OneStepSignal(0.0), SIGNAL_TRAIN_POOL),
        signal_stream(OneStepSignal(0.9, bins_ahead=0), SIGNAL_TRAIN_POOL),
    ],
    ids=["absent", "explicit-none", "rho-zero", "already-landed"],
)
def test_an_uninformative_signal_leaves_the_shocks_bitwise_unchanged(stream):
    """The whole reason the pools are separate, at the level of the draws.

    Four ways to have no usable signal, including a *correlation of 0.9* pointed
    at a shock that has already landed. All four must leave the price path exactly
    where it was: same draws, same order, same count off the same generator, and a
    composition that is the identity.

    The fourth case is the sharpest. It is not signal-free — the law has a
    correlation and the stream draws from it — and it still must not move a shock
    by a bit, because what it predicts was already charged on inventory the
    previous decision fixed.
    """
    market = _case_market()
    schedule = -np.diff(twap_trajectory(market, 100_000.0))

    def run(signal):
        env = ExecutionEnv(
            market,
            100_000.0,
            1e-4,
            signal=signal,
            root_seed=20260824,
            pool=M5_DIFFERENTIAL_POOL,
            stream_index=3,
        )
        return _episode(env, schedule)

    baseline_observations, baseline_shocks = run(None)
    observations, shocks = run(stream)

    assert np.array_equal(shocks, baseline_shocks), (
        "the price path moved through a seam that adds no information; the signal "
        "is reaching the price generator"
    )
    assert np.array_equal(observations, baseline_observations)
    assert observations.shape[1] == 2, (
        "a world with nothing to see grew an observation coordinate; the width is "
        "supposed to be a statement about what is knowable, not a constant"
    )


def test_the_three_noise_sources_are_addressed_in_three_different_pools():
    """The failure that would be invisible, made arithmetic — for a third source.

    Same root seed, same stream index, three different pools. Asserted on the
    *addresses* rather than on a sample, because two streams that happened to
    agree on their first few draws would pass a sampled check and still be the
    same stream.
    """
    env = ExecutionEnv(
        _case_market(),
        100_000.0,
        1e-4,
        signal=training_signal(load_experiment(REPO_ROOT / "configs" / "m5_alpha.yaml")),
        root_seed=5,
        pool="train",
        stream_index=17,
    )
    addresses = [env.seed_address, env.liquidity_address, env.signal_address]
    roots = {root for root, _, _ in addresses}
    indices = {index for _, _, index in addresses}
    pools = [pool for _, pool, _ in addresses]

    assert roots == {5} and indices == {17}, "the three addresses drifted apart"
    assert len(set(pools)) == 3, f"two noise sources share a pool: {pools}"
    assert pools[0] == "train" and pools[2] == SIGNAL_TRAIN_POOL


def test_training_and_evaluation_signals_are_different_pools():
    """Invariant 5 doing M5's out-of-sample work, by spawn key rather than stride."""
    experiment = load_experiment(REPO_ROOT / "configs" / "m5_alpha.yaml")
    train, evaluate = training_signal(experiment), evaluation_signal(experiment)
    assert train.pool == SIGNAL_TRAIN_POOL
    assert evaluate.pool == SIGNAL_EVAL_POOL
    assert train.pool != evaluate.pool
    assert train.signal == evaluate.signal == experiment.signal
    assert train.informative and evaluate.informative


def test_the_signal_stream_refuses_a_law_without_a_pool_and_an_unknown_pool():
    """A law with no address is the defect this object exists to make impossible."""
    with pytest.raises(ValueError, match="unknown seed pool"):
        SignalStream(signal=OneStepSignal(0.01), pool="wherever")
    with pytest.raises(TypeError, match="AlphaSignal"):
        SignalStream(signal="0.01")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pinned signal index"):
        SignalStream(signal=NoSignal(), index=-1)


# ---------------------------------------------------------------------------
# The seam is right where there *is* a signal
# ---------------------------------------------------------------------------


def test_the_env_realises_the_joint_law_the_oracle_defines():
    """``Corr(s_k, xi_{k+1}) = rho``, and every other pair independent.

    The env composes its shocks from two generators in two pools; the oracle
    composes the same law in one call from one. Deliberately different routes —
    which is what makes this a measurement of the env rather than a restatement of
    the oracle — and they have to land on the same distribution.

    ``rho = 0.4`` rather than the milestone's 0.01, because this is a check on the
    *seam's* arithmetic and not on the milestone's parameter: at 1e-4 of explained
    variance the correlation is two orders inside the sampling error of any path
    count this suite can afford, and a test that cannot see the effect cannot see
    it disappear either.
    """
    market = _case_market()
    rho = 0.4
    env = ExecutionEnv(
        market,
        100_000.0,
        1e-4,
        signal=signal_stream(OneStepSignal(rho), SIGNAL_TRAIN_POOL),
        root_seed=20260824,
        pool=M5_DIFFERENTIAL_POOL,
        stream_index=0,
    )
    schedule = -np.diff(twap_trajectory(market, 100_000.0))

    signals, shocks = [], []
    for episode in range(12_000):
        env.reset(seed=episode)
        signals.append(env.signals)
        _, standardised = _episode(env, schedule)
        shocks.append(standardised)
    signals, shocks = np.array(signals), np.array(shocks)

    assert shocks.std() == pytest.approx(1.0, abs=0.02), (
        "the composed shock lost its unit variance; the market the signal is a "
        "signal about is no longer the market every earlier milestone ran in"
    )
    assert signals.std() == pytest.approx(1.0, abs=0.02)

    bins = market.n_bins
    for k in range(bins):
        for j in range(bins):
            observed = float(np.corrcoef(signals[:, k], shocks[:, j])[0, 1])
            expected = rho if j == k + 1 else 0.0
            assert observed == pytest.approx(expected, abs=0.04), (
                f"corr(s_{k}, xi_{j}) = {observed:.4f}, expected {expected}"
            )


def test_the_observation_carries_the_signal_for_the_bin_about_to_be_decided():
    """Ordering, and it is load-bearing rather than cosmetic.

    ``(k, x_k, s_k)`` is a sufficient statistic only if the agent sees ``s_k`` in
    time to act on it. The observation returned by ``reset`` must therefore carry
    ``s_0`` and the one returned by step ``k`` must carry ``s_{k+1}`` — an
    off-by-one here would hand the agent a prediction of a shock that had already
    been charged, which is worth nothing and would look like a training failure.
    """
    market = _case_market()
    env = ExecutionEnv(
        market,
        100_000.0,
        1e-4,
        signal=signal_stream(OneStepSignal(0.3), SIGNAL_TRAIN_POOL),
        root_seed=20260824,
        pool=M5_DIFFERENTIAL_POOL,
        stream_index=1,
    )
    observation, _ = env.reset(seed=1)
    published = env.signals
    assert observation.shape == (3,)
    assert observation[2] == published[0]
    for index in range(market.n_bins):
        observation, _, _, _, _ = env.step(100_000.0 / market.n_bins)
        expected = published[index + 1] if index + 1 < market.n_bins else 0.0
        assert observation[2] == expected, f"step {index} published the wrong signal"
    assert observation[2] == 0.0, "the terminal observation must carry no signal"

    assert env.signals is not env.signals, "callers got the env's own buffer"
    assert env.signals.shape == (market.n_bins,)


def test_the_mirror_is_built_in_the_same_signal_world_as_the_primary():
    """M4a's lesson, applied to the third seam before an estimator can be bitten.

    A mirror that defaulted would be a signal-free env averaged against a
    signal-bearing primary. **Amended by task 3:** the mirror is handed the
    primary's signal *negated* rather than shared, so what "same world" means here
    is same law, same pool, same index, opposite draw.
    """
    stream = signal_stream(OneStepSignal(0.2), SIGNAL_TRAIN_POOL)
    env = ExecutionEnv(
        _case_market(),
        100_000.0,
        1e-4,
        signal=stream,
        root_seed=11,
        pool=M5_DIFFERENTIAL_POOL,
        stream_index=4,
    )
    mirror = mirror_of(env)
    assert isinstance(mirror.signal.signal, NegatedSignal)
    assert mirror.signal.signal.base is stream.signal
    assert mirror.signal.pool == stream.pool and mirror.signal.index == stream.index
    assert mirror.signal_address == env.signal_address
    assert mirror.observation_space.shape == env.observation_space.shape
    # The correlation is NOT negated — negating it would put the mirror in a
    # different world at the same address rather than reflecting this one.
    assert mirror.signal.signal.correlation() == stream.signal.correlation()
    assert mirror.signal.signal.lag == stream.signal.lag


def test_a_signal_free_mirror_is_the_object_it_always_was():
    """The wrap is applied only where there is something to negate.

    M0 through M4b get the identical :class:`SignalStream`, so this constructor is
    provably unchanged for them rather than merely equivalent.
    """
    env = ExecutionEnv(
        _case_market(),
        100_000.0,
        1e-4,
        root_seed=11,
        pool=M5_DIFFERENTIAL_POOL,
        stream_index=4,
    )
    assert mirror_of(env).signal is env.signal is NO_SIGNAL_STREAM


def test_negating_the_signal_is_what_keeps_the_shock_negation_exact():
    """Why the mirror negates the signal, measured rather than argued.

    With ``xi = rho s + sqrt(1 - rho^2) e`` and only the price generator negated,
    a mirror *sharing* the signal realises ``rho s - sqrt(1 - rho^2) e``, which is
    not ``-xi``. Negating both gives ``-xi`` to the bit. So the choice is forced by
    the pairing's one exact property, not by preference — and this shows both
    sides of it rather than only the one that works.
    """
    market = _case_market()
    schedule = -np.diff(twap_trajectory(market, 100_000.0))
    stream = signal_stream(OneStepSignal(0.4), SIGNAL_TRAIN_POOL)

    def primary():
        return ExecutionEnv(
            market,
            100_000.0,
            1e-4,
            signal=stream,
            root_seed=11,
            pool=M5_DIFFERENTIAL_POOL,
            stream_index=5,
        )

    # The arrangement that works: the pair, as built.
    pair = AntitheticPair(primary())
    pair.reset(seed=5)
    for shares in schedule:
        pair.step(float(shares))
    assert np.array_equal(pair.mirror.signals, -pair.primary.signals)

    # The arrangement that does not: a mirror on the *shared* signal. Built by
    # hand, because `mirror_of` no longer produces one.
    shared = MirrorEnv(
        market,
        100_000.0,
        1e-4,
        temporary_impact=primary().temporary_impact,
        signal=stream,
        root_seed=11,
        pool=M5_DIFFERENTIAL_POOL,
        stream_index=5,
    )
    reference, shared_env = primary(), shared
    reference.reset(seed=5)
    shared_env.reset(seed=5)
    for shares in schedule:
        _, _, _, _, info = reference.step(float(shares))
        _, _, _, _, m_info = shared_env.step(float(shares))
    assert m_info[SHOCK_KEY] != -info[SHOCK_KEY], (
        "a mirror sharing the signal happened to negate the shock anyway, so this "
        "test cannot distinguish the two arrangements"
    )


def test_action_identity_has_ended_and_the_replacement_is_exact():
    """The retirement, and what stands in its place.

    Retired: "the two halves see the same observation". Standing in its place:
    every coordinate but the signal is bitwise equal, and the signal is the exact
    negation. Both halves of that are checked here, and the retirement is shown to
    be *material* rather than nominal — a policy shown the mirror's observation
    would choose a different action, which is the whole content of the claim that
    ended.
    """
    market = _case_market()
    schedule = -np.diff(twap_trajectory(market, 100_000.0))
    pair = AntitheticPair(
        ExecutionEnv(
            market,
            100_000.0,
            1e-4,
            signal=signal_stream(OneStepSignal(0.4), SIGNAL_TRAIN_POOL),
            root_seed=11,
            pool=M5_DIFFERENTIAL_POOL,
            stream_index=6,
        )
    )
    assert pair.mirrors_signal

    def tilt(observation):
        """A policy that reads the signal — the simplest one that can."""
        return float(observation[2])

    observation, _ = pair.reset(seed=6)
    differing = 0
    for shares in schedule:
        mirrored = pair.mirror._observation()
        assert np.array_equal(observation[:2], mirrored[:2]), (
            "the halves disagree about time or inventory, which no amendment "
            "permits"
        )
        assert mirrored[2] == -observation[2]
        if tilt(observation) != tilt(mirrored):
            differing += 1
        observation, _, _, _, _ = pair.step(float(shares))

    assert differing >= market.n_bins - 1, (
        f"a signal-reading policy chose the same action on both halves at "
        f"{market.n_bins - differing} of {market.n_bins} decision points; the "
        "retirement of action identity would be nominal rather than real"
    )


def test_action_identity_is_kept_verbatim_where_it_still_holds():
    """Three milestones keep the check they were built under.

    The retirement is scoped to worlds with an informative signal. A signal-free
    world — and a world whose signal is pointed at an already-committed shock, so
    the observation never grows — must still satisfy the old bitwise assertion,
    and the pair must be asserting it rather than skipping to the weaker rule.
    """
    market = _case_market()
    schedule = -np.diff(twap_trajectory(market, 100_000.0))
    for label, stream in (
        ("absent", None),
        (
            "already-committed",
            signal_stream(OneStepSignal(0.9, bins_ahead=0), SIGNAL_TRAIN_POOL),
        ),
    ):
        pair = AntitheticPair(
            ExecutionEnv(
                market,
                100_000.0,
                1e-4,
                signal=stream,
                root_seed=11,
                pool=M5_DIFFERENTIAL_POOL,
                stream_index=7,
            )
        )
        assert not pair.mirrors_signal, label
        observation, _ = pair.reset(seed=7)
        for shares in schedule:
            assert np.array_equal(observation, pair.mirror._observation()), label
            observation, _, _, _, _ = pair.step(float(shares))


def test_the_pair_refuses_a_mirror_on_a_fresh_signal_path():
    """The replacement has teeth: "different" is not enough, it must be the negation.

    A mirror drawing its own signal would disagree with the primary on exactly the
    coordinate the amendment permits them to disagree on — and would be averaging
    two unrelated worlds. This is the M4a mirror bug wearing M5's clothes, and the
    exact-negation form of the check is what refuses it.
    """
    market = _case_market()
    pair = AntitheticPair(
        ExecutionEnv(
            market,
            100_000.0,
            1e-4,
            signal=signal_stream(OneStepSignal(0.4), SIGNAL_TRAIN_POOL),
            root_seed=11,
            pool=M5_DIFFERENTIAL_POOL,
            stream_index=8,
        )
    )
    pair.mirror.signal = pair.mirror.signal.pinned_to(4242)
    pair.mirror._signal_rng = None
    with pytest.raises(PairDiverged, match="not the exact negation"):
        pair.reset(seed=8)


def test_the_pair_refuses_a_mirror_in_another_world_before_a_step(monkeypatch):
    """M4a's bug, refused at construction now rather than by a per-step band.

    That bug cost a run: `mirror_of` rebuilt the mirror without the primary's
    impact model, so a Phase-1 env was averaged against a power-law primary, and
    it was found by a cancellation band four orders wide rather than by anything
    structural. The pair now asks the question once, at construction, where the
    answer is free — so a fourth injected seam cannot repeat it quietly.

    `mirror_of` is replaced rather than the mirror mutated, because the check runs
    inside ``__init__`` and mutating afterwards would test nothing.
    """
    market = _case_market()
    stream = signal_stream(OneStepSignal(0.4), SIGNAL_TRAIN_POOL)
    env = ExecutionEnv(
        market,
        100_000.0,
        1e-4,
        temporary_impact=impact_for(POWER_LAW_ENCODING, market, 100_000.0),
        signal=stream,
        root_seed=11,
        pool=M5_DIFFERENTIAL_POOL,
        stream_index=9,
    )
    assert AntitheticPair(env).mirror.cost_encoding == POWER_LAW_ENCODING

    def defaulting_mirror(primary):
        """M4a's defect verbatim: the impact model is not handed over."""
        root_seed, pool, index = primary.seed_address
        return MirrorEnv(
            primary.market,
            primary.order_size,
            primary.lambda_risk,
            signal=primary.signal,
            root_seed=root_seed,
            pool=pool,
            stream_index=index,
        )

    monkeypatch.setattr(antithetic, "mirror_of", defaulting_mirror)
    with pytest.raises(PairDiverged, match="must be one world"):
        AntitheticPair(env)


def test_the_pair_refuses_a_mirror_at_another_signal_address(monkeypatch):
    """A mirrored signal is the same stream negated, never a second stream.

    The exact-negation check catches a mirror on a different *index*; this catches
    one in a different *pool*, which would draw an unrelated path that the
    per-step check would also refuse — but one bin later and with a message about
    arithmetic rather than about addressing. Refusing it at construction says the
    right thing about the right thing.
    """
    market = _case_market()
    stream = signal_stream(OneStepSignal(0.4), SIGNAL_TRAIN_POOL)
    env = ExecutionEnv(
        market,
        100_000.0,
        1e-4,
        signal=stream,
        root_seed=11,
        pool=M5_DIFFERENTIAL_POOL,
        stream_index=10,
    )
    assert AntitheticPair(env).mirror.signal_address == env.signal_address

    def wrong_pool(primary):
        root_seed, pool, index = primary.seed_address
        moved = SignalStream(
            signal=NegatedSignal(primary.signal.signal),
            pool=SIGNAL_EVAL_POOL,
            index=primary.signal.index,
        )
        assert moved.pool != primary.signal.pool
        return MirrorEnv(
            primary.market,
            primary.order_size,
            primary.lambda_risk,
            temporary_impact=primary.temporary_impact,
            signal=moved,
            root_seed=root_seed,
            pool=pool,
            stream_index=index,
        )

    monkeypatch.setattr(antithetic, "mirror_of", wrong_pool)
    with pytest.raises(PairDiverged, match="never a second stream"):
        AntitheticPair(env)


# ---------------------------------------------------------------------------
# The guard that must refuse, and what it is refusing
# ---------------------------------------------------------------------------


def test_the_observation_minimality_guard_refuses_the_signal_world():
    """Task 2 must leave the M1a guard **red** on M5's env. Task 3 amends it.

    The guard is ``tests/test_env_identities.py``'s: run the same deterministic
    schedule against two shock streams and require the observation sequences to be
    bitwise equal while the shocks are not. It is run verbatim here, and it must
    fail — because M5's observation genuinely does change with the stream, and it
    changes with it *because* it carries information about the prices.

    If this passed, the guard would not be testing what it claims: it would mean
    an observation that predicts a shock can be added without the guard noticing,
    and every reassurance the guard has provided since M1a would be worth less
    than it looks. Seeing it refuse is what makes task 3's amendment a narrowing
    of something real rather than the deletion of a formality.

    **Task 3 has since landed that amendment**, and this test is kept unchanged as
    the evidence it rests on: the *unamended* clause refuses this env, and the
    amended one — pin the signal, vary the price, then require every seam
    coordinate to correlate only with shocks the current decision can still act on
    — permits it (``tests/test_m5_observation_guard.py``). Two clauses, one
    refusing and one permitting the same env, is what "narrowed" means here.

    The refusal is quantified rather than merely observed, because "different" is
    a weak thing to know: the coordinate that differs is the third one, it is the
    only one that differs, and it is correlated with a shock that has **not yet
    landed** and with no shock that has.
    """
    market = _case_market()
    schedule = -np.diff(twap_trajectory(market, 100_000.0))
    stream = signal_stream(OneStepSignal(0.4), SIGNAL_TRAIN_POOL)

    def run(stream_index):
        env = ExecutionEnv(
            market,
            100_000.0,
            1e-4,
            signal=stream,
            root_seed=20260824,
            pool=M5_DIFFERENTIAL_POOL,
            stream_index=stream_index,
        )
        observations, shocks = _episode(env, schedule)
        return observations, shocks, env.signals

    first, first_shocks, first_signals = run(900)
    second, second_shocks, second_signals = run(901)

    assert not np.array_equal(first_shocks, second_shocks), (
        "the two streams drew the same shocks; the comparison is vacuous"
    )
    # The guard, verbatim — and it does not hold.
    assert not np.array_equal(first, second), (
        "the observation-minimality guard PASSED on M5's env. That is the "
        "milestone's finding rather than a convenience: an observation that "
        "predicts a price shock was added and the guard did not notice, so it was "
        "not testing what it claims and task 3 would be amending a formality. "
        "Stop and report."
    )

    # What differs, exactly: the signal coordinate and nothing else.
    assert np.array_equal(first[:, :2], second[:, :2]), (
        "the schedule's own coordinates moved with the shock stream; that would be "
        "a leak of a completely different kind"
    )
    assert not np.array_equal(first[:, 2], second[:, 2])
    assert not np.array_equal(first_signals, second_signals)

    # And the coordinate that differs is price-bearing in exactly the way the
    # milestone says: about the shock that has not landed, and about no other.
    assert float(np.corrcoef(first_signals[:-1], first_shocks[1:])[0, 1]) != 0.0
    assert np.corrcoef(first_signals, first_shocks)[0, 1] == pytest.approx(0.0, abs=0.9)


def test_the_guard_still_holds_everywhere_it_always_did():
    """The refusal above must be specific to M5, not a guard that stopped working.

    Same comparison, same code path, on the two worlds the guard has always
    covered plus a signal seam that carries no information. All three must still
    be bitwise identical across shock streams.
    """
    market = _case_market()
    schedule = -np.diff(twap_trajectory(market, 100_000.0))

    for label, stream in (
        ("absent", None),
        ("explicit-none", NO_SIGNAL_STREAM),
        ("already-landed", signal_stream(OneStepSignal(0.9, bins_ahead=0), SIGNAL_TRAIN_POOL)),
    ):
        def run(stream_index):
            env = ExecutionEnv(
                market,
                100_000.0,
                1e-4,
                signal=stream,
                root_seed=20260824,
                pool=M5_DIFFERENTIAL_POOL,
                stream_index=stream_index,
            )
            return _episode(env, schedule)

        first, first_shocks = run(910)
        second, second_shocks = run(911)
        assert not np.array_equal(first_shocks, second_shocks)
        assert np.array_equal(first, second), (
            f"{label}: the observation moved with the shock stream in a world that "
            "shows the agent nothing about the prices"
        )


# ---------------------------------------------------------------------------
# The three-world regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world", sorted(POINTS))
def test_the_regression_points_are_what_they_claim_to_be(world):
    """Cheap, and it runs in ``make test``: the fixtures have not moved worlds."""
    experiment, document = _load(world)
    assert experiment.lambda_risk == 10.0**-3.5
    assert experiment.estimator.regime == "antithetic"
    assert experiment.ppo.torch_threads == 8
    assert document["config"]["lambda_risk"] == experiment.lambda_risk
    assert document["provenance"]["git_dirty"] is False
    if world == SIGNAL_WORLD:
        assert experiment.liquidity.stochastic
    else:
        assert experiment.cost_encoding == world
        assert not experiment.liquidity.stochastic


@pytest.mark.parametrize("world", sorted(POINTS))
def test_no_committed_point_acquired_a_signal_by_omission(world):
    """Constitution §4 on the *third* seam, and the stake is the highest yet.

    All three configs predate ``world.signal`` entirely, so they must resolve to
    no signal — and every env their training path builds must be two or three
    coordinates wide exactly as it was. A default moving here is a one-line edit
    that would re-run three milestones in a world where the agent can see one step
    of the future, with every result still regenerating perfectly from its own
    config.
    """
    experiment, _ = _load(world)
    assert not experiment.signal.informative
    assert experiment.signal.correlation() == 0.0
    assert not training_signal(experiment).informative
    assert not evaluation_signal(experiment).informative

    env = ExecutionEnv(
        experiment.case.market,
        experiment.case.order_size,
        experiment.lambda_risk,
        liquidity=training_liquidity(experiment),
        signal=training_signal(experiment),
        root_seed=experiment.seeds.root_seed,
        pool=experiment.seeds.train_pool,
    )
    expected = 3 if experiment.liquidity.stochastic else 2
    assert env.observation_space.shape == (expected,)
    observation, _ = env.reset(seed=0)
    assert observation.shape == (expected,)
    assert np.all(env.signals == 0.0)


@pytest.mark.training
@pytest.mark.parametrize("world", sorted(POINTS))
def test_one_committed_seed_per_world_retrains_bitwise(world):
    """The seam's acceptance. Bitwise, not ``allclose``, in **three** worlds.

    ``allclose`` would pass on a seam that changed the order of a float addition —
    exactly the failure worth catching, because PPO compounds it over ~750 updates
    and M2 measured the same seed address landing at 0.165 and 0.066 of the TWAP
    gap under nothing worse than a different thread count. Equality is the only bar
    that means "the arithmetic in the worlds M5 did not touch is unchanged".

    Run **before** the seam is wired into anything that trains, which is the only
    time the answer is cheap: an hour now against a milestone's worth of numbers
    produced in a market nobody described.
    """
    experiment, document = _load(world)
    committed = document["seeds"][0]["grade"]

    _, policy = train_seed(experiment, 0)
    assert isinstance(policy, PPOPolicy)

    if world == SIGNAL_WORLD:
        # M4b's world grades by conditional expectation on held-out liquidity
        # paths, so the regression goes through *that* route rather than the
        # analytic one — the production path, not a re-implementation of it.
        regraded, _ = grade_liquidity(
            experiment, policy, liquidity_reference(experiment), name="seed0"
        )
        observed = np.asarray(regraded.mean_trajectory, dtype=float)
        expected = np.asarray(committed["mean_trajectory"], dtype=float)
    else:
        regraded = grade_policy(
            policy,
            experiment.case.market,
            experiment.case.order_size,
            experiment.reference(),
            root_seed=experiment.seeds.root_seed,
            pool=experiment.seeds.eval_pool,
            streams=experiment.seeds.eval_streams,
            name="seed0",
        )
        observed = np.asarray(regraded.trajectory, dtype=float)
        expected = np.asarray(committed["trajectory"], dtype=float)

    assert regraded.objective == committed["objective_bps"], (
        f"{world}: seed 0's objective moved by "
        f"{regraded.objective - committed['objective_bps']:.3e} bps through the "
        "signal seam. The seam is reaching into the price generator, so the "
        "correlation M5 is about to measure would be partly manufactured by two "
        "noise sources sharing a stream rather than by the model"
    )
    assert np.array_equal(observed, expected), (
        f"{world}: seed 0's trajectory moved through the signal seam"
    )
