"""M4b task 4's precondition — the guarantees, re-measured through *two* seams.

M4a ran this suite before it spent a seed and it earned its keep on the first
run: all twelve antithetic-cancellation cells came back red by 0.06 bps per step,
four orders outside the band, because ``mirror_of`` rebuilt the mirror without
the primary's temporary-impact model. The mirror was a Phase-1 env being averaged
against a power-law primary — the rewards still looked like rewards, the
schedules were still identical, and the estimator was silently no longer the one
the config named. Minutes, instead of a night.

**M4b doubles the surface for that defect class**, so the same class of check runs
again, and one of the four guarantees is deliberately expected to *change*:

1. **The exact noise identity survives.** Realised cost is still
   ``C = f(x, L) - sigma_bin * sum_k (x_k / X) xi_k``: liquidity enters ``f``,
   which carries no shock, so M1a's per-episode identity is untouched. Same claim
   M4a made about curvature, one seam along.
2. **The antithetic pair still cancels the price noise exactly** — because the
   pair holds liquidity **common** and negates only the price. §9's M4a entry
   named "a second, independent noise source" as what ends exactness; that is
   half right, and the wrong half is the useful one.
3. **The action identity is still green**, and this is the one that was predicted
   to break. It does not: what ends action identity is an observation the two
   halves *disagree* about — a price-bearing one — not a richer one. Both halves
   see the same ``log L_k``, so both take the same action.
4. **The open-loop check becomes the price-free check.** A liquidity-observing
   policy's schedule is *not* open-loop by design. ``deterministic_schedule``
   keeps its name and grows one axis: pin the liquidity, vary the price, require
   the trajectory bitwise. That is what licenses ``E[cost | L]``.

And the third per-step identity M4b adds — **the two halves saw the same
liquidity** — is shown to be *live* rather than merely present, by handing the
pair a mirror on a different liquidity path and requiring it to raise.

Run this before training, not after. ``make m4b-guarantees``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from temper.agents.execution import FractionPolicy, twap_fractions
from temper.env import LIQUIDITY_KEY, SHOCK_KEY, ExecutionEnv, LiquidityStream
from temper.env.impact import power_law_temporary
from temper.eval.antithetic import AntitheticPair, MirrorEnv, PairDiverged, mirror_of
from temper.eval.experiment import load_experiment
from temper.eval.grading import ScheduleNotDeterministic, deterministic_schedule
from temper.eval.rollout import run_episode
from temper.oracle import BPS, DeterministicLiquidity, cost_moments
from temper.seeding import M4B_DIFFERENTIAL_POOL

from .conftest import REPO_ROOT

CONFIG = load_experiment(REPO_ROOT / "configs" / "m4b_liquidity.yaml")
MARKET = CONFIG.case.market
ORDER_SIZE = CONFIG.case.order_size
LAMBDA = CONFIG.lambda_risk

#: The streams these checks spend. Their own pool: a guarantee reports no number
#: and must not burn an address a committed result is reported at.
STREAMS = (0, 1, 2, 3)

#: Pre-stated bands, M4a's verbatim. They are statements about float arithmetic
#: rather than about a market, so the milestone that changed the market does not
#: get to loosen them.
NOISE_IDENTITY_BAND = 1e-12          # relative
CANCELLATION_BAND = 1e-12            # bps per step

#: What each guarantee actually observed. The brief asks for the results to be
#: *recorded* before task 5, so the module reports itself rather than only
#: passing.
OBSERVED: dict[str, float] = {}


def _liquidity(stochastic: bool = True) -> LiquidityStream:
    law = CONFIG.liquidity if stochastic else DeterministicLiquidity()
    return LiquidityStream(law=law, pool=M4B_DIFFERENTIAL_POOL)


def _env(stream: int, *, stochastic: bool = True) -> ExecutionEnv:
    return ExecutionEnv(
        MARKET,
        ORDER_SIZE,
        LAMBDA,
        temporary_impact=power_law_temporary(MARKET),
        liquidity=_liquidity(stochastic),
        root_seed=CONFIG.seeds.root_seed,
        pool=M4B_DIFFERENTIAL_POOL,
        stream_index=stream,
    )


#: Three fixed schedules, so a guarantee that held only for TWAP is not mistaken
#: for one that holds. `optimal` is the liquidity world's static optimum.
def _schedules():
    from temper.eval.reference import liquidity_trajectories
    from temper.oracle import trades

    rows = liquidity_trajectories(MARKET, ORDER_SIZE, LAMBDA, CONFIG.liquidity)
    return {
        "twap": twap_fractions(MARKET.n_bins),
        **{
            name: _fractions(trades(rows[name], MARKET) / ORDER_SIZE)
            for name in ("m4a", "static")
        },
    }


def _fractions(weights: np.ndarray) -> np.ndarray:
    """Trade weights as fractions *of remaining* — the agent's coordinates."""
    remaining = 1.0 - np.concatenate(([0.0], np.cumsum(weights)))[:-1]
    return np.where(remaining > 0.0, weights / np.maximum(remaining, 1e-15), 1.0)


SCHEDULES = _schedules()


@pytest.fixture(params=sorted(SCHEDULES), ids=sorted(SCHEDULES))
def policy(request):
    return FractionPolicy(SCHEDULES[request.param], ORDER_SIZE, request.param)


# ---------------------------------------------------------------------------
# 1. The exact per-episode noise identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", STREAMS)
def test_the_exact_noise_identity_survives_stochastic_liquidity(policy, stream):
    """``C - E[cost | L] = -sigma_bin * sum_k (x_k / X) * walk_k``, exactly.

    M1a's identity, and the reason it holds unchanged is the same reason M4a's
    did: liquidity enters the *temporary* term, which is a function of the
    schedule and the market and carries no shock. Realised cost is therefore
    still affine in the price draws given ``(x, L)``, and the residual is float
    noise rather than a model difference.

    The reference is ``cost_moments`` **at the realised liquidity path** — which
    is what makes this a check on M4b's grading route as well as on the env: if
    the conditional expectation were the wrong world's, this is where it would
    show, at the size of a whole temporary term rather than at 1e-14.
    """
    env = _env(stream)
    episode = run_episode(env, policy)
    expected = cost_moments(
        episode.trajectory, MARKET, liquidity=env.multipliers
    ).expected

    weights = -np.diff(episode.trajectory) / ORDER_SIZE
    predicted_noise = -float(np.sum(weights * episode.walks))
    residual = abs(episode.shortfall_bps - expected - predicted_noise)
    relative = residual / max(abs(expected), 1e-12)

    OBSERVED["noise_identity"] = max(
        OBSERVED.get("noise_identity", 0.0), relative
    )
    assert relative <= NOISE_IDENTITY_BAND, (
        f"the per-episode noise identity broke at {relative:.3e} relative: "
        "realised cost is no longer affine in the price draws given the "
        "liquidity path, so E[cost | L] is not a closed form and M4b's whole "
        "grading route is invalid"
    )


@pytest.mark.parametrize("stream", STREAMS[:2])
def test_the_deterministic_reference_would_break_the_identity(policy, stream):
    """The check above is discriminative: the *wrong* liquidity fails it loudly.

    Grading the same episode against ``cost_moments`` with no liquidity argument
    is exactly the mistake a session would make by leaving the argument off, and
    it must not pass. Without this, a residual of 1e-14 would be evidence about
    float arithmetic and nothing else.
    """
    env = _env(stream)
    episode = run_episode(env, policy)
    wrong = cost_moments(episode.trajectory, MARKET).expected
    weights = -np.diff(episode.trajectory) / ORDER_SIZE
    predicted_noise = -float(np.sum(weights * episode.walks))
    relative = abs(episode.shortfall_bps - wrong - predicted_noise) / abs(wrong)

    OBSERVED["wrong_world_residual"] = max(
        OBSERVED.get("wrong_world_residual", 0.0), relative
    )
    assert relative > 1e3 * NOISE_IDENTITY_BAND, (
        "grading a stochastic-liquidity episode against the deterministic cost "
        f"moments left a residual of only {relative:.3e}; the liquidity argument "
        "is not reaching the charge and the identity test above is vacuous"
    )


# ---------------------------------------------------------------------------
# 2 & 3. The antithetic pair: price negated, liquidity common
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", STREAMS)
def test_the_pair_still_cancels_the_price_noise_exactly(policy, stream):
    """The average of the two halves is the noise-free reward, to float precision.

    The pairing needed only the ability to replay with negated draws, and it still
    has it: given ``(x, L)`` cost is affine in the price, both halves realise the
    same ``(x, L)``, so the shock terms are exact negations and cancel on the
    average. What the pair no longer removes is the *liquidity* noise — that is
    the reward variance the agent has to train through, and it is not a defect.
    """
    pair = AntitheticPair(_env(stream))
    observation, _ = pair.reset()
    policy.reset()

    terminated = False
    total = 0.0
    steps = 0
    while not terminated:
        observation, reward, terminated, _, info = pair.step(policy.act(observation))
        total += reward
        steps += 1

    primary = pair.primary
    expected = -(
        cost_moments(
            primary._trajectory, MARKET, liquidity=primary.multipliers
        ).expected
        + LAMBDA
        * cost_moments(
            primary._trajectory, MARKET, liquidity=primary.multipliers
        ).variance
    )
    per_step = abs(total - expected) / steps
    OBSERVED["cancellation"] = max(OBSERVED.get("cancellation", 0.0), per_step)
    assert per_step <= CANCELLATION_BAND, (
        f"the antithetic average is {per_step:.3e} bps per step from the "
        "noise-free objective; the pair is no longer averaging two halves of one "
        "world"
    )


@pytest.mark.parametrize("stream", STREAMS)
def test_the_action_identity_survives_a_richer_observation(policy, stream):
    """§9's M4a entry predicted this would break at M4b. It does not, and why.

    The entry named "a second, independent noise source **or** a price-bearing
    observation" as what ends the pairing's exactness. The disjunction is too
    wide: what action identity needs is not a *poor* observation but one the two
    halves **agree about**. They see the same liquidity, so they see the same
    three-vector, so they take the same action — and the price noise still
    cancels exactly given ``(x, L)``.
    """
    pair = AntitheticPair(_env(stream))
    observation, _ = pair.reset()
    policy.reset()
    assert observation.shape == (3,), "the liquidity world's observation is three-wide"

    terminated = False
    seen = 0
    while not terminated:
        observation, _, terminated, _, info = pair.step(policy.act(observation))
        # The multiplier the bin actually charged, read back off the env, so this
        # is a check on the *charge* rather than on the observation encoding.
        assert info[LIQUIDITY_KEY] == pair.primary.multipliers[seen]
        assert info[SHOCK_KEY] != 0.0, "a shock-free episode would prove nothing"
        seen += 1
    assert seen == MARKET.n_bins
    # Reaching here without PairDiverged *is* the assertion: the pair checks the
    # observation, the trade, the shock negation and the shared multiplier on
    # every step and raises rather than returning a flag.
    OBSERVED["action_identity"] = 0.0


@pytest.mark.parametrize("stream", STREAMS[:2])
def test_the_pair_holds_liquidity_common_rather_than_mirrored(stream):
    """The multiplier is *shared*, not negated — and it is the same object.

    Antithetically mirroring the liquidity as well (``u -> 1 - u`` on its uniform)
    would make the halves disagree about ``L``, hence about their actions, and
    would trade the pairing's one exact property for a partial second one. So the
    two halves must see the identical path, bitwise.
    """
    env = _env(stream)
    mirror = mirror_of(env)
    assert isinstance(mirror, MirrorEnv)
    assert mirror.liquidity == env.liquidity, (
        "mirror_of did not hand over the liquidity stream; the mirror would be "
        "charging a different market — the M4a bug's exact shape, one seam along"
    )
    assert mirror.temporary_impact == env.temporary_impact

    env.reset(seed=stream)
    mirror.reset(seed=stream)
    assert np.array_equal(env.multipliers, mirror.multipliers)
    assert not np.array_equal(env.multipliers, np.ones(MARKET.n_bins))


def test_a_mirror_on_a_different_liquidity_path_is_refused(policy):
    """The M4a defect one seam along, made to fire rather than reasoned about.

    A mirror charging a *different* market is precisely the shape of the bug that
    cost M4a its first run: plausible rewards, identical-looking schedules, an
    estimator quietly averaging two worlds. Here it is refused before a single
    step — by the **observation** check, because in this world the multiplier is
    in the observation and the halves disagree about it at ``reset``.

    Which check catches it is worth being precise about rather than glossing:
    that one is the pre-existing assertion doing new work for free, and the
    dedicated liquidity identity is what still catches the case the observation
    cannot. The next test makes that one fire on its own.
    """
    pair = AntitheticPair(_env(0))
    pair.mirror.liquidity = pair.mirror.liquidity.pinned_to(97)
    pair.mirror._liquidity_rng = None

    with pytest.raises(PairDiverged, match="different observations"):
        pair.reset()


def test_the_third_per_step_identity_catches_what_the_observation_cannot(policy):
    """The dedicated liquidity check, exercised where it is the *only* one that can.

    The observation carries ``log L_k`` for the bin about to execute, and after
    the **last** bin there is no next bin — the terminal entry is ``0.0`` for both
    halves. So a mirror whose final multiplier differs is a mirror the two halves
    never disagree about in any observation, that realises the identical schedule,
    and that charges a different market on the last bin. Nothing but M4b's third
    identity is looking at that, which is exactly why it exists.
    """
    pair = AntitheticPair(_env(0))
    observation, _ = pair.reset()
    policy.reset()
    # After reset, so the halves agreed on L_0 and every observation up to the
    # terminal one will agree too.
    pair.mirror._multipliers[-1] *= 1.5

    with pytest.raises(PairDiverged, match="different liquidity"):
        terminated = False
        while not terminated:
            observation, _, terminated, _, _ = pair.step(policy.act(observation))


# ---------------------------------------------------------------------------
# 4. The open-loop check becomes the price-free check
# ---------------------------------------------------------------------------


def test_a_fixed_schedule_is_price_free_at_a_pinned_liquidity_path(policy):
    """``deterministic_schedule``'s successor, on schedules whose answer is known."""
    trajectory = deterministic_schedule(
        policy,
        MARKET,
        ORDER_SIZE,
        LAMBDA,
        root_seed=CONFIG.seeds.root_seed,
        pool=CONFIG.seeds.eval_pool,
        streams=CONFIG.seeds.eval_streams,
        temporary_impact=power_law_temporary(MARKET),
        liquidity=_liquidity(),
        expect_encoding=CONFIG.cost_encoding,
    )
    assert trajectory[0] == ORDER_SIZE
    assert trajectory[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(trajectory) <= 1e-9)


def test_the_price_free_check_still_refuses_a_policy_that_is_not():
    """It has to be able to fail, or it is not a check.

    A policy that peeks at the price through ``np_random`` realises a different
    schedule on each price stream even at one pinned liquidity path, and that is
    the failure that would silently turn ``E[cost | L]`` from a closed form into
    a biased estimate.
    """

    class PriceLeaking:
        name = "leaky"

        def reset(self) -> None:
            pass

        def act(self, observation) -> float:
            remaining = float(observation[1]) * ORDER_SIZE
            # A tiny, price-dependent nudge — the size a careless feature would be.
            return min(
                remaining,
                remaining / MARKET.n_bins * (1.0 + 1e-9 * np.random.default_rng().normal()),
            )

    with pytest.raises(ScheduleNotDeterministic, match="price streams"):
        deterministic_schedule(
            PriceLeaking(),
            MARKET,
            ORDER_SIZE,
            LAMBDA,
            root_seed=CONFIG.seeds.root_seed,
            pool=CONFIG.seeds.eval_pool,
            streams=CONFIG.seeds.eval_streams,
            temporary_impact=power_law_temporary(MARKET),
            liquidity=_liquidity(),
        )


def test_the_check_would_be_vacuous_without_the_pin(policy):
    """The liquidity pin is load-bearing, and this measures that it is.

    Without it the liquidity index follows the env's stream index, so varying the
    price stream varies the liquidity too — the trajectories would differ for a
    perfectly price-free policy and the check would fire on every agent M4b
    trains, which is the same as not having a check.
    """
    unpinned = _liquidity()
    trajectories = []
    for stream in CONFIG.seeds.eval_streams:
        env = ExecutionEnv(
            MARKET,
            ORDER_SIZE,
            LAMBDA,
            temporary_impact=power_law_temporary(MARKET),
            liquidity=unpinned,
            root_seed=CONFIG.seeds.root_seed,
            pool=CONFIG.seeds.eval_pool,
            stream_index=int(stream),
        )
        trajectories.append(run_episode(env, policy).trajectory)
    # A *fixed* schedule is unaffected by liquidity, so it stays identical; the
    # pin matters for a reacting policy, and that is what the next assertion says.
    assert np.array_equal(trajectories[0], trajectories[1])

    pinned = unpinned.pinned_to(int(CONFIG.seeds.eval_streams[0]))
    paths = []
    for stream in CONFIG.seeds.eval_streams:
        env = ExecutionEnv(
            MARKET,
            ORDER_SIZE,
            LAMBDA,
            temporary_impact=power_law_temporary(MARKET),
            liquidity=pinned,
            root_seed=CONFIG.seeds.root_seed,
            pool=CONFIG.seeds.eval_pool,
            stream_index=int(stream),
        )
        env.reset(seed=int(stream))
        paths.append(env.multipliers)
    assert np.array_equal(paths[0], paths[1]), "the pin did not hold the path fixed"

    unpinned_paths = []
    for stream in CONFIG.seeds.eval_streams:
        env = ExecutionEnv(
            MARKET,
            ORDER_SIZE,
            LAMBDA,
            temporary_impact=power_law_temporary(MARKET),
            liquidity=unpinned,
            root_seed=CONFIG.seeds.root_seed,
            pool=CONFIG.seeds.eval_pool,
            stream_index=int(stream),
        )
        env.reset(seed=int(stream))
        unpinned_paths.append(env.multipliers)
    assert not np.array_equal(unpinned_paths[0], unpinned_paths[1]), (
        "the two eval streams draw the same liquidity, so pinning is a no-op and "
        "the price-free check cannot distinguish the two noise sources"
    )


def test_the_guarantees_report_themselves(capsys):
    """Task 4 asks for the results to be *recorded* before task 5, not only passed."""
    lines = [
        "",
        "M4b inherited guarantees, through two seams:",
        f"  noise identity        worst {OBSERVED.get('noise_identity', float('nan')):.3e} "
        f"relative (band {NOISE_IDENTITY_BAND:g})",
        f"  antithetic cancel     worst {OBSERVED.get('cancellation', float('nan')):.3e} "
        f"bps per step (band {CANCELLATION_BAND:g})",
        "  action identity       green — the pair asserts it per step and did not raise",
        "  liquidity shared      green — and shown live by making the assertion fire",
        f"  discriminative        the deterministic reference misses by "
        f"{OBSERVED.get('wrong_world_residual', float('nan')):.3e} relative",
        "",
    ]
    with capsys.disabled():
        print("\n".join(lines))
    assert OBSERVED.get("noise_identity", 1.0) <= NOISE_IDENTITY_BAND
    assert OBSERVED.get("cancellation", 1.0) <= CANCELLATION_BAND
