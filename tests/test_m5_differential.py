"""M5 task 5 — the differential, through three seams (invariant 6).

Constitution invariant 6: *no environment feature without an independent
expectation test*. M5 adds a signal that is **correlated with the price shocks on
purpose**, which is a different kind of feature from the two before it and needs a
differential that can tell "correlated by the model" from "correlated because two
noise sources share a generator".

**The process.** The env's own signal draws against the closed forms — mean zero,
unit variance, no serial correlation — and then the part that is M5's rather than
M4b's: the **residual**. ``corr(s_k, xi_{k+1})`` is ``rho`` by design, so a check
that the signal and the price are independent would be false. What must hold is
that after removing the modelled correlation nothing is left. Recover
``e_k = (xi_k - rho s_{k-1}) / sqrt(1 - rho^2)`` from the env's own published walk
and require it unit-variance, serially uncorrelated, and uncorrelated with *every*
``s_j``. If the signal came out of the price generator the residual carries the
trace; nothing else here would notice.

**The world, conditionally.** ``z = (C - E[cost | s]) / sqrt((1 - rho^2) V)``,
standardised per episode against its own realised signal path. Exactly ``N(0, 1)``
under the null, and the ``(1 - rho^2)`` is a finding rather than a fudge:
conditioning on the signal makes the predictable part of each shock *known*, so
what is left is ``sqrt(1 - rho^2) e`` per bin. The graded ``V`` is unchanged — it
is the unconditional price-shortfall variance, the shock still has unit variance
by construction, and invariant 7 needs no amendment.

**The world, unconditionally.** The sample mean cost against the closed form with
**no** alpha term, because ``E[alpha] = -A rho sum_k h_k E[s] = 0`` exactly for a
fixed schedule. This is the half that catches a mixture that failed to
renormalise — the shock's variance moving away from one — which the conditional
check cannot see: it standardises the mistake away.

**The pairing identity, at M1's tiers.** ``(xi + xi') / 2 = rho s_{k-1}``, per
step, with the step count asserted. Everything M5 reports rests on the pair's
average being the conditional mean — it is what makes the training reward and the
grading formula the same object — and as of task 4 that claim had exactly one
measurement behind it. It has the tiers behind it now.

**What this differential cannot see, said out loud.** At the milestone's
``rho = 0.01`` an off-by-one in the alpha term's index is invisible here: it leaves
the unconditional mean untouched and moves ``var(z)`` from ``1 - rho^2`` to
``1 + rho^2``, which is 1.0001 against a band of 0.057. The index is checked on
paper in ``tests/test_m5_conditional_grading.py``. What runs here is the same
check at a ``rho`` where the arithmetic is unmistakable, so that "the differential
is green" is not read as more than it is.

If a cell misses its band, the milestone's product is that finding. Do not tune
the env toward the bands.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from temper.agents import baseline
from temper.env import (
    DETERMINISTIC_LIQUIDITY,
    EPISODE_KEY,
    SHOCK_KEY,
    ExecutionEnv,
    impact_for,
)
from temper.eval import sample_costs
from temper.eval.antithetic import AntitheticPair
from temper.eval.conditional import (
    LIQUIDITY_SEAM,
    SIGNAL_SEAM,
    ConditioningMismatch,
    check_conditioning_matches_observation,
    observed_seams,
    signal_rollouts,
)
from temper.oracle import (
    LognormalLiquidity,
    OneStepSignal,
    alpha_coefficient,
    conditional_alpha_bps,
    conditional_shortfall_variance_bps2,
    cost_moments,
    liquidity_for,
)

from .conftest import (
    M5_CONFIG,
    M5_ENCODING,
    DifferentialPair,
    build_signal_env,
    case_by_id,
    m5_signal_law,
    signal_pairs,
)

LAW = m5_signal_law()
RHO = LAW.correlation()
PROCESS = M5_CONFIG["process"]
PAIRING = M5_CONFIG["pairing"]
CONDITIONAL = M5_CONFIG["conditional"]
DISCRIMINATION = M5_CONFIG["discrimination"]

#: M5's four schedules map straight onto the power-law world's baselines: with a
#: zero-mean signal the best *fixed* schedule is the one that knows nothing about
#: it, so ``optimal`` is M4a's certified optimum and there is no fifth rung.
BASELINE_NAMES = {
    "twap": "twap",
    "ac": "ac",
    "tangent": "tangent",
    "optimal": "optimal",
}


def _schedule_policy(name: str, case):
    return baseline(
        BASELINE_NAMES[name],
        case.market,
        case.order_size,
        case.lambda_risk,
        encoding=M5_ENCODING,
    )


def _standardised_shocks(env, schedule, episodes: int, seed_base: int = 0):
    """Run `episodes` fixed-schedule episodes; return the signals and the shocks.

    The shocks come off the env's *published* cumulative walk, differenced and
    divided by ``sigma_bin * BPS`` — the one public route to the price path, and
    deliberately the only one.
    """
    amplitude = alpha_coefficient(env.market)
    signals = np.empty((episodes, env.market.n_bins))
    shocks = np.empty((episodes, env.market.n_bins))
    for index in range(episodes):
        env.reset(seed=seed_base + index)
        signals[index] = env.signals
        walk = []
        for shares in schedule:
            _, _, _, _, info = env.step(float(shares))
            walk.append(info[SHOCK_KEY])
        shocks[index] = np.diff(np.concatenate(([0.0], walk))) / amplitude
    return signals, shocks


# ---------------------------------------------------------------------------
# The process
# ---------------------------------------------------------------------------


def test_the_envs_own_signal_draws_match_the_oracles_closed_forms():
    """Invariant 6, applied to a distribution rather than to a market.

    The env draws; the oracle states the moments; neither computes the other's
    number. ``E[s] = 0`` is the one the whole lambda argument rests on — a signal
    with a non-zero mean would move every *fixed* schedule's objective and M5's
    "the static reading is bit-identical to M4a's" would be false — and unit
    variance is what makes ``rho`` a correlation rather than a scale.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    env = build_signal_env(case, int(M5_CONFIG["seeding"]["process_stream"]))
    episodes = int(PROCESS["draws"]) // case.market.n_bins

    sample = sample_costs(
        env, _schedule_policy("twap", case), episodes, record_signals=True
    )
    assert sample.signals is not None
    assert sample.signals.shape == (episodes, case.market.n_bins)
    flat = sample.signals.reshape(-1)

    band = float(PROCESS["mean_sigmas"]) * float(flat.std(ddof=1)) / math.sqrt(flat.size)
    assert abs(float(flat.mean()) - LAW.mean()) <= band, (
        f"the env's signal has mean {float(flat.mean()):.6f}, not 0, outside "
        f"{band:.6f}; a signal with a mean moves every fixed schedule's objective "
        "and M5's whole claim that lambda needs no new reading goes with it"
    )
    assert float(flat.var(ddof=1)) == pytest.approx(
        LAW.variance(), rel=float(PROCESS["variance_rtol"])
    )


def test_the_signal_draws_are_independent_across_bins():
    """i.i.d. is what makes ``(k, x_k, s_k)`` a sufficient statistic.

    Lag-1 autocorrelation within an episode and across the boundary, against its
    own ``1/sqrt(n)`` null. A signal with memory would make the dynamic program
    something other than the optimum over all adapted policies, and M5's
    denominator quietly wrong, while every other check here stayed green.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    env = build_signal_env(case, int(M5_CONFIG["seeding"]["process_stream"]) + 1)
    episodes = int(PROCESS["draws"]) // case.market.n_bins
    sample = sample_costs(
        env, _schedule_policy("twap", case), episodes, record_signals=True
    )
    flat = sample.signals.reshape(-1)

    centred = flat - flat.mean()
    lag1 = float(np.sum(centred[:-1] * centred[1:]) / np.sum(centred**2))
    band = float(PROCESS["autocorrelation_sigmas"]) / math.sqrt(centred.size)
    assert abs(lag1) <= band, (
        f"the signal has lag-1 autocorrelation {lag1:.5f}, outside {band:.5f}: the "
        "process is not memoryless, so (k, x_k, s_k) is not sufficient and the "
        "dynamic program is not the optimum over adapted policies"
    )


def test_the_signal_and_the_price_are_independent_apart_from_the_model():
    """The per-draw independence claim, stated on the **residual**.

    ``corr(s_k, xi_{k+1}) = rho`` by design, so "the signal and the price are
    independent" is false as written and has to be sharpened into something that
    can fail. It is this: after removing the modelled correlation,

    .. code::

        e_k = (xi_k - rho s_{k-1}) / sqrt(1 - rho^2)

    must be unit-variance, serially uncorrelated, and uncorrelated with **every**
    ``s_j`` — the one it predicts and the ones it does not.

    That is the check a signal drawn out of the price generator fails and nothing
    else here would. The addresses are separate by construction and asserted as
    such in ``tests/test_m5_signal_seam.py``; this is the same claim at the level
    of the realised numbers, which is what invariant 6 asks for.

    Run at the committed ``rho`` — the residual is a residual at every ``rho``, and
    doing it in the world the milestone runs in is the point.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    env = build_signal_env(case, int(M5_CONFIG["seeding"]["process_stream"]) + 2)
    bins = case.market.n_bins
    episodes = int(PROCESS["draws"]) // bins // 5
    schedule = -np.diff(_schedule_policy("twap", case).trajectory)
    signals, shocks = _standardised_shocks(env, schedule, episodes)

    # The modelled correlation is there, first: a residual test on a signal that
    # predicts nothing would pass and mean nothing.
    forward = float(np.corrcoef(signals[:, :-1].reshape(-1), shocks[:, 1:].reshape(-1))[0, 1])
    band = float(PROCESS["correlation_sigmas"]) / math.sqrt(signals[:, :-1].size)
    assert abs(forward - RHO) <= band, (
        f"corr(s_k, xi_(k+1)) is {forward:.5f} against a modelled {RHO}, outside "
        f"{band:.5f}; the seam is not realising the law the oracle prices"
    )

    residual = np.empty_like(shocks)
    residual[:, 0] = shocks[:, 0]
    residual[:, 1:] = (shocks[:, 1:] - RHO * signals[:, :-1]) / math.sqrt(1.0 - RHO**2)

    assert float(residual.var(ddof=1)) == pytest.approx(
        1.0, rel=float(PROCESS["residual_variance_rtol"])
    ), "the residual is not unit-variance; the mixture did not renormalise"

    flat = residual.reshape(-1)
    centred = flat - flat.mean()
    lag1 = float(np.sum(centred[:-1] * centred[1:]) / np.sum(centred**2))
    serial_band = float(PROCESS["residual_sigmas"]) / math.sqrt(centred.size)
    assert abs(lag1) <= serial_band, f"the residual has memory: lag-1 {lag1:.5f}"

    # Every (k, j) pair, not only the ones the model names. A leak from the price
    # generator into the signal would show up somewhere in this grid and nowhere
    # else in this file.
    pair_band = float(PROCESS["residual_sigmas"]) / math.sqrt(episodes)
    worst, worst_at = 0.0, None
    for k in range(bins):
        for j in range(bins):
            observed = abs(float(np.corrcoef(signals[:, k], residual[:, j])[0, 1]))
            if observed > worst:
                worst, worst_at = observed, (k, j)
    assert worst <= pair_band, (
        f"corr(s_{worst_at[0]}, e_{worst_at[1]}) = {worst:.5f} outside {pair_band:.5f}: "
        "the signal and the price shocks share information the model does not "
        "describe, which is what a signal drawn from the price generator looks like"
    )


def test_the_draws_reproduce_from_the_seed_address():
    """Invariant 1, on the third noise source: same address, same signal path."""
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    stream = int(M5_CONFIG["seeding"]["process_stream"]) + 3
    first, second = (build_signal_env(case, stream) for _ in range(2))
    first.reset()
    second.reset()
    assert np.array_equal(first.signals, second.signals)
    assert not np.array_equal(first.signals, np.zeros(case.market.n_bins))

    other = build_signal_env(case, stream + 1)
    other.reset()
    assert not np.array_equal(first.signals, other.signals), (
        "two stream indices drew the same signal path; the third noise source is "
        "not addressed"
    )
    other.reset(seed=stream)
    assert np.array_equal(other.signals, first.signals)


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------


def _run_cell(pair: DifferentialPair, *, rho: float | None = None) -> dict:
    """One (case, schedule) cell: both checks, one pass through the loop."""
    case = pair.case
    env = build_signal_env(case, pair.stream_index, rho=rho)
    law = m5_signal_law(rho)
    policy = _schedule_policy(pair.schedule, case)

    before = env.step_count
    started = time.perf_counter()
    sample = sample_costs(
        env, policy, pair.n_sim, require_fixed_schedule=True, record_signals=True
    )
    seconds = time.perf_counter() - started
    steps = env.step_count - before

    trajectory = sample.trajectory
    moments = cost_moments(trajectory, case.market)
    # The conditional variance is V less the part the signal predicts, and the
    # first `lag` bins keep theirs whole because nothing predicts them. At rho = 0
    # this returns V bitwise, so the standardisation below is M1's at a special
    # value rather than a second one that agrees with it.
    conditional_variance = conditional_shortfall_variance_bps2(
        trajectory, case.market, law
    )
    conditional = np.array(
        [
            moments.expected
            + conditional_alpha_bps(trajectory, case.market, path, law)
            for path in sample.signals
        ]
    )
    z = (sample.costs - conditional) / math.sqrt(conditional_variance)
    return {
        "steps": steps,
        "seconds": seconds,
        "mean_z": float(z.mean()),
        "var_z": float(z.var(ddof=1)),
        "mean_cost": float(sample.costs.mean()),
        "reference": moments.expected,
        "cost_se": float(sample.costs.std(ddof=1)) / math.sqrt(pair.n_sim),
    }


def _assert_cell(pair: DifferentialPair, observed: dict, spec: dict) -> None:
    assert observed["steps"] == pair.n_sim * pair.case.market.n_bins, (
        f"{pair}: the loop was called {observed['steps']} times for "
        f"{pair.n_sim} episodes of {pair.case.market.n_bins} bins; something took "
        "a route around ExecutionEnv.step"
    )
    assert abs(observed["mean_z"]) <= pair.mean_band, (
        f"{pair}: mean(z) = {observed['mean_z']:+.5f} outside ±{pair.mean_band:.5f}. "
        "The conditional band is EXACT — given s the shortfall is still a fixed "
        "linear combination of independent Gaussians — so this is the env and the "
        "conditional cost function disagreeing, not sampling."
    )
    assert abs(observed["var_z"] - 1.0) <= pair.var_band, (
        f"{pair}: var(z) = {observed['var_z']:.5f} outside 1 ± {pair.var_band:.5f}"
    )
    band = float(spec["mean_cost_sigmas"]) * observed["cost_se"]
    assert abs(observed["mean_cost"] - observed["reference"]) <= band, (
        f"{pair}: unconditional mean cost {observed['mean_cost']:.6f} bps against "
        f"the closed form {observed['reference']:.6f}, outside ±{band:.6f}. The "
        "alpha term is absent from that reference on purpose — E[s] = 0 — so this "
        "is the check that catches a mixture whose shock no longer has unit "
        "variance, which the conditional one standardises away."
    )


@pytest.mark.parametrize("pair", signal_pairs("fast"), ids=str)
def test_fast_tier(pair):
    """In ``make test``: twelve cells at 10 000 episodes each."""
    _assert_cell(pair, _run_cell(pair), M5_CONFIG["tiers"]["fast"])


@pytest.mark.deep
@pytest.mark.parametrize("pair", signal_pairs("deep"), ids=str)
def test_deep_tier(pair):
    """``make m5-differential``: the full 3 x 3 golden grid at M1's N_sim."""
    _assert_cell(pair, _run_cell(pair), M5_CONFIG["tiers"]["deep"])


def test_the_conditional_variance_is_where_the_signal_shows_and_the_graded_one_is_not():
    """``(1 - rho^2)`` measured, at a rho where it is unmistakable.

    Two things at once, and they are the reason this cell exists. The conditional
    variance really is ``(1 - rho^2) V`` — standardising by ``V`` instead would put
    ``var(z)`` at 0.64 rather than 1 — and the **graded** ``V`` is untouched, which
    is why invariant 7 needs no amendment: the frozen objective penalises
    unconditional price-shortfall variance and the shock still has unit variance by
    construction.

    It also happens to be the cell in which an off-by-one in the alpha term's index
    would be visible: a shifted index inflates ``var(z)`` from ``1 - rho^2`` to
    ``1 + rho^2``, 0.64 against 1.36 here and 0.9999 against 1.0001 at the
    milestone's rho. So this is what stops "the differential is green at rho =
    0.01" from being read as evidence about the index.
    """
    rho = float(DISCRIMINATION["rho"])
    case = case_by_id(DISCRIMINATION["case"])
    n_sim = int(DISCRIMINATION["n_sim"])
    pair = DifferentialPair(
        tier="discrimination",
        case=case,
        schedule=str(DISCRIMINATION["schedule"]),
        n_sim=n_sim,
        stream_index=int(DISCRIMINATION["stream_base"]),
        mean_band=float(DISCRIMINATION["mean_z_sigmas"]) / math.sqrt(n_sim),
        var_band=float(DISCRIMINATION["var_z_sigmas"]) * math.sqrt(2.0 / n_sim),
    )
    observed = _run_cell(pair, rho=rho)
    assert observed["steps"] == n_sim * case.market.n_bins
    assert abs(observed["mean_z"]) <= pair.mean_band
    assert abs(observed["var_z"] - 1.0) <= pair.var_band, (
        f"var(z) = {observed['var_z']:.5f} at rho = {rho}; the conditional "
        "variance is not (1 - rho^2) V"
    )

    # The three standardisations this cell can now tell apart, spelled out so the
    # claim is a measurement rather than an inference from a passing test. Each is
    # the ratio var(z) would take under a wrong denominator.
    law = m5_signal_law(rho)
    trajectory = _schedule_policy(pair.schedule, case).trajectory
    graded = cost_moments(trajectory, case.market).variance
    conditional = conditional_shortfall_variance_bps2(trajectory, case.market, law)
    holdings = trajectory[:-1] / trajectory[0]
    flat = (1.0 - rho**2) * graded

    under_graded = conditional / graded
    under_flat = conditional / flat
    for label, ratio in (("the graded V", under_graded), ("a flat (1-rho^2) V", under_flat)):
        # Twice the band, which is eight sampling sigmas: enough that a wrong
        # denominator is a failure rather than a bad day.
        assert abs(ratio - 1.0) > 2.0 * pair.var_band, (
            f"standardising by {label} would have been indistinguishable from the "
            f"conditional variance (ratio {ratio:.5f}, band {pair.var_band:.5f}); "
            "this cell is not discriminative"
        )
    # And the flat form is wrong by exactly the first bin's share, which is the
    # off-by-one this cell caught: h_0 = 1 keeps its whole variance.
    assert conditional - flat == pytest.approx(
        rho**2 * alpha_coefficient(case.market) ** 2 * float(holdings[0] ** 2),
        rel=1e-9,
    )


# ---------------------------------------------------------------------------
# The pairing identity, at the tiers
# ---------------------------------------------------------------------------


def _pairing_residual(case, episodes: int, *, rho: float | None, stream: int):
    """Worst ``|(walk + walk') / 2 - E[walk | s]|`` over `episodes`, and the steps."""
    env = build_signal_env(case, stream, rho=rho)
    pair = AntitheticPair(env)
    schedule = -np.diff(_schedule_policy("twap", case).trajectory)
    amplitude = alpha_coefficient(case.market) * m5_signal_law(rho).correlation()

    before = env.step_count
    worst = 0.0
    exact = True
    for index in range(episodes):
        pair.reset(seed=index)
        signals = env.signals
        for step, shares in enumerate(schedule):
            _, _, _, _, info = pair.step(float(shares))
            expected = amplitude * float(np.sum(signals[:step]))
            middle = 0.5 * (info[SHOCK_KEY] + pair.mirror._walk)
            worst = max(worst, abs(middle - expected))
            exact = exact and (pair.mirror._walk == -info[SHOCK_KEY])
    return worst, env.step_count - before, exact


@pytest.mark.parametrize("tier", ["fast"])
def test_the_pairing_identity_holds_at_the_tiers(tier):
    """``(xi + xi') / 2 = rho s_(k-1)``, per step, with the step count asserted.

    Everything M5 reports rests on this: it is what makes the pair's averaged
    reward the conditional expectation, and therefore what makes the training
    reward and the grading formula the same object. As of task 4 it had one
    measurement behind it at one rho on one schedule, which is not enough for a
    claim carrying a milestone.

    The bar is arithmetic rather than statistical. The two sides sum the same
    terms in different orders and nothing else separates them, so a band would be
    the wrong shape of instrument entirely.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    episodes = int(PAIRING[f"{tier}_episodes"])
    worst, steps, _ = _pairing_residual(
        case, episodes, rho=None, stream=int(M5_CONFIG["seeding"]["process_stream"]) + 10
    )
    assert steps == episodes * case.market.n_bins, (
        f"the pair stepped {steps} times for {episodes} episodes of "
        f"{case.market.n_bins} bins; the identity was not checked on every step"
    )
    assert worst <= float(PAIRING["walk_tolerance_bps"]), (
        f"the pair's shocks average to something other than E[xi | s]: worst "
        f"{worst:.3e} bps over {steps} steps"
    )


@pytest.mark.deep
def test_the_pairing_identity_holds_at_the_deep_tier():
    """``make m5-differential``: the same identity at M1's episode count."""
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    episodes = int(PAIRING["deep_episodes"])
    worst, steps, _ = _pairing_residual(
        case, episodes, rho=None, stream=int(M5_CONFIG["seeding"]["process_stream"]) + 11
    )
    assert steps == episodes * case.market.n_bins
    assert worst <= float(PAIRING["walk_tolerance_bps"])


def test_at_rho_zero_the_generalised_identity_is_m1s_own_assertion():
    """The same assertion at a special value, not a second one that agrees with it.

    ``(xi + xi') / 2 = rho s`` reduces to ``xi' = -xi`` when ``rho = 0``, and this
    shows it *reduces* rather than *approximates*: the exact negation holds
    bitwise, on every step, and the generalised residual is not merely small but
    identically zero. That is a stronger statement than "M1's check still passes
    and so does M5's", which is what asserting the two separately would have said.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    episodes = max(64, int(PAIRING["fast_episodes"]) // 100)
    worst, steps, exact = _pairing_residual(
        case, episodes, rho=0.0, stream=int(M5_CONFIG["seeding"]["process_stream"]) + 12
    )
    assert steps == episodes * case.market.n_bins
    assert exact, "at rho = 0 the mirror's walk is not the exact negation"
    assert worst == 0.0, (
        f"the generalised residual is {worst:.3e} rather than identically zero at "
        "rho = 0, so the two forms are not the same assertion"
    )

    # And the standardisation collapses the same way, bitwise: at rho = 0 the
    # conditional variance IS the graded V, not a float away from it, so
    # `_run_cell` at rho = 0 divides by exactly what M1's cell divides by.
    trajectory = _schedule_policy("twap", case).trajectory
    assert conditional_shortfall_variance_bps2(
        trajectory, case.market, m5_signal_law(0.0)
    ) == cost_moments(trajectory, case.market).variance


def test_at_rho_zero_the_conditional_cell_is_m1s_cell():
    """The world half of the same statement: bitwise, not within a band.

    An env with ``rho = 0`` must produce exactly the costs an env with no signal
    seam at all produces, and the conditional standardisation must reduce to the
    unconditional one — the alpha term is a float zero, and dividing by
    ``(1 - 0) V`` is dividing by ``V``. So M5's differential at ``rho = 0`` is
    M1's differential, arithmetic included.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    policy = _schedule_policy("twap", case)
    episodes = 400
    stream = int(M5_CONFIG["seeding"]["process_stream"]) + 13

    zero = build_signal_env(case, stream, rho=0.0)
    plain = ExecutionEnv(
        case.market,
        case.order_size,
        case.lambda_risk,
        temporary_impact=impact_for(M5_ENCODING, case.market, case.order_size),
        liquidity=DETERMINISTIC_LIQUIDITY,
        root_seed=int(M5_CONFIG["seeding"]["root_seed"]),
        pool=M5_CONFIG["seeding"]["pool"],
        stream_index=stream,
    )
    with_signal = sample_costs(zero, policy, episodes, record_signals=True)
    without = sample_costs(plain, policy, episodes)
    assert np.array_equal(with_signal.costs, without.costs), (
        "a rho = 0 signal seam moved the realised costs; the seam is not inert "
        "where it carries no information"
    )
    assert np.array_equal(with_signal.trajectory, without.trajectory)

    law = m5_signal_law(0.0)
    alphas = np.array(
        [
            conditional_alpha_bps(with_signal.trajectory, case.market, path, law)
            for path in with_signal.signals
        ]
    )
    assert np.all(alphas == 0.0), "the alpha term is not identically zero at rho = 0"


# ---------------------------------------------------------------------------
# E[cost | s] against sampled prices — the full tier
# ---------------------------------------------------------------------------


def test_the_conditional_expectation_is_the_mean_over_price_draws():
    """Task 4's cheap check at the tier, over several pinned signal paths.

    Each path pins the signal stream and lets the price stream move, so the
    sampled mean is estimating one conditional expectation rather than an average
    of many. The alpha term has to be several half-widths wide on every path or
    the check cannot see what it is verifying — that bar is in the config and is
    asserted, not hoped for, which is why ``rho`` is raised for this section and
    says so there.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    rho = float(CONDITIONAL["rho"])
    law = m5_signal_law(rho)
    policy = _schedule_policy("twap", case)
    episodes = int(CONDITIONAL["episodes_per_path"])
    stream = int(M5_CONFIG["seeding"]["process_stream"]) + 20

    for path_index in range(int(CONDITIONAL["pinned_paths"])):
        env = build_signal_env(case, stream + path_index, rho=rho)
        env.signal = env.signal.pinned_to(500 + path_index)
        env._signal_rng = None
        # Driven by hand rather than through sample_costs, and the reason is the
        # pin: reset() WITHOUT a seed draws the next block from the signal
        # generator, so the path would move episode by episode and the mean would
        # be an average of conditional expectations rather than one of them.
        # reset(seed=i) re-addresses both streams, and with the signal pinned that
        # re-creates the SAME signal block while the price stream moves — which is
        # exactly the experiment.
        schedule = -np.diff(policy.trajectory)
        costs = np.empty(episodes)
        drawn = np.empty((episodes, case.market.n_bins))
        for episode in range(episodes):
            env.reset(seed=episode)
            drawn[episode] = env.signals
            for shares in schedule:
                _, _, _, _, info = env.step(float(shares))
            costs[episode] = info[EPISODE_KEY]["cost_bps"]
        signals = drawn[0]
        assert np.array_equal(drawn, np.tile(signals, (episodes, 1))), (
            "the signal path moved between episodes; the pin is not holding and "
            "this is not a conditional expectation"
        )
        trajectory = policy.trajectory

        moments = cost_moments(trajectory, case.market, signal=law, signals=signals)
        half_width = (
            float(CONDITIONAL["mean_sigmas"])
            * float(costs.std(ddof=1))
            / math.sqrt(episodes)
        )
        alpha = conditional_alpha_bps(trajectory, case.market, signals, law)
        # Non-vacuity, stated as the thing it actually needs to be: the SIGNAL-FREE
        # closed form has to fall outside the interval, so this cell cannot pass on
        # a formula with no alpha term at all. Expressed as a multiple of the
        # half-width because that is the same statement with a margin on it, and
        # the margin is per-path because the alpha term depends on the draw.
        assert abs(alpha) >= float(CONDITIONAL["alpha_half_widths"]) * half_width, (
            f"path {path_index}: the alpha term is {alpha:.4f} bps against a "
            f"half-width of {half_width:.4f}; this cell cannot resolve the term it "
            "exists to verify"
        )
        assert abs(float(costs.mean()) - cost_moments(trajectory, case.market).expected) > half_width, (
            f"path {path_index}: the signal-free closed form is inside the interval "
            "too, so this cell would pass on a formula with no alpha term"
        )
        assert abs(float(costs.mean()) - moments.expected) <= half_width, (
            f"path {path_index}: sampled {float(costs.mean()):.4f} against the "
            f"closed form {moments.expected:.4f} bps, outside ±{half_width:.4f}"
        )


# ---------------------------------------------------------------------------
# The conditioning set is the observation set
# ---------------------------------------------------------------------------


def test_the_grade_conditions_on_exactly_what_the_policy_observes():
    """The property that keeps "the reward is the grade" honest.

    M5's world exposes one stochastic seam and the grade conditions on one, and
    they are the same one. That has been true of every conditional grade since
    M4b and nothing in the repo asserted it — each was legitimate for a reason
    argued in prose, and prose is not a check.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    env = build_signal_env(case, int(M5_CONFIG["seeding"]["process_stream"]) + 30)
    assert observed_seams(env) == frozenset({SIGNAL_SEAM})
    assert check_conditioning_matches_observation(env, {SIGNAL_SEAM}) == frozenset(
        {SIGNAL_SEAM}
    )


def test_a_grade_that_conditions_on_less_than_the_policy_sees_is_refused():
    """The biased direction, on the world the brief explicitly backlogs.

    Stack M4b's stochastic liquidity under M5's signal — a real milestone, named
    as out of scope — and grade with ``signal_costs`` alone. The policy reacts to
    ``log L`` and the grade averages over it, so the number is a conditional
    expectation with respect to the wrong sigma-algebra: biased, in the direction
    of the policy's own cleverness, with every identity in this file still green.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    from temper.env import LiquidityStream, SignalStream
    from temper.seeding import LIQUIDITY_EVAL_POOL, SIGNAL_EVAL_POOL

    stacked = ExecutionEnv(
        case.market,
        case.order_size,
        case.lambda_risk,
        temporary_impact=impact_for(M5_ENCODING, case.market, case.order_size),
        liquidity=LiquidityStream(
            law=liquidity_for("lognormal", sigma_log=0.5), pool=LIQUIDITY_EVAL_POOL
        ),
        signal=SignalStream(signal=LAW, pool=SIGNAL_EVAL_POOL),
        root_seed=int(M5_CONFIG["seeding"]["root_seed"]),
        pool=M5_CONFIG["seeding"]["pool"],
        stream_index=int(M5_CONFIG["seeding"]["process_stream"]) + 31,
    )
    assert observed_seams(stacked) == frozenset({LIQUIDITY_SEAM, SIGNAL_SEAM})
    assert stacked.observation_space.shape == (4,)

    with pytest.raises(ConditioningMismatch, match="biased"):
        check_conditioning_matches_observation(stacked, {SIGNAL_SEAM})

    # And the graded path refuses it rather than producing a number, which is the
    # half that matters: a check nothing calls is a comment.
    with pytest.raises(ConditioningMismatch):
        signal_rollouts(
            _schedule_policy("twap", case),
            case.market,
            case.order_size,
            case.lambda_risk,
            temporary_impact=impact_for(M5_ENCODING, case.market, case.order_size),
            signal=SignalStream(signal=LAW, pool=SIGNAL_EVAL_POOL),
            liquidity=LiquidityStream(
                law=liquidity_for("lognormal", sigma_log=0.5), pool=LIQUIDITY_EVAL_POOL
            ),
            root_seed=int(M5_CONFIG["seeding"]["root_seed"]),
            pool=M5_CONFIG["seeding"]["pool"],
            paths=4,
        )


def test_a_grade_that_conditions_on_more_than_the_policy_sees_is_refused():
    """The other direction, and it is the more dangerous one.

    A grade that knows something the policy did not removes noise the policy
    actually faced: the interval collapses and the policy is scored as though it
    were more deterministic than it is. The extreme of it is conditioning on the
    realised price, where "expected cost" becomes realised cost and every agent
    looks perfect.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    env = build_signal_env(case, int(M5_CONFIG["seeding"]["process_stream"]) + 32)
    with pytest.raises(ConditioningMismatch, match="never observed"):
        check_conditioning_matches_observation(env, {SIGNAL_SEAM, LIQUIDITY_SEAM})


def test_an_uninformative_seam_is_not_in_the_observation_set():
    """Present is not the same as exposed, and the set is about information.

    A deterministic liquidity law, an absent signal, and a signal pointed at an
    already-committed shock all leave the observation at the width it has always
    had. They are correctly absent from the observation set, and a grade that
    conditioned on one of them would be conditioning on a constant.
    """
    case = case_by_id(M5_CONFIG["tiers"]["fast"]["cases"][0])
    from temper.env import NO_SIGNAL_STREAM, SignalStream
    from temper.seeding import SIGNAL_EVAL_POOL

    for label, stream, expected in (
        ("absent", NO_SIGNAL_STREAM, frozenset()),
        (
            "already-committed",
            SignalStream(
                signal=OneStepSignal(0.9, bins_ahead=0), pool=SIGNAL_EVAL_POOL
            ),
            frozenset(),
        ),
        (
            "informative",
            SignalStream(signal=LAW, pool=SIGNAL_EVAL_POOL),
            frozenset({SIGNAL_SEAM}),
        ),
    ):
        env = ExecutionEnv(
            case.market,
            case.order_size,
            case.lambda_risk,
            temporary_impact=impact_for(M5_ENCODING, case.market, case.order_size),
            liquidity=DETERMINISTIC_LIQUIDITY,
            signal=stream,
            root_seed=int(M5_CONFIG["seeding"]["root_seed"]),
            pool=M5_CONFIG["seeding"]["pool"],
            stream_index=int(M5_CONFIG["seeding"]["process_stream"]) + 33,
        )
        assert observed_seams(env) == expected, label
        assert env.observation_space.shape == (2 + len(expected),), label

    # Deterministic liquidity, for completeness: present, and exposing nothing.
    assert not LognormalLiquidity(0.0).stochastic


# ---------------------------------------------------------------------------
# The committed contract
# ---------------------------------------------------------------------------


def test_the_tiers_cover_what_the_brief_pre_stated():
    """The config is the contract; this is what stops it drifting quietly."""
    fast, deep = M5_CONFIG["tiers"]["fast"], M5_CONFIG["tiers"]["deep"]
    assert M5_CONFIG["world"]["cost_encoding"] == "power_law"
    assert "liquidity" not in M5_CONFIG["world"], (
        "M5's differential runs in M4a's world plus a signal, not M4b's; a "
        "liquidity block here would be a second noise source nobody named"
    )
    assert M5_CONFIG["world"]["signal"]["rho"] == RHO == 0.01
    assert M5_CONFIG["schedules"] == ["twap", "ac", "tangent", "optimal"]
    assert len(deep["cases"]) == 9 and len(fast["cases"]) == 3
    assert deep["n_sim"] == 200_000
    assert fast["expected_steps"] == len(signal_pairs("fast")) * fast["n_sim"] * 13
    assert deep["expected_steps"] == len(signal_pairs("deep")) * deep["n_sim"] * 13
    assert M5_CONFIG["seeding"]["pool"] == "m5/differential"
    assert float(DISCRIMINATION["rho"]) > RHO, (
        "the discriminative cell has to run at a rho the arithmetic can resolve, "
        "or it is not discriminative"
    )
