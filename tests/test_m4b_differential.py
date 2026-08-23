"""M4b task 4 — the differential, through two seams (invariant 6).

Constitution invariant 6: *no environment feature without an independent
expectation test*. M4b adds two features — a second noise source and a third
observation coordinate — so the differential runs again, and it runs in **two
parts**, because the two features fail in different ways and one check cannot see
both.

**The process.** The env's own draws against
:mod:`temper.oracle.liquidity`'s closed forms: ``E[L] = 1``, the variance, and
``E[L^-0.6]`` — the moment the whole static rung is priced by. Plus lag-1
autocorrelation, because "i.i.d." is not decoration here: it is what makes
``(k, x_k, L_k)`` a sufficient statistic, and therefore what makes the dynamic
program the optimum over *all* adapted policies rather than only over the ones
with that observation. Two deliberately different routes to one distribution.

**The world, conditionally.** ``z = (C - E[cost | L]) / sqrt(V)``, standardised
per episode against its *own* realised path. This is exactly ``N(0, 1)`` under the
null and M1's bands survive **verbatim** — no chi-squared quantile, no scipy —
for a reason worth stating rather than inheriting: liquidity enters ``E[cost]``
and does not touch ``V`` at all, so conditioned on the path a deterministic
schedule's shortfall is still a fixed linear combination of independent Gaussian
shocks. The frozen objective's ``V`` is *price*-shortfall variance and nothing
else, which is also why invariant 7 needed no amendment.

**The world, unconditionally.** The sample mean cost against the closed form
``A E[L^-0.6] sum w^(1+0.6) + permanent + spread``. Not exact — realised cost is
not Gaussian once liquidity is in it — so the band is a sample-standard-error one
and says so. This is the half that catches the multiplier applied with the wrong
power or the wrong sign, which the conditional check *cannot* see: it conditions
the mistake away.

Every cell asserts its step count. ``step_count``'s claim — ``N_sim`` episodes
went through *this* loop, one bin at a time — is what the single injected env was
kept single to preserve, and M4b injects a second model rather than subclassing
for the same reason M4a injected the first.

If a cell misses its band, the milestone's product is that finding. Do not tune
the env toward the bands.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from temper.agents import baseline
from temper.eval import sample_costs, standardise
from temper.oracle import (
    LognormalLiquidity,
    cost_moments,
    expected_cost_moments,
    static_optimum,
    trades,
)

from .conftest import (
    M4B_CONFIG,
    M4B_ENCODING,
    DifferentialPair,
    build_liquidity_env,
    case_by_id,
    liquidity_pairs,
    m4b_liquidity_law,
)

LAW = m4b_liquidity_law()
PROCESS = M4B_CONFIG["process"]


#: The reference-table name for each schedule, and the *baseline* name it is
#: reached by where one exists. ``m4a`` is the power-law world's ``optimal`` —
#: the certified optimum that knows no liquidity — renamed here because in this
#: world it is no longer the answer, exactly as M4a renamed the sinh ``tangent``
#: when the power law stopped being the world it solved.
BASELINE_NAMES = {"twap": "twap", "ac": "ac", "tangent": "tangent", "m4a": "optimal"}


def _schedule_policy(name: str, case):
    """The five reference schedules as policies, through the shared baselines.

    ``static`` is not in :mod:`temper.agents.baselines` because it depends on the
    liquidity law, which the baselines module has no business knowing about — so
    it is built here from the oracle and wrapped in the same ``SchedulePolicy``
    the other four use, which keeps "every graded thing runs through identical
    code" (§5) true rather than nearly true.
    """
    if name in BASELINE_NAMES:
        return baseline(
            BASELINE_NAMES[name],
            case.market,
            case.order_size,
            case.lambda_risk,
            encoding=M4B_ENCODING,
        )
    from temper.agents.baselines import SchedulePolicy

    return SchedulePolicy(
        static_optimum(case.market, case.order_size, case.lambda_risk, LAW),
        name="static",
    )


# ---------------------------------------------------------------------------
# The process
# ---------------------------------------------------------------------------


def test_the_envs_own_liquidity_draws_match_the_oracles_closed_forms():
    """Invariant 6, applied to a distribution rather than to a market.

    The env draws; the oracle states a closed form; neither computes the other's
    number. A dropped Jensen correction — ``exp(sigma Z)`` instead of
    ``exp(sigma Z - sigma^2/2)`` — would leave every schedule still liquidating,
    every reward still looking like a reward, and ``E[L]`` at 1.13 instead of 1,
    which is exactly the size of the level shift this milestone is measuring.
    """
    case = case_by_id(M4B_CONFIG["tiers"]["fast"]["cases"][0])
    env = build_liquidity_env(case, int(M4B_CONFIG["seeding"]["process_stream"]))
    draws = int(PROCESS["draws"])
    episodes = draws // case.market.n_bins

    policy = _schedule_policy("twap", case)
    sample = sample_costs(env, policy, episodes, record_liquidity=True)
    paths = sample.liquidity
    assert paths is not None and paths.shape == (episodes, case.market.n_bins)
    flat = paths.reshape(-1)

    mean = float(flat.mean())
    band = float(PROCESS["mean_sigmas"]) * float(flat.std(ddof=1)) / math.sqrt(flat.size)
    assert abs(mean - LAW.mean_multiplier()) <= band, (
        f"the env's liquidity has mean {mean:.6f}, not 1, outside {band:.6f}; a "
        "multiplier whose mean is not one changes the market's total liquidity "
        "rather than reallocating it, and every rung in the reference table is "
        "priced on the reallocation"
    )
    assert float(flat.var(ddof=1)) == pytest.approx(
        LAW.variance(), rel=float(PROCESS["variance_rtol"])
    )
    assert float(np.mean(flat ** (-case.market.temp_exponent))) == pytest.approx(
        LAW.inverse_power_moment(case.market.temp_exponent),
        rel=float(PROCESS["inverse_moment_rtol"]),
    )


def test_the_liquidity_draws_are_independent_across_bins():
    """i.i.d. is the assumption the dynamic program's sufficiency rests on.

    Lag-1 autocorrelation, *within* an episode and across the episode boundary,
    against its own ``1/sqrt(n)`` null. A process with memory would make
    ``(k, x_k, L_k)`` insufficient, the DP no longer the optimum over all adapted
    policies, and M4b's denominator quietly wrong — while every other check in
    this file stayed green.
    """
    case = case_by_id(M4B_CONFIG["tiers"]["fast"]["cases"][0])
    env = build_liquidity_env(case, int(M4B_CONFIG["seeding"]["process_stream"]) + 1)
    episodes = int(PROCESS["draws"]) // case.market.n_bins
    sample = sample_costs(
        env, _schedule_policy("twap", case), episodes, record_liquidity=True
    )
    flat = np.log(sample.liquidity.reshape(-1))

    centred = flat - flat.mean()
    lag1 = float(np.sum(centred[:-1] * centred[1:]) / np.sum(centred**2))
    band = float(PROCESS["autocorrelation_sigmas"]) / math.sqrt(centred.size)
    assert abs(lag1) <= band, (
        f"log-liquidity has lag-1 autocorrelation {lag1:.5f}, outside {band:.5f}: "
        "the process is not memoryless, so (k, x_k, L_k) is not a sufficient "
        "statistic and the dynamic program is not the optimum over adapted policies"
    )


def test_the_draws_reproduce_from_the_seed_address():
    """Invariant 1, on the second noise source: same address, same market.

    The address is the *constructor's*, and the envs are reset with no argument.
    ``reset(seed=i)`` means "rewind to stream ``i``" — it re-addresses the price
    stream and the liquidity stream together, deliberately — so passing it here
    would put both envs on stream 0 and the comparison below would be between one
    address and itself. That is not a hypothetical: it is what the first version
    of this test did, and it passed the reproduction half while making the
    independence half vacuous.
    """
    case = case_by_id(M4B_CONFIG["tiers"]["fast"]["cases"][0])
    stream = int(M4B_CONFIG["seeding"]["process_stream"]) + 2
    first, second = (build_liquidity_env(case, stream) for _ in range(2))
    first.reset()
    second.reset()
    assert np.array_equal(first.multipliers, second.multipliers)
    assert not np.array_equal(first.multipliers, np.ones(case.market.n_bins))

    other = build_liquidity_env(case, stream + 1)
    other.reset()
    assert not np.array_equal(first.multipliers, other.multipliers), (
        "two stream indices drew the same liquidity path; the second noise "
        "source is not addressed"
    )

    # And rewinding does re-address both, which is the semantics the sentence
    # above relies on. Stated as an assertion rather than as a comment, because
    # it is the property a future `reset` refactor could silently drop.
    other.reset(seed=stream)
    assert np.array_equal(other.multipliers, first.multipliers), (
        "reset(seed=i) did not re-address the liquidity stream; the two noise "
        "sources are no longer read off one address"
    )


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------


def _run_cell(pair: DifferentialPair) -> dict:
    """One (case, schedule) cell: both checks, one pass through the loop."""
    case = pair.case
    env = build_liquidity_env(case, pair.stream_index)
    policy = _schedule_policy(pair.schedule, case)

    before = env.step_count
    started = time.perf_counter()
    sample = sample_costs(
        env,
        policy,
        pair.n_sim,
        require_fixed_schedule=True,
        record_liquidity=True,
    )
    seconds = time.perf_counter() - started
    steps = env.step_count - before

    trajectory = sample.trajectory
    variance = cost_moments(trajectory, case.market).variance
    conditional = np.array(
        [
            cost_moments(trajectory, case.market, liquidity=path).expected
            for path in sample.liquidity
        ]
    )
    z = (sample.costs - conditional) / math.sqrt(variance)
    unconditional = expected_cost_moments(trajectory, case.market, LAW).expected
    return {
        "steps": steps,
        "seconds": seconds,
        "mean_z": float(z.mean()),
        "var_z": float(z.var(ddof=1)),
        "mean_cost": float(sample.costs.mean()),
        "reference": unconditional,
        "cost_se": float(sample.costs.std(ddof=1)) / math.sqrt(pair.n_sim),
    }


def _assert_cell(pair: DifferentialPair, observed: dict, spec: dict) -> None:
    assert observed["steps"] == pair.n_sim * pair.case.market.n_bins, (
        f"{pair}: the loop was called {observed['steps']} times for "
        f"{pair.n_sim} episodes of {pair.case.market.n_bins} bins; something "
        "took a route around ExecutionEnv.step"
    )
    assert abs(observed["mean_z"]) <= pair.mean_band, (
        f"{pair}: mean(z) = {observed['mean_z']:+.5f} outside ±{pair.mean_band:.5f}. "
        "The conditional band is EXACT — liquidity does not touch V — so this is "
        "the env and the conditional cost function disagreeing, not sampling."
    )
    assert abs(observed["var_z"] - 1.0) <= pair.var_band, (
        f"{pair}: var(z) = {observed['var_z']:.5f} outside 1 ± {pair.var_band:.5f}"
    )
    band = float(spec["mean_cost_sigmas"]) * observed["cost_se"]
    assert abs(observed["mean_cost"] - observed["reference"]) <= band, (
        f"{pair}: unconditional mean cost {observed['mean_cost']:.6f} bps against "
        f"the closed form {observed['reference']:.6f}, outside ±{band:.6f}. This "
        "is the check the conditional one cannot make — it conditions on the very "
        "path a wrong exponent or a flipped sign would have distorted."
    )


@pytest.mark.parametrize("pair", liquidity_pairs("fast"), ids=str)
def test_fast_tier(pair):
    """In ``make test``: fifteen cells at 10 000 episodes each."""
    _assert_cell(pair, _run_cell(pair), M4B_CONFIG["tiers"]["fast"])


@pytest.mark.deep
@pytest.mark.parametrize("pair", liquidity_pairs("deep"), ids=str)
def test_deep_tier(pair):
    """``make differential``: the full 3 x 3 golden grid at M1's N_sim."""
    _assert_cell(pair, _run_cell(pair), M4B_CONFIG["tiers"]["deep"])


def test_the_graded_variance_is_price_only_and_liquidity_never_enters_it():
    """The distinction the brief asks to be written down, written down as a test.

    The frozen objective penalises *price*-shortfall variance. Liquidity
    dispersion enters ``E[cost]`` through Jensen and never ``lambda V``, so
    invariant 7 holds with no amendment — one functional, still encoded once. The
    realised-cost variance this differential measures now has two sources while
    the graded ``V`` has one, and that is a distinction that drifts silently
    unless something checks it.
    """
    case = case_by_id(M4B_CONFIG["tiers"]["fast"]["cases"][0])
    trajectory = static_optimum(case.market, case.order_size, case.lambda_risk, LAW)
    plain = cost_moments(trajectory, case.market)
    for sigma in (0.0, 0.25, 0.9):
        law = LognormalLiquidity(sigma)
        under_law = expected_cost_moments(trajectory, case.market, law)
        assert under_law.variance == plain.variance
        assert under_law.temporary == plain.temporary * law.inverse_power_moment(
            case.market.temp_exponent
        )

    # And realised cost really does carry the second source: the unconditional
    # spread must exceed the graded V, or the check above is about nothing.
    env = build_liquidity_env(case, int(M4B_CONFIG["seeding"]["process_stream"]) + 3)
    sample = sample_costs(env, _schedule_policy("static", case), 4000)
    assert float(sample.costs.var(ddof=1)) > plain.variance, (
        "realised cost is no more dispersed than the price shock alone implies; "
        "the liquidity multiplier is not reaching the charge"
    )


def test_the_tiers_cover_what_the_brief_pre_stated():
    """The config is the contract; this is the check that it still says so."""
    for tier, spec in M4B_CONFIG["tiers"].items():
        pairs = liquidity_pairs(tier)
        assert len(pairs) == len(spec["cases"]) * len(M4B_CONFIG["schedules"])
        assert len({p.stream_index for p in pairs}) == len(pairs), (
            f"{tier}: two cells share a stream index"
        )
        assert (
            sum(p.n_sim * p.case.market.n_bins for p in pairs)
            == spec["expected_steps"]
        )
    assert tuple(M4B_CONFIG["schedules"]) == ("twap", "ac", "tangent", "m4a", "static")
    assert M4B_CONFIG["world"]["liquidity"]["model"] == "lognormal"
    assert M4B_CONFIG["seeding"]["pool"] == "m4b/differential"
