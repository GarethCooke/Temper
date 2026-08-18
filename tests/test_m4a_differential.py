"""M4a task 3 — the differential, re-run against the power-law env (invariant 6).

Constitution invariant 6: *no environment feature without an independent
expectation test*. The temporary-impact model became injectable and the injected
model changed, so the differential runs again — same tiers, same CI level, same
harness, with the analytic reference swapped from
:func:`~temper.oracle.cost.schedule_moments` to
:func:`~temper.oracle.cost.cost_moments`.

For each (case, schedule) cell of a tier, ``N_sim`` episodes run through the real
``step`` loop and their realised costs are standardised against the oracle's
moments for the schedule the env actually realised:

.. code::

    z_i = (C_i - E) / sqrt(V)

**The bands are still exact.** This is the one thing the power law could have
broken and did not, and it is worth being explicit about rather than inheriting
by habit. Temporary impact is a function of the *schedule*; it carries no shock.
So realised cost is still ``f(x) - sigma_bin * sum_k (x_k / X) * xi_k`` with only
``f`` changed, still affine in the draws, and a deterministic schedule's
shortfall is still *exactly* Gaussian. ``mean(z)`` is exactly ``N(0, 1/N)``,
``var(z)`` has variance exactly ``2/(N-1)``, and the 4-sigma bounds need no
chi-squared quantile and no scipy — exactly as in M1.

**Only one of the two moments needed a new reference.** ``V`` is unchanged
between the encodings, because ``shortfall_variance_bps2`` does not depend on the
impact model at all. That is asserted rather than assumed, here and in
``tests/test_objective_registry.py``, because it is the reason half of M1's work
carried over untouched.

Four schedules per cell, not three: in this world the tangent-derived sinh is a
schedule like any other and the certified power-law optimum is a fourth. Every
cell asserts its step count, because ``step_count``'s claim — ``N_sim`` episodes
went through *this* loop, one bin at a time — is what the single injected env was
kept single to preserve.

If a cell misses its band, the milestone's product is that finding. Do not tune
the env toward the bands.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from temper.agents import baseline
from temper.eval import sample_costs, standardise
from temper.oracle import cost_moments, schedule_moments

from .conftest import (
    M4A_CONFIG,
    M4A_ENCODING,
    DifferentialPair,
    build_power_law_env,
    power_law_pairs,
)

FAST = power_law_pairs("fast")
DEEP = power_law_pairs("deep")
CANONICAL_BINS = int(M4A_CONFIG["canonical_bins"])

_ELAPSED: dict[str, list[float]] = {"fast": [], "deep": []}
_STEPS: dict[str, list[int]] = {"fast": [], "deep": []}
_CELLS = {"fast": len(FAST), "deep": len(DEEP)}
_BAND_USE: dict[str, list[tuple[str, float, float]]] = {"fast": [], "deep": []}


@pytest.fixture(scope="module", autouse=True)
def report_runtime(request):
    """Print each tier's wall time and step count, and assert the step total.

    The two numbers are reported together for M1's reason: making the loop faster
    is fine, making it faster by processing more than one episode per ``step``
    call is what the counter exists to catch, and only the *rate* tells those
    apart.
    """
    yield
    lines = []
    for tier, elapsed in _ELAPSED.items():
        if not elapsed:
            continue
        budget = float(M4A_CONFIG["tiers"][tier]["runtime_budget_seconds"])
        total = sum(elapsed)
        steps = sum(_STEPS[tier])
        verdict = "within" if total <= budget else "OVER"
        rate = steps / total if total > 0 else float("nan")
        lines.append(
            f"  {tier:5s} {len(elapsed):2d} cells  {total:7.1f}s  "
            f"({verdict} the {budget:.0f}s budget)  {steps:,} steps  "
            f"{rate / 1000.0:.0f}k steps/s"
        )
    for tier, used in _BAND_USE.items():
        if not used:
            continue
        mean_cell, mean_use, _ = max(used, key=lambda row: row[1])
        var_cell, _, var_use = max(used, key=lambda row: row[2])
        lines.append(
            f"  {tier:5s} worst band use: mean {100 * mean_use:4.0f}% ({mean_cell}), "
            f"variance {100 * var_use:4.0f}% ({var_cell})"
        )

    if lines:
        writer = request.config.get_terminal_writer()
        writer.line("")
        writer.line("power-law differential runtime:")
        for line in lines:
            writer.line(line)

    for tier, steps in _STEPS.items():
        if len(steps) != _CELLS[tier]:
            continue  # a subset of the tier was selected; the per-cell asserts stand
        expected = int(M4A_CONFIG["tiers"][tier]["expected_steps"])
        assert sum(steps) == expected, (
            f"the {tier} tier made {sum(steps):,} calls into ExecutionEnv.step, not "
            f"the pre-stated {expected:,}; either the loop is not running every "
            "episode bin by bin or the tier is no longer what the config says"
        )


def _sample(pair: DifferentialPair, *, record_time: bool = True):
    """Run one cell's episodes and return ``(sample, oracle moments, steps)``."""
    case = pair.case
    policy = baseline(
        pair.schedule,
        case.market,
        case.order_size,
        case.lambda_risk,
        encoding=M4A_ENCODING,
    )
    env = build_power_law_env(case, pair.stream_index)
    assert env.cost_encoding == M4A_ENCODING

    started = time.perf_counter()
    sample = sample_costs(
        env,
        policy,
        pair.n_sim,
        seed=pair.stream_index,
        require_fixed_schedule=True,
    )
    if record_time:
        _ELAPSED[pair.tier].append(time.perf_counter() - started)
        _STEPS[pair.tier].append(env.step_count)

    # The *realised* schedule, not the planned one, and this world's moments.
    return sample, cost_moments(sample.trajectory, case.market), env.step_count


def _check(pair: DifferentialPair) -> None:
    """Run one cell and hold it to the config's bands, and to its step count."""
    sample, moments, steps = _sample(pair)

    expected_steps = pair.n_sim * pair.case.market.n_bins
    assert steps == expected_steps, (
        f"{pair}: ExecutionEnv.step was called {steps:,} times, not "
        f"{expected_steps:,} ({pair.n_sim:,} episodes x {pair.case.market.n_bins} "
        "bins) — the episodes did not all go through the real step loop, and the "
        "bands below are measuring something other than the env"
    )

    z = standardise(sample.costs, moments.expected, moments.variance)
    mean_z = float(np.mean(z))
    var_z = float(np.var(z, ddof=1))

    assert abs(mean_z) <= pair.mean_band, (
        f"{pair}: mean(z) = {mean_z:+.5f}, band {pair.mean_band:.5f} "
        f"(E = {moments.expected:.6f} bps over {pair.n_sim} episodes) — the "
        "power-law env's mean cost disagrees with the closed form"
    )
    assert abs(var_z - 1.0) <= pair.var_band, (
        f"{pair}: var(z) - 1 = {var_z - 1.0:+.5f}, band {pair.var_band:.5f} "
        f"(V = {moments.variance:.4f} bps^2 over {pair.n_sim} episodes) — the "
        "power-law env's cost dispersion disagrees with the closed form"
    )

    _BAND_USE[pair.tier].append(
        (str(pair), abs(mean_z) / pair.mean_band, abs(var_z - 1.0) / pair.var_band)
    )


@pytest.mark.parametrize("pair", FAST, ids=str)
def test_fast_tier(pair):
    """One case per symbol at the middle lambda, in `make test`."""
    _check(pair)


@pytest.mark.deep
@pytest.mark.parametrize("pair", DEEP, ids=str)
def test_deep_tier(pair):
    """The full 3 x 3 golden grid — `make differential`, the acceptance gate."""
    _check(pair)


# ---------------------------------------------------------------------------
# What the swap did and did not change
# ---------------------------------------------------------------------------


def test_the_variance_reference_is_the_one_m1_already_certified(golden_case):
    """``V`` is the same number in both worlds, so only ``E`` needed a new reference.

    Stated as arithmetic on every vendored case rather than argued from the
    formula. It is the reason this module could reuse M1's tiers verbatim: half
    of what the bands check was never touched by the impact model.
    """
    market = golden_case.market
    for name in ("ac", "twap"):
        trajectory = np.asarray(getattr(golden_case, name)["trajectory"], dtype=float)
        assert (
            cost_moments(trajectory, market).variance
            == schedule_moments(trajectory, market).variance
        )


def test_the_expectation_reference_did_change(golden_case):
    """Non-vacuity: the mean band is being held to a different number than M1's.

    If the two encodings' ``E`` agreed, this tier would be M1's tier again under
    a new name and would say nothing about the power law.
    """
    market = golden_case.market
    for name in ("ac", "twap"):
        trajectory = np.asarray(getattr(golden_case, name)["trajectory"], dtype=float)
        power = cost_moments(trajectory, market).expected
        linear = schedule_moments(trajectory, market).expected
        assert abs(power - linear) > 1e-6 * abs(power)


def test_the_bands_reject_a_wrong_answer():
    """The differential has teeth: perturb the reference and it must fail.

    A green differential is only evidence if red was reachable. This re-scores one
    fast cell against deliberately wrong moments — a mean shifted by ten band
    widths, and a variance mis-scaled by 5 % — and requires both to breach.
    """
    pair = FAST[0]
    sample, moments, _ = _sample(pair, record_time=False)

    shifted = standardise(
        sample.costs,
        moments.expected + 10.0 * pair.mean_band * np.sqrt(moments.variance),
        moments.variance,
    )
    assert abs(float(np.mean(shifted))) > pair.mean_band

    mis_scaled = standardise(sample.costs, moments.expected, moments.variance * 1.05)
    assert abs(float(np.var(mis_scaled, ddof=1)) - 1.0) > pair.var_band


def test_how_far_the_tiers_can_tell_the_two_encodings_apart():
    """Where the differential can see the world it is in, and where it cannot.

    The obvious claim to want here is "score a power-law cell against Phase 1's
    moments and the band breaks". It is *false at most cells*, and finding that
    out is worth more than the claim would have been.

    The separation is ``|E_power - E_linear| / sigma_C`` measured against the
    tier's mean band. At the fast tier it is under the band on every core cell:
    for AAPL the two encodings differ by 0.19 bps against a per-episode cost SD
    of 93 bps, which is 0.002 of sigma_C against a band of 0.0283. Twenty
    thousand episodes cannot distinguish them. At the deep tier ten times the
    episodes buy a band of 0.0089 and the JPM cells (0.0132-0.111) and MSFT's
    high-lambda AC cell (0.0476) come clear of it, while every AAPL cell except
    one stays under.

    So the tiers are **not** M4a's gate on the cost assembly, and this test
    records that rather than pretending otherwise. That job belongs to the exact
    per-episode noise identity, which pins the realised cost against an
    independently assembled reference to 1e-12 with no sampling at all — the same
    division of labour ``ARCHITECTURE.md`` §9 records for M1a, where the tiers
    stopped being the cost gate and became a check on the *draws*. What the tiers
    certify here is what they certified there: that the shocks are iid standard
    normal and uncorrelated across bins, in a world whose cost is still affine in
    them.

    The assertion is the honest residue: the deep tier separates on *some* cells,
    so it is not blind, and the count is reported rather than assumed.
    """
    deep_band = float(DEEP[0].mean_band)
    fast_band = float(FAST[0].mean_band)
    separations = {}
    for pair in DEEP:
        case = pair.case
        policy = baseline(
            pair.schedule,
            case.market,
            case.order_size,
            case.lambda_risk,
            encoding=M4A_ENCODING,
        )
        power = cost_moments(policy.trajectory, case.market)
        linear = schedule_moments(
            policy.trajectory, case.market, order_size=case.order_size
        )
        separations[str(pair)] = abs(power.expected - linear.expected) / np.sqrt(
            power.variance
        )

    resolved = {k: v for k, v in separations.items() if v > deep_band}
    assert resolved, (
        "no deep cell separates the two encodings at 4 sigma; the tiers cannot "
        "see the world they are in *at all*, which would make the reference swap "
        "untestable by sampling"
    )
    assert max(separations.values()) > 10.0 * deep_band, (
        "the deep tier's best separation is under ten band widths; the cell list "
        "no longer contains a case where the encodings are far apart"
    )
    # And the fast tier genuinely cannot, which is why `make test` is not where
    # this milestone's cost claim is settled.
    assert max(separations.values()) > fast_band > min(separations.values())


# ---------------------------------------------------------------------------
# The tiers are what the config says they are
# ---------------------------------------------------------------------------


def test_the_tiers_cover_what_the_brief_pre_stated():
    """The config is the spec; this is the guard against it being quietly trimmed."""
    schedules = M4A_CONFIG["schedules"]
    assert set(schedules) == {"twap", "ac", "tangent", "optimal"}
    assert len(FAST) == 3 * len(schedules)
    assert len(DEEP) == 9 * len(schedules)
    assert M4A_CONFIG["tiers"]["fast"]["n_sim"] >= 20_000
    assert M4A_CONFIG["tiers"]["deep"]["n_sim"] >= 200_000

    streams = [(pair.tier, pair.stream_index) for pair in FAST + DEEP]
    assert len(set(streams)) == len(streams)
    assert not {pair.stream_index for pair in FAST} & {pair.stream_index for pair in DEEP}


@pytest.mark.parametrize(("tier", "pairs"), [("fast", FAST), ("deep", DEEP)])
def test_the_pre_stated_step_totals_are_the_arithmetic_of_the_config(tier, pairs):
    """``expected_steps`` is cells x N_sim x N, and N is the canonical 13."""
    cells = {(pair.case.case_id, pair.schedule) for pair in pairs}
    assert len(cells) == len(pairs), f"the {tier} tier lists a cell twice"

    bins = {pair.case.market.n_bins for pair in pairs}
    assert bins == {CANONICAL_BINS}

    n_sim = int(M4A_CONFIG["tiers"][tier]["n_sim"])
    assert len(pairs) * n_sim * CANONICAL_BINS == int(
        M4A_CONFIG["tiers"][tier]["expected_steps"]
    )


def test_the_deep_tier_is_the_full_golden_grid():
    """Three symbols x three lambdas, exactly the M0 core cases."""
    covered = {(pair.case.symbol, pair.case.lambda_risk) for pair in DEEP}
    assert len(covered) == 9
    assert {symbol for symbol, _ in covered} == {"AAPL", "MSFT", "JPM"}
    assert {lam for _, lam in covered} == {1e-7, 1e-5, 1e-3}


def test_the_certified_optimum_is_one_of_the_schedules_under_test():
    """The reference an agent is graded against also goes through the env.

    Every other schedule in this tier was already a policy; the power-law optimum
    is new, is the thing M4a grades against, and would otherwise be the one
    trajectory in the milestone that the simulator had never been asked to price.
    """
    assert "optimal" in M4A_CONFIG["schedules"]
    case = FAST[0].case
    from temper.oracle import power_law_optimum

    policy = baseline(
        "optimal", case.market, case.order_size, case.lambda_risk, encoding=M4A_ENCODING
    )
    expected = power_law_optimum(case.market, case.order_size, case.lambda_risk)
    assert policy.trajectory == pytest.approx(expected, rel=0.0, abs=0.0)
