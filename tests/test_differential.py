"""Task 4 — the analytic differential: what the env does vs what the maths says.

For each (case, schedule) cell of a tier, ``N_sim`` episodes run through the real
``step`` loop — no vectorised shortcut, because the loop is the thing under test —
and their realised costs are standardised against
:func:`~temper.oracle.cost.schedule_moments` of the schedule the env actually
realised:

.. code::

    z_i = (C_i - E) / sqrt(V)

Under Phase-1 dynamics a deterministic schedule's shortfall is a fixed linear
combination of independent Gaussian shocks, so it is *exactly* Gaussian: ``z`` is
exactly standard normal under the null, ``mean(z)`` is exactly ``N(0, 1/N)`` and
``var(z)`` has variance exactly ``2/(N-1)``. The bands are therefore exact rather
than asymptotic, which is why they need no chi-squared quantile and this file
needs no scipy.

What the bands buy (the point of stating them):

============  ==========  =============  ==============================
Tier          N_sim       mean(z) band   detects a mean shift of
============  ==========  =============  ==============================
fast           20 000     0.0283         ~2.8 % of sigma_C
deep          100 000     0.0126         ~1.3 % of sigma_C
============  ==========  =============  ==============================

with variance bands of 4.0 % and 1.8 % respectively. The two bug classes M0 said
to watch for — grading against the wrong kappa (~18 % on the objective) and an
off-by-one in ``sum x_k^2`` (~2-5 % on V, 21 % for TWAP at N = 13) — are both far
outside the deep tier's bands, and the deep tier is the acceptance gate.

If a cell misses its band, the milestone's product is that finding. Do not tune
the env toward the bands.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from temper.agents import baseline
from temper.eval import sample_costs, standardise
from temper.oracle import schedule_moments

from .conftest import M1_CONFIG, DifferentialPair, build_env, differential_pairs

FAST = differential_pairs("fast")
DEEP = differential_pairs("deep")

#: Wall time per cell, filled in as the tiers run and reported at teardown so the
#: runtime budget in the config is a number someone can see, not a hope.
_ELAPSED: dict[str, list[float]] = {"fast": [], "deep": []}


@pytest.fixture(scope="module", autouse=True)
def report_runtime(request):
    """Print each tier's measured wall time against its committed budget."""
    yield
    lines = []
    for tier, elapsed in _ELAPSED.items():
        if not elapsed:
            continue
        budget = float(M1_CONFIG["tiers"][tier]["runtime_budget_seconds"])
        total = sum(elapsed)
        verdict = "within" if total <= budget else "OVER"
        lines.append(
            f"  {tier:5s} {len(elapsed):2d} cells  {total:7.1f}s  "
            f"({verdict} the {budget:.0f}s budget)"
        )
    if lines:
        writer = request.config.get_terminal_writer()
        writer.line("")
        writer.line("differential runtime:")
        for line in lines:
            writer.line(line)


def _sample(pair: DifferentialPair, *, record_time: bool = True):
    """Run one cell's episodes and return ``(costs, oracle moments)``."""
    case = pair.case
    policy = baseline(pair.schedule, case.market, case.order_size, case.lambda_risk)
    env = build_env(case, pair.stream_index)

    started = time.perf_counter()
    sample = sample_costs(
        env,
        policy,
        pair.n_sim,
        seed=pair.stream_index,
        require_fixed_schedule=True,
    )
    if record_time:  # the tier's budget is the tier's cells, not this module's
        _ELAPSED[pair.tier].append(time.perf_counter() - started)

    # The *realised* schedule, not the planned one: if the env had to
    # force-liquidate a tail, that is the schedule whose moments apply.
    return sample, schedule_moments(sample.trajectory, case.market)


def _check(pair: DifferentialPair) -> None:
    """Run one cell and hold it to the config's bands."""
    sample, moments = _sample(pair)
    z = standardise(sample.costs, moments.expected, moments.variance)
    mean_z = float(np.mean(z))
    var_z = float(np.var(z, ddof=1))  # unbiased; the band is stated for this estimator

    assert abs(mean_z) <= pair.mean_band, (
        f"{pair}: mean(z) = {mean_z:+.5f}, band {pair.mean_band:.5f} "
        f"(E = {moments.expected:.6f} bps over {pair.n_sim} episodes) — the env's "
        "mean cost disagrees with the closed form"
    )
    assert abs(var_z - 1.0) <= pair.var_band, (
        f"{pair}: var(z) - 1 = {var_z - 1.0:+.5f}, band {pair.var_band:.5f} "
        f"(V = {moments.variance:.4f} bps^2 over {pair.n_sim} episodes) — the env's "
        "cost dispersion disagrees with the closed form"
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


def test_the_bands_reject_a_wrong_answer():
    """The differential has teeth: perturb the reference and it must fail.

    A green differential is only evidence if red was reachable. This re-scores one
    fast cell against deliberately wrong moments — a mean shifted by half a
    standard deviation of the *estimator* times ten, and a variance mis-scaled by
    5 % — and requires both to breach their bands. The variance figure is the
    size of the off-by-one-in ``sum x_k^2`` class M0 flagged; if this test ever
    passes vacuously, so is the tier above it.
    """
    pair = FAST[0]
    sample, moments = _sample(pair, record_time=False)

    shifted = standardise(
        sample.costs,
        moments.expected + 10.0 * pair.mean_band * np.sqrt(moments.variance),
        moments.variance,
    )
    assert abs(float(np.mean(shifted))) > pair.mean_band

    mis_scaled = standardise(sample.costs, moments.expected, moments.variance * 1.05)
    assert abs(float(np.var(mis_scaled, ddof=1)) - 1.0) > pair.var_band


def test_the_tiers_cover_what_the_brief_pre_stated():
    """The config is the spec; this is the guard against it being quietly trimmed.

    Cutting fast-tier *cases* is the brief's sanctioned response to a runtime
    overrun, and cutting N_sim is not. Both go through an amendment to the brief,
    so both should fail here first.
    """
    schedules = M1_CONFIG["schedules"]
    assert set(schedules) == {"twap", "ac", "optimal"}
    assert len(FAST) == 3 * len(schedules)
    assert len(DEEP) == 9 * len(schedules)
    assert M1_CONFIG["tiers"]["fast"]["n_sim"] >= 20_000
    assert M1_CONFIG["tiers"]["deep"]["n_sim"] >= 100_000

    # Every deep cell gets its own stream, and no fast cell shares one with it.
    streams = [(pair.tier, pair.stream_index) for pair in FAST + DEEP]
    assert len(set(streams)) == len(streams)
    assert not {pair.stream_index for pair in FAST} & {pair.stream_index for pair in DEEP}


def test_the_deep_tier_is_the_full_golden_grid():
    """Three symbols x three lambdas, exactly the M0 core cases."""
    covered = {(pair.case.symbol, pair.case.lambda_risk) for pair in DEEP}
    assert len(covered) == 9
    assert {symbol for symbol, _ in covered} == {"AAPL", "MSFT", "JPM"}
    assert {lam for _, lam in covered} == {1e-7, 1e-5, 1e-3}
