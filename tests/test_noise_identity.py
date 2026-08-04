"""M1a task 1 — the exact noise identity, which retires the last statistical leg.

M1 left the differential resting on statistics in exactly one place. Test 5(b)
pins the inventory penalty against ``lambda * V`` exactly; the per-episode
expectation identity pins the deterministic skeleton exactly; so the only claim
still made by sampling was that the env's *realised noise carries the right
variance*. This module removes that too.

The claim
---------
For a deterministic schedule under Phase-1 dynamics the realised cost splits
cleanly into a deterministic part and a linear functional of the shocks::

    C  =  E[cost]  +  noise(xi)

and ``noise`` is not merely mean-zero with the right variance — it is a specific
linear combination of the specific draws the env made, episode by episode. If
that identity holds to round-off then ``V = sigma^2 tau sum_k x_k^2`` follows *by
construction*: the variance of a known linear form in iid standard normals is
arithmetic, not an estimate. The Monte-Carlo tiers then certify only that the
draws really are iid standard normal and uncorrelated across bins — belt and
braces, no longer the gate.

The functional, stated rather than fitted
-----------------------------------------
Assembled here from the constitution §4 dynamics and the case parameters, with
nothing imported from the env's internals and nothing from an oracle helper that
already computes it::

    noise(xi)  =  -sigma_bin_bps * sum_{k=0}^{N-1} (x_k / X) * xi_k

* ``sigma_bin_bps = sigma_daily * sqrt(tau / TRADING_HOURS_PER_DAY) * BPS`` — built
  from the raw symbol parameters and the grid, not read off ``Market.sigma_bin``.
* ``x_k`` is inventory **before** bin ``k`` executes, so the sum runs over
  ``k = 0 .. N-1`` and its first term is the whole order, ``x_0 = X``. See
  "which x_k" below — this is the load-bearing index convention, not a detail.
* ``xi_k`` are the raw draws, regenerated in this module from the *seed address*
  through :func:`temper.seeding.pool_rng`. The env is never asked what it drew.
* **The sign is sell-side, per constitution §4, and is stated not inferred.** A
  positive shock raises the price; a seller sells into it and does better; the
  shortfall against arrival therefore *falls*. Hence the leading minus. Nothing
  below matches a sign against the env to find out which way round it goes.

Which ``x_k`` — before the bin, not after
-----------------------------------------
M1a's brief writes the functional as ``sigma sqrt(tau) * sum_{k=1}^{N-1} x_k xi_k``
over *post-bin* holdings. That is the textbook Almgren–Chriss convention, in which
the first trade executes at the arrival price and bears no volatility. It is
**not** this project's convention, and using it here would make this test red.

The vendored FrontierView model — normative under invariant 2, and what
``oracle.shortfall_variance_bps2`` reproduces to 9.5e-16 — charges the shock
*before* the bin executes, so every share still held at the start of a bin bears
it and the sum keeps its ``x_0 = X`` term. §4 defers to the goldens for the
numeric spec ("the goldens, not this document"), so the goldens decide.

The gap between the two conventions is exactly one bin's variance on the whole
order, and it is the named off-by-one-in-``sum x_k^2`` bug class:
``sum_{k=0}^{N-1}(x_k/X)^2 - sum_{k=1}^{N}(x_k/X)^2 = 1`` identically, which for
TWAP at N = 13 is 26 % of V — the figure the deep tier was sized to detect.
:func:`test_the_post_bin_convention_is_the_named_off_by_one` pins that algebraically,
so this module states which convention is in force *and* measures what choosing the
other one would cost.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from temper.agents import baseline
from temper.eval import run_episode
from temper.oracle import BPS, TRADING_HOURS_PER_DAY, schedule_moments
from temper.seeding import pool_rng

from .conftest import M1_CONFIG, build_env, case_by_id, guard_case
from .test_env_identities import UnderTradePolicy, assert_identity
from .test_sinh_asymptote_guard import asymptote_policy

IDENTITIES = M1_CONFIG["identities"]
NOISE_RTOL = float(IDENTITIES["noise_rtol"])
EPISODES = int(IDENTITIES["episodes"])
UNDERTRADE_FRACTION = float(IDENTITIES["undertrade_fraction"])
SEEDING = M1_CONFIG["seeding"]
ROOT_SEED = int(SEEDING["root_seed"])
POOL = SEEDING["pool"]

#: Constitution §4 is sell-side only. A positive shock raises the price, the
#: seller realises more, and the shortfall against arrival falls — so the noise
#: enters the *cost* with a minus. Stated here, once, as a constant, so that no
#: line below can quietly acquire the opposite sign by matching the env.
SELL_SIDE_SIGN = -1.0

#: Every (case, schedule) cell of the deep grid — the noise identity is cheap
#: enough per episode that there is no reason to check it on fewer cells than
#: the acceptance tier itself covers.
NOISE_CASES = [
    case_by_id(case_id)
    for case_id in M1_CONFIG["tiers"][IDENTITIES["noise_cases"]]["cases"]
]
SCHEDULES = list(M1_CONFIG["schedules"])

#: Worst relative residual seen, for the acceptance report.
_WORST: dict[str, float] = {}


@pytest.fixture(scope="module", autouse=True)
def report_worst(request):
    """Print the worst residual against the band — evidence, not just a pass."""
    yield
    if not _WORST:
        return
    where, worst = max(_WORST.items(), key=lambda item: item[1])
    writer = request.config.get_terminal_writer()
    writer.line("")
    writer.line(
        f"noise identity: worst {worst:.2e} relative (band {NOISE_RTOL:g}), "
        f"{len(_WORST)} cells x {EPISODES} episodes, worst at {where}"
    )


# ---------------------------------------------------------------------------
# The functional, assembled from §4 and the parameters
# ---------------------------------------------------------------------------


def shock_scale_bps(case) -> float:
    """``sigma_bin`` in bps, from the raw symbol parameters and the grid.

    Deliberately not ``case.market.sigma_bin``: the whole point of this module is
    that the functional is rebuilt from the units contract
    (``ARCHITECTURE.md`` §9, 2026-08-04) rather than borrowed from something that
    already knows the answer. :func:`test_the_assembled_scale_matches_the_units_contract`
    checks the rebuild agrees with the oracle's, which is a cross-check on the
    oracle, not a source for this number.
    """
    market = case.market
    tau = market.horizon_hours / market.n_bins
    return market.params.sigma * math.sqrt(tau / TRADING_HOURS_PER_DAY) * BPS


def noise_terms(trajectory: np.ndarray, order_size: float, draws: np.ndarray, case):
    """``-sigma_bin * (x_k / X) * xi_k`` for each bin, in bps.

    ``trajectory`` is the realised inventory path (``N + 1`` levels), so
    ``trajectory[:-1]`` is inventory *before* each bin — the coefficient the
    vendored convention puts on that bin's shock.
    """
    holdings_before_bin = np.asarray(trajectory, dtype=np.float64)[:-1]
    return SELL_SIDE_SIGN * shock_scale_bps(case) * (holdings_before_bin / order_size) * draws


def episode_draws(stream_index: int, n_bins: int, n_episodes: int) -> np.ndarray:
    """The draws the env must have made, regenerated from the seed address alone.

    The env takes randomness only by pool address and consumes exactly one
    standard normal per bin, in order, so ``n_episodes * n_bins`` draws off a
    freshly addressed generator reproduce the whole run. Nothing is read back out
    of the env to build this: if the env drew a different number of shocks, or
    drew them from anywhere else, the reconstruction desynchronises and every
    identity below fails loudly.
    """
    generator = pool_rng(ROOT_SEED, POOL, stream_index)
    return generator.standard_normal(n_episodes * n_bins).reshape(n_episodes, n_bins)


# ---------------------------------------------------------------------------
# The cells: 27 (case, schedule) pairs, the under-trader, the guard case
# ---------------------------------------------------------------------------


def _cells():
    """Every deep cell, plus the two schedules the env has to repair."""
    stream = int(SEEDING["identity_stream"]) + 1000
    cells = []
    for case in NOISE_CASES:
        for schedule in SCHEDULES:
            cells.append((f"{case.case_id}:{schedule}", case, schedule, stream))
            stream += 1
    # 5(d)'s under-trader: genuinely force-liquidated, so the realised schedule
    # is not the planned one and the functional has to follow what was executed.
    cells.append((f"{NOISE_CASES[0].case_id}:undertrade", NOISE_CASES[0], "undertrade", stream))
    stream += 1
    # Task 4's asymptote guard: a named baseline whose trajectory does not end at
    # a hard zero. Not a golden — see the config's `guard_case` block.
    guard = guard_case()
    cells.append((f"{guard.case_id}:{guard.schedule}", guard, "asymptote", stream))
    return cells


CELLS = _cells()
CELL_IDS = [name for name, _, _, _ in CELLS]


def _policy(case, schedule: str):
    if schedule == "undertrade":
        return UnderTradePolicy(case.market, case.order_size, UNDERTRADE_FRACTION)
    if schedule == "asymptote":
        return asymptote_policy(case)
    return baseline(schedule, case.market, case.order_size, case.lambda_risk)


@pytest.fixture(params=CELLS, ids=CELL_IDS)
def cell(request):
    """One cell's episodes, with the draws regenerated independently beside them."""
    name, case, schedule, stream = request.param
    env = build_env(case, stream)
    env.reset(seed=stream)
    results = [run_episode(env, _policy(case, schedule)) for _ in range(EPISODES)]
    draws = episode_draws(stream, case.market.n_bins, EPISODES)
    return name, case, results, draws


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------


def test_the_residual_is_exactly_the_noise_functional(cell):
    """``C - E[cost] == noise(xi)``, every episode, to round-off.

    This is the whole module. The left-hand side is the env's realised cost minus
    the oracle's expected cost for the schedule the env realised; the right-hand
    side is built here out of sigma, tau, that schedule and the draws the seed
    address resolves to. Nothing on the right came from the env.

    Green here means the variance leg of the differential is no longer sampled:
    a known linear form in iid standard normals has an arithmetic variance, and
    it is ``sigma_bin^2 * sum_k x_k^2``.
    """
    name, case, results, draws = cell
    worst = 0.0
    for index, result in enumerate(results):
        terms = noise_terms(result.trajectory, case.order_size, draws[index], case)
        expected = schedule_moments(result.trajectory, case.market).expected
        scale = abs(result.cost_bps) + abs(expected) + float(np.sum(np.abs(terms)))
        assert_identity(
            result.cost_bps - expected,
            float(np.sum(terms)),
            scale=scale,
            rtol=NOISE_RTOL,
            what=f"{name} episode {index}: realised cost less E[cost] vs the noise functional",
        )
        worst = max(worst, abs(result.cost_bps - expected - float(np.sum(terms))) / scale)
    _WORST[name] = worst


def test_the_regenerated_draws_are_the_ones_the_env_used(cell):
    """The reconstruction is not an approximation — it is the same shock sequence.

    The env publishes the *cumulative* shock each bin executed against, so
    differencing it and dividing by ``sigma_bin`` recovers the per-bin draws by a
    completely different route than addressing the pool. That the two agree is
    what licenses the identity above calling its ``xi`` "the draws the env used",
    and it simultaneously pins that the env consumes exactly one draw per bin:
    a hidden extra draw anywhere would slide the whole reconstruction by one.
    """
    _, case, results, draws = cell
    scale = shock_scale_bps(case)
    for index, result in enumerate(results):
        from_walk = np.diff(np.concatenate(([0.0], result.walks))) / scale
        assert from_walk == pytest.approx(draws[index], rel=1e-12, abs=1e-12), (
            f"episode {index}: the shocks the env published are not the shocks its "
            "seed address resolves to"
        )


def test_the_assembled_scale_matches_the_units_contract(cell):
    """The rebuilt ``sigma_bin`` agrees with the oracle's — a check, not a source."""
    _, case, _, _ = cell
    assert shock_scale_bps(case) == pytest.approx(case.market.sigma_bin * BPS, rel=1e-15)


# ---------------------------------------------------------------------------
# The convention this module is in, and what the other one would cost
# ---------------------------------------------------------------------------


def test_the_post_bin_convention_is_the_named_off_by_one(cell):
    """Choosing post-bin holdings drops exactly one bin of variance on the order.

    Algebraic, not statistical: with ``x_0 = X`` and ``x_N = 0``,

    .. code::

        sum_{k=0}^{N-1} (x_k/X)^2  -  sum_{k=1}^{N} (x_k/X)^2  =  1

    identically, for every schedule. So the textbook post-bin convention — the one
    M1a's brief writes the functional in — understates ``V`` by ``sigma_bin^2``
    however the schedule is shaped. For TWAP at N = 13 the missing term is 20.6 %
    of the correct ``V`` (equivalently the correct ``V`` is 26 % larger than the
    post-bin one, which is the figure the parent brief quotes). This is the
    off-by-one-in ``sum x_k^2`` class M0 flagged and the deep tier was sized
    against; the test records what it is worth on this cell rather than leaving
    it as prose.
    """
    _, case, results, _ = cell
    trajectory = results[0].trajectory / case.order_size
    before = float(np.sum(trajectory[:-1] ** 2))
    after = float(np.sum(trajectory[1:] ** 2))

    assert before - after == pytest.approx(1.0, rel=1e-12), (
        "x_0 = X or x_N = 0 does not hold for the realised schedule; the algebraic "
        "gap between the two conventions is not one bin"
    )
    # Expressed against `before` — the correct denominator — because a
    # front-loaded schedule can drive `after` to zero, and the question is what
    # fraction of the real variance the other convention would drop.
    assert (before - after) / before > 0.01, (
        f"{case.case_id}: the two conventions differ by only "
        f"{100.0 * (before - after) / before:.2f}% of V, so this cell would not "
        "distinguish them — the deep tier's bands assume they are far apart"
    )
